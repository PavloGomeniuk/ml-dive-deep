use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use web_sys::CanvasRenderingContext2d;

use crate::combat::{
    click_disambiguation, player_damage_roll, enemy_damage_roll,
    warrior_cleave, magician_frost_nova,
    ClickAction, HitStop, ScreenShake,
};
use crate::dungeon::DungeonMap;
use crate::entities::{Enemy, EnemyState, Player, PlayerClass};
use crate::items::{Item, nearest_item, spawn_loot};
use crate::particles::ParticlePool;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum GamePhase {
    Title,
    CharacterSelect,
    Playing,
    Dead,
}

pub struct DamageNumber {
    pub value: u32,
    pub x: f32,
    pub y: f32,
    pub life: u8,
}

/// Flash text shown at the bottom of the HUD (e.g. "nothing to equip").
pub struct HudFlash {
    pub text: &'static str,
    pub life: u8,  // counts down from 60
}

pub struct GameState {
    pub ctx: CanvasRenderingContext2d,
    pub rng: u32,
    pub phase: GamePhase,
    pub player: Player,
    pub enemies: Vec<Enemy>,
    pub items_on_floor: Vec<Item>,
    pub particles: ParticlePool,
    pub damage_numbers: Vec<DamageNumber>,
    pub dungeon: DungeonMap,
    pub hit_stop: HitStop,
    pub screen_shake: ScreenShake,
    pub last_frame_ms: f32,
    pub gold: u32,
    /// 16→0 counts down; room switch happens at frame 8.
    pub transition_fade: u8,
    /// Target room for the in-progress transition.
    pub pending_transition: Option<usize>,
    /// Which character class card is highlighted (0=Warrior, 1=Magician).
    pub char_selected: Option<usize>,
    /// HUD flash message.
    pub hud_flash: Option<HudFlash>,
}

impl GameState {
    pub fn new() -> Result<Self, JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let document = window.document().ok_or("no document")?;
        let canvas = document
            .get_element_by_id("canvas")
            .ok_or("no #canvas element")?
            .dyn_into::<web_sys::HtmlCanvasElement>()?;
        let ctx = canvas
            .get_context("2d")?
            .ok_or("no 2d context")?
            .dyn_into::<CanvasRenderingContext2d>()?;

        let seed_f = js_sys::Math::random();
        let rng = ((seed_f * u32::MAX as f64) as u32).max(1);

        let dungeon = DungeonMap::new();
        let enemies = dungeon.enemy_cache[0].iter().map(|e| Enemy::new(e.x, e.y)).collect();

        Ok(GameState {
            ctx,
            rng,
            phase: GamePhase::Title,
            player: Player::new(400.0, 200.0),
            enemies,
            items_on_floor: Vec::new(),
            particles: ParticlePool::new(),
            damage_numbers: Vec::new(),
            dungeon,
            hit_stop: HitStop::default(),
            screen_shake: ScreenShake::default(),
            last_frame_ms: 0.0,
            gold: 0,
            transition_fade: 0,
            pending_transition: None,
            char_selected: None,
            hud_flash: None,
        })
    }

    pub fn tick(&mut self, dt: f32) {
        match self.phase {
            GamePhase::Title | GamePhase::CharacterSelect => {
                crate::renderer::draw(self);
            }
            GamePhase::Dead => {
                crate::renderer::draw(self);
            }
            GamePhase::Playing => {
                self.tick_playing(dt);
            }
        }
    }

    fn tick_playing(&mut self, dt: f32) {
        let start = js_sys::Date::now();
        let dt = dt.min(0.05);

        // Tick ability cooldown
        self.player.ability_cooldown = (self.player.ability_cooldown - dt).max(0.0);

        // Tick HUD flash
        if let Some(ref mut flash) = self.hud_flash {
            if flash.life == 0 {
                self.hud_flash = None;
            } else {
                flash.life -= 1;
            }
        }

        // Cooldowns always tick even during hit-stop
        self.player.attack_cooldown = (self.player.attack_cooldown - dt).max(0.0);
        for enemy in self.enemies.iter_mut() {
            if enemy.alive {
                enemy.attack_cooldown = (enemy.attack_cooldown - dt).max(0.0);
                enemy.frozen_timer = (enemy.frozen_timer - dt).max(0.0);
            }
        }

        // Room transition in progress
        if self.transition_fade > 0 {
            self.transition_fade -= 1;
            if self.transition_fade == 8 {
                if let Some(target) = self.pending_transition {
                    self.do_room_switch(target);
                }
            }
            if self.transition_fade == 0 {
                self.pending_transition = None;
            }
            crate::renderer::draw(self);
            return;
        }

        // Hit-stop: skip entity movement/AI but still render
        if self.hit_stop.tick() {
            self.screen_shake.update(&mut self.rng);
            self.particles.update(dt);
            update_damage_numbers(&mut self.damage_numbers);
            crate::renderer::draw(self);
            return;
        }

        self.screen_shake.update(&mut self.rng);
        update_player(self, dt);
        update_enemies(self, dt);
        update_items(&mut self.items_on_floor);
        self.particles.update(dt);
        update_damage_numbers(&mut self.damage_numbers);

        let elapsed = (js_sys::Date::now() - start) as f32;
        self.last_frame_ms = elapsed;

        crate::renderer::draw(self);
    }

    pub fn on_click(&mut self, x: f32, y: f32) {
        match self.phase {
            GamePhase::Title => {
                if crate::title::hit_test_new_game(x, y) {
                    self.phase = GamePhase::CharacterSelect;
                    self.char_selected = None;
                }
            }
            GamePhase::CharacterSelect => {
                let hit_warrior = crate::character_select::hit_test_card(
                    x, y, crate::character_select::WARRIOR_CARD,
                );
                let hit_magician = crate::character_select::hit_test_card(
                    x, y, crate::character_select::MAGICIAN_CARD,
                );
                if hit_warrior {
                    if self.char_selected == Some(0) {
                        self.start_playing(PlayerClass::Warrior);
                    } else {
                        self.char_selected = Some(0);
                    }
                } else if hit_magician {
                    if self.char_selected == Some(1) {
                        self.start_playing(PlayerClass::Magician);
                    } else {
                        self.char_selected = Some(1);
                    }
                }
            }
            GamePhase::Playing => {
                if self.transition_fade > 0 {
                    return;
                }
                match click_disambiguation(x, y, &self.enemies) {
                    ClickAction::Attack(i) => {
                        self.player.attack_target = Some(i);
                        self.player.moving = false;
                    }
                    ClickAction::Move(tx, ty) => {
                        self.player.move_target_x = tx;
                        self.player.move_target_y = ty;
                        self.player.moving = true;
                        self.player.attack_target = None;
                    }
                }
            }
            GamePhase::Dead => {
                self.phase = GamePhase::Title;
            }
        }
    }

    pub fn on_key(&mut self, key: &str) {
        match self.phase {
            GamePhase::Title => { /* ESC = no-op on title */ }
            GamePhase::CharacterSelect => {
                match key {
                    "Escape" => {
                        self.phase = GamePhase::Title;
                        self.char_selected = None;
                    }
                    "Enter" => {
                        if let Some(sel) = self.char_selected {
                            let class = if sel == 0 {
                                PlayerClass::Warrior
                            } else {
                                PlayerClass::Magician
                            };
                            self.start_playing(class);
                        }
                    }
                    _ => {}
                }
            }
            GamePhase::Playing => {
                match key {
                    "Escape" => {
                        self.phase = GamePhase::Title;
                    }
                    "e" | "E" => {
                        let px = self.player.x;
                        let py = self.player.y;
                        if let Some(idx) = nearest_item(&self.items_on_floor, px, py) {
                            let item = self.items_on_floor.remove(idx);
                            let slot = item.kind.slot();
                            self.player.equipment[slot] = Some(item.kind);
                            // Play equip sound via JS (see game.js)
                            let _ = js_sys::eval("window.__playEquipSound && window.__playEquipSound()");
                        } else {
                            self.hud_flash = Some(HudFlash {
                                text: "nothing to equip",
                                life: 60,
                            });
                        }
                    }
                    " " | "Space" => {
                        if self.player.ability_cooldown <= 0.0 {
                            use_ability(self);
                        }
                    }
                    _ => {}
                }
            }
            GamePhase::Dead => {
                if key == "Escape" || key == "Enter" {
                    self.phase = GamePhase::Title;
                }
            }
        }
    }

    fn start_playing(&mut self, class: PlayerClass) {
        let room = &self.dungeon.rooms[0];
        let spawn_x = room.floor_x() + room.floor_w() / 2.0;
        let spawn_y = room.floor_y() + room.floor_h() / 2.0;

        self.player = Player::new_with_class(spawn_x, spawn_y, class);
        self.dungeon.current_room = 0;
        // Reload room 0 enemies from cache (fresh copies)
        self.enemies = self.dungeon.enemy_cache[0]
            .iter()
            .map(|e| Enemy::new(e.home_x, e.home_y))
            .collect();
        // Reset room 1 enemies too
        self.dungeon.enemy_cache[1] = vec![
            Enemy::new(250.0, 120.0),
            Enemy::new(550.0, 200.0),
            Enemy::new(380.0, 280.0),
            Enemy::new(480.0, 100.0),
        ];
        self.items_on_floor.clear();
        self.gold = 0;
        self.particles = ParticlePool::new();
        self.damage_numbers.clear();
        self.transition_fade = 0;
        self.pending_transition = None;
        self.hud_flash = None;
        self.phase = GamePhase::Playing;
    }

    fn do_room_switch(&mut self, target: usize) {
        // Save current enemies back to cache
        let cur = self.dungeon.current_room;
        self.dungeon.enemy_cache[cur] = self.enemies.drain(..).collect();

        // Switch room
        self.dungeon.switch_room(target);

        // Load target room enemies
        self.enemies = self.dungeon.enemy_cache[target]
            .iter()
            .map(|e| Enemy { ..*e })
            .collect();

        // Reposition player at entry spawn
        let (sx, sy) = self.dungeon.current_room().entry_spawn();
        self.player.x = sx;
        self.player.y = sy;
        self.player.move_target_x = sx;
        self.player.move_target_y = sy;
        self.player.moving = false;
        self.player.attack_target = None;

        // Clear floor items (they stay in the room they were dropped)
        self.items_on_floor.clear();

        // Play transition sound
        let _ = js_sys::eval("window.__playTransitionSound && window.__playTransitionSound()");
    }
}

fn update_items(items: &mut Vec<Item>) {
    for item in items.iter_mut() {
        if item.label_life > 0 {
            item.label_life -= 1;
        }
    }
}

fn update_damage_numbers(nums: &mut Vec<DamageNumber>) {
    for dn in nums.iter_mut() {
        dn.y -= 2.0;
        dn.life = dn.life.saturating_sub(1);
    }
    nums.retain(|dn| dn.life > 0);
}

fn use_ability(gs: &mut GameState) {
    let px = gs.player.x;
    let py = gs.player.y;
    let cooldown = gs.player.class.ability_cooldown_max();
    gs.player.ability_cooldown = cooldown;

    match gs.player.class {
        PlayerClass::Warrior => {
            let bonus = gs.player.damage_bonus();
            let hits = warrior_cleave(px, py, &gs.enemies, &mut gs.rng, bonus);
            for (idx, dmg) in hits {
                let ex = gs.enemies[idx].x;
                let ey = gs.enemies[idx].y;
                gs.enemies[idx].hp -= dmg;
                gs.damage_numbers.push(DamageNumber {
                    value: dmg as u32,
                    x: ex,
                    y: ey - 20.0,
                    life: 30,
                });
                gs.particles.spawn_burst(ex, ey, 6, &mut gs.rng);
                if gs.enemies[idx].hp <= 0.0 {
                    gs.enemies[idx].alive = false;
                    gs.enemies[idx].hp = 0.0;
                    gs.gold += 10;
                    let drop_x = ex + (crate::combat::xorshift(&mut gs.rng) % 20) as f32 - 10.0;
                    let drop_y = ey + (crate::combat::xorshift(&mut gs.rng) % 20) as f32 - 10.0;
                    if crate::combat::xorshift(&mut gs.rng) % 2 == 0 {
                        gs.items_on_floor.push(spawn_loot(drop_x, drop_y, &mut gs.rng));
                        gs.particles.spawn_burst(drop_x, drop_y, 8, &mut gs.rng);
                    }
                }
            }
            gs.screen_shake.trigger();
        }
        PlayerClass::Magician => {
            const FREEZE_DURATION: f32 = 2.0;
            let frozen = magician_frost_nova(px, py, &gs.enemies);
            for idx in frozen {
                gs.enemies[idx].frozen_timer = FREEZE_DURATION;
            }
            gs.particles.spawn_burst(px, py, 12, &mut gs.rng);
        }
    }
}

fn update_player(gs: &mut GameState, dt: f32) {
    const MOVE_SPEED: f32 = 120.0;
    const ATTACK_RANGE: f32 = 50.0;
    const ATTACK_COOLDOWN: f32 = 0.8;

    if let Some(target_idx) = gs.player.attack_target {
        if target_idx >= gs.enemies.len() || !gs.enemies[target_idx].alive {
            gs.player.attack_target = None;
            return;
        }

        let ex = gs.enemies[target_idx].x;
        let ey = gs.enemies[target_idx].y;
        let dist = crate::combat::distance(gs.player.x, gs.player.y, ex, ey);

        if dist > ATTACK_RANGE {
            let dx = ex - gs.player.x;
            let dy = ey - gs.player.y;
            let len = (dx * dx + dy * dy).sqrt();
            gs.player.x += (dx / len) * MOVE_SPEED * dt;
            gs.player.y += (dy / len) * MOVE_SPEED * dt;
        } else if gs.player.attack_cooldown <= 0.0 {
            let class = gs.player.class;
            let bonus = gs.player.damage_bonus();
            let dmg = player_damage_roll(&mut gs.rng, class, bonus);
            gs.enemies[target_idx].hp -= dmg;

            gs.damage_numbers.push(DamageNumber {
                value: dmg as u32,
                x: ex,
                y: ey - 20.0,
                life: 30,
            });

            gs.particles.spawn_burst(ex, ey, 8, &mut gs.rng);
            gs.hit_stop.trigger();
            gs.screen_shake.trigger();
            gs.player.attack_cooldown = ATTACK_COOLDOWN;

            // Play hit sound
            let _ = js_sys::eval("window.__playHitSound && window.__playHitSound()");

            if gs.enemies[target_idx].hp <= 0.0 {
                gs.enemies[target_idx].alive = false;
                gs.enemies[target_idx].hp = 0.0;
                gs.player.attack_target = None;
                gs.gold += 10;

                let drop_x = ex + (crate::combat::xorshift(&mut gs.rng) % 20) as f32 - 10.0;
                let drop_y = ey + (crate::combat::xorshift(&mut gs.rng) % 20) as f32 - 10.0;
                if crate::combat::xorshift(&mut gs.rng) % 2 == 0 {
                    gs.items_on_floor.push(spawn_loot(drop_x, drop_y, &mut gs.rng));
                    gs.particles.spawn_burst(drop_x, drop_y, 8, &mut gs.rng);
                }

                // Play death sound
                let _ = js_sys::eval("window.__playDeathSound && window.__playDeathSound()");
            }
        }
    } else if gs.player.moving {
        let dx = gs.player.move_target_x - gs.player.x;
        let dy = gs.player.move_target_y - gs.player.y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 4.0 {
            gs.player.moving = false;
        } else {
            gs.player.x += (dx / dist) * MOVE_SPEED * dt;
            gs.player.y += (dy / dist) * MOVE_SPEED * dt;
        }
    }

    // Check door trigger BEFORE floor clamp
    if gs.transition_fade == 0 && gs.pending_transition.is_none() {
        if let Some(target) = gs.dungeon.check_transition(gs.player.x, gs.player.y) {
            gs.transition_fade = 16;
            gs.pending_transition = Some(target);
            gs.player.moving = false;
            gs.player.attack_target = None;
            return;
        }
    }

    // Clamp player to floor
    let (fx, fy, fw, fh) = {
        let room = gs.dungeon.current_room();
        (room.floor_x(), room.floor_y(), room.floor_w(), room.floor_h())
    };
    gs.player.x = gs.player.x.clamp(fx + 10.0, fx + fw - 10.0);
    gs.player.y = gs.player.y.clamp(fy + 10.0, fy + fh - 10.0);
}

fn update_enemies(gs: &mut GameState, dt: f32) {
    const AGGRO_RADIUS: f32 = 200.0;
    const MELEE_RANGE: f32 = 40.0;
    const MOVE_SPEED: f32 = 60.0;
    const ATTACK_COOLDOWN: f32 = 1.5;
    const LEASH_RADIUS: f32 = 400.0;

    let px = gs.player.x;
    let py = gs.player.y;

    for i in 0..gs.enemies.len() {
        if !gs.enemies[i].alive {
            continue;
        }

        // Frozen enemies don't move or attack
        if gs.enemies[i].frozen_timer > 0.0 {
            continue;
        }

        let ex = gs.enemies[i].x;
        let ey = gs.enemies[i].y;
        let dist_to_player = crate::combat::distance(ex, ey, px, py);
        let dist_to_home = crate::combat::distance(ex, ey, gs.enemies[i].home_x, gs.enemies[i].home_y);

        let new_state = match gs.enemies[i].state {
            EnemyState::Patrolling => {
                if dist_to_player <= AGGRO_RADIUS { EnemyState::Chasing } else { EnemyState::Patrolling }
            }
            EnemyState::Chasing => {
                if dist_to_player > LEASH_RADIUS { EnemyState::Patrolling }
                else if dist_to_player <= MELEE_RANGE { EnemyState::Attacking }
                else { EnemyState::Chasing }
            }
            EnemyState::Attacking => {
                if dist_to_player > LEASH_RADIUS { EnemyState::Patrolling }
                else if dist_to_player > MELEE_RANGE + 10.0 { EnemyState::Chasing }
                else { EnemyState::Attacking }
            }
        };
        gs.enemies[i].state = new_state;

        match gs.enemies[i].state {
            EnemyState::Patrolling => {
                if dist_to_home > 5.0 {
                    let hx = gs.enemies[i].home_x;
                    let hy = gs.enemies[i].home_y;
                    let dx = hx - ex;
                    let dy = hy - ey;
                    let len = (dx * dx + dy * dy).sqrt();
                    gs.enemies[i].x += (dx / len) * MOVE_SPEED * 0.3 * dt;
                    gs.enemies[i].y += (dy / len) * MOVE_SPEED * 0.3 * dt;
                }
            }
            EnemyState::Chasing => {
                let dx = px - ex;
                let dy = py - ey;
                let len = (dx * dx + dy * dy).sqrt();
                if len > 0.1 {
                    gs.enemies[i].x += (dx / len) * MOVE_SPEED * dt;
                    gs.enemies[i].y += (dy / len) * MOVE_SPEED * dt;
                }
            }
            EnemyState::Attacking => {
                if gs.enemies[i].attack_cooldown <= 0.0 {
                    let dmg = enemy_damage_roll(&mut gs.rng);
                    gs.player.hp -= dmg;
                    gs.enemies[i].attack_cooldown = ATTACK_COOLDOWN;

                    gs.damage_numbers.push(DamageNumber {
                        value: dmg as u32,
                        x: px,
                        y: py - 20.0,
                        life: 30,
                    });

                    if gs.player.hp <= 0.0 {
                        gs.player.hp = 0.0;
                        gs.phase = GamePhase::Dead;
                    }
                }
            }
        }
    }
}
