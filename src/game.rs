use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use web_sys::CanvasRenderingContext2d;

use crate::combat::{click_disambiguation, player_damage_roll, enemy_damage_roll, ClickAction, HitStop, ScreenShake};
use crate::dungeon::Dungeon;
use crate::entities::{Enemy, EnemyState, Player};
use crate::particles::ParticlePool;

pub struct DamageNumber {
    pub value: u32,
    pub x: f32,
    pub y: f32,
    pub life: u8,
}

pub struct GameState {
    pub ctx: CanvasRenderingContext2d,
    pub rng: u32,
    pub player: Player,
    pub enemies: Vec<Enemy>,
    pub particles: ParticlePool,
    pub damage_numbers: Vec<DamageNumber>,
    pub dungeon: Dungeon,
    pub hit_stop: HitStop,
    pub screen_shake: ScreenShake,
    pub last_frame_ms: f32,
    pub game_over: bool,
    pub gold: u32,
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

        // Seed RNG from JS Math.random (safe in WASM, avoids getrandom)
        let seed_f = js_sys::Math::random();
        let rng = ((seed_f * u32::MAX as f64) as u32).max(1);

        let player = Player::new(400.0, 200.0);

        let enemies = vec![
            Enemy::new(200.0, 150.0),
            Enemy::new(600.0, 150.0),
            Enemy::new(400.0, 300.0),
        ];

        Ok(GameState {
            ctx,
            rng,
            player,
            enemies,
            particles: ParticlePool::new(),
            damage_numbers: Vec::new(),
            dungeon: Dungeon::new_single_room(),
            hit_stop: HitStop::default(),
            screen_shake: ScreenShake::default(),
            last_frame_ms: 0.0,
            game_over: false,
            gold: 0,
        })
    }

    pub fn tick(&mut self, dt: f32) {
        if self.game_over {
            crate::renderer::draw(self);
            return;
        }

        let start = js_sys::Date::now();

        let dt = dt.min(0.05);

        // Cooldowns always tick even during hit-stop
        self.player.attack_cooldown = (self.player.attack_cooldown - dt).max(0.0);
        for enemy in self.enemies.iter_mut() {
            if enemy.alive {
                enemy.attack_cooldown = (enemy.attack_cooldown - dt).max(0.0);
            }
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
        self.particles.update(dt);
        update_damage_numbers(&mut self.damage_numbers);

        let elapsed = (js_sys::Date::now() - start) as f32;
        self.last_frame_ms = elapsed;

        crate::renderer::draw(self);
    }

    pub fn on_click(&mut self, x: f32, y: f32) {
        if self.game_over {
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
}

fn update_damage_numbers(nums: &mut Vec<DamageNumber>) {
    for dn in nums.iter_mut() {
        dn.y -= 2.0; // float upward
        dn.life = dn.life.saturating_sub(1);
    }
    nums.retain(|dn| dn.life > 0);
}

fn update_player(gs: &mut GameState, dt: f32) {
    const MOVE_SPEED: f32 = 120.0;
    const ATTACK_RANGE: f32 = 50.0;
    const ATTACK_COOLDOWN: f32 = 0.8;

    if let Some(target_idx) = gs.player.attack_target {
        // Check target still alive
        if target_idx >= gs.enemies.len() || !gs.enemies[target_idx].alive {
            gs.player.attack_target = None;
            return;
        }

        let ex = gs.enemies[target_idx].x;
        let ey = gs.enemies[target_idx].y;
        let dist = crate::combat::distance(gs.player.x, gs.player.y, ex, ey);

        if dist > ATTACK_RANGE {
            // Move toward enemy
            let dx = ex - gs.player.x;
            let dy = ey - gs.player.y;
            let len = (dx * dx + dy * dy).sqrt();
            gs.player.x += (dx / len) * MOVE_SPEED * dt;
            gs.player.y += (dy / len) * MOVE_SPEED * dt;
        } else if gs.player.attack_cooldown <= 0.0 {
            // Attack
            let dmg = player_damage_roll(&mut gs.rng);
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

            if gs.enemies[target_idx].hp <= 0.0 {
                gs.enemies[target_idx].alive = false;
                gs.enemies[target_idx].hp = 0.0;
                gs.player.attack_target = None;
                gs.gold += 10;
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

    // Clamp player to floor
    let room = &gs.dungeon.room;
    gs.player.x = gs.player.x.clamp(room.floor_x() + 10.0, room.floor_x() + room.floor_w() - 10.0);
    gs.player.y = gs.player.y.clamp(room.floor_y() + 10.0, room.floor_y() + room.floor_h() - 10.0);
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

        let ex = gs.enemies[i].x;
        let ey = gs.enemies[i].y;
        let dist_to_player = crate::combat::distance(ex, ey, px, py);
        let dist_to_home = crate::combat::distance(ex, ey, gs.enemies[i].home_x, gs.enemies[i].home_y);

        let new_state = match gs.enemies[i].state {
            EnemyState::Patrolling => {
                if dist_to_player <= AGGRO_RADIUS {
                    EnemyState::Chasing
                } else {
                    EnemyState::Patrolling
                }
            }
            EnemyState::Chasing => {
                if dist_to_player > LEASH_RADIUS {
                    EnemyState::Patrolling
                } else if dist_to_player <= MELEE_RANGE {
                    EnemyState::Attacking
                } else {
                    EnemyState::Chasing
                }
            }
            EnemyState::Attacking => {
                if dist_to_player > LEASH_RADIUS {
                    EnemyState::Patrolling
                } else if dist_to_player > MELEE_RANGE + 10.0 {
                    EnemyState::Chasing
                } else {
                    EnemyState::Attacking
                }
            }
        };
        gs.enemies[i].state = new_state;

        match gs.enemies[i].state {
            EnemyState::Patrolling => {
                // Drift back toward home
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
                        gs.game_over = true;
                    }
                }
            }
        }
    }
}
