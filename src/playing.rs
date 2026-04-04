use bevy::prelude::*;
use crate::state::GameState;
use crate::components::*;
use crate::resources::*;
use crate::dungeon::DungeonMap;
use crate::entities::{PlayerClass, EnemyState, ItemKind};
use crate::combat::{xorshift, player_damage_roll, enemy_damage_roll, distance as cdist};

pub struct PlayingPlugin;
impl Plugin for PlayingPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_event::<DamageEvent>()
            .insert_resource(DungeonRes(DungeonMap::new()))
            .add_systems(OnEnter(GameState::Playing), (setup_playing, setup_dungeon_visuals))
            .add_systems(Update, (
                player_click_input,
                player_keyboard_input,
                tick_cooldowns,
                player_movement,
                enemy_ai,
                player_auto_attack,
                apply_damage,
                check_deaths,
                damage_number_tick,
                particle_tick,
                screen_shake_tick,
                room_transition_check,
                fade_overlay_tick,
            ).chain().run_if(in_state(GameState::Playing)));
    }
}

// ── Events ────────────────────────────────────────────────────────────────────

#[derive(Event)]
pub struct DamageEvent {
    pub target: Entity,
    pub amount: f32,
    pub from_player: bool,
}

// ── Resource wrapper ──────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct DungeonRes(pub DungeonMap);

// ── Colours ───────────────────────────────────────────────────────────────────

const C_BG: Color       = Color::srgb(0.039, 0.039, 0.063);
const C_WALL: Color     = Color::srgb(0.102, 0.102, 0.180);
const C_FLOOR: Color    = Color::srgb(0.118, 0.118, 0.118);
const C_GRID: Color     = Color::srgba(1.0, 1.0, 1.0, 0.02);
const C_PLAYER_BODY: Color = Color::srgb(0.784, 0.722, 0.604);
const C_ENEMY_BODY: Color  = Color::srgb(0.533, 0.533, 0.667);
const C_ENEMY_SKULL: Color = Color::srgb(0.800, 0.800, 0.933);
const C_HUD_BG: Color   = Color::srgb(0.039, 0.039, 0.078);
const C_HUD_BORDER: Color  = Color::srgb(0.353, 0.227, 0.082);
const C_GOLD: Color     = Color::srgb(0.831, 0.686, 0.216);
const C_HP_BAR: Color   = Color::srgb(0.800, 0.133, 0.133);
const C_HP_BG: Color    = Color::srgb(0.200, 0.000, 0.000);
const C_MP_BAR: Color   = Color::srgb(0.133, 0.267, 0.800);
const C_MP_BG: Color    = Color::srgb(0.000, 0.067, 0.200);
const C_ABILITY_CD: Color  = Color::srgb(0.600, 0.500, 0.200);
const C_ABILITY_RDY: Color = Color::srgb(0.831, 0.686, 0.216);
const C_DAMAGE: Color   = Color::srgb(1.0, 0.933, 0.267);
const C_PARTICLE: Color = Color::srgb(0.800, 0.133, 0.133);
const C_DOOR: Color     = Color::srgb(0.180, 0.133, 0.027);

// Enemy colours by type
const C_WARRIOR_BODY: Color  = Color::srgb(0.784, 0.722, 0.604);
const C_MAGICIAN_BODY: Color = Color::srgb(0.600, 0.533, 0.800);

// ── Setup ─────────────────────────────────────────────────────────────────────

fn setup_dungeon_visuals(mut commands: Commands, dungeon: Res<DungeonRes>) {
    let room = dungeon.0.current();

    // Full background
    commands.spawn((
        Sprite {
            color: C_BG,
            custom_size: Some(Vec2::new(SCREEN_W, SCREEN_H)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, Z_BG),
        StateScoped(GameState::Playing),
    ));

    // Room wall (outer)
    commands.spawn((
        Sprite {
            color: C_WALL,
            custom_size: Some(Vec2::new(room.w, room.h)),
            ..default()
        },
        Transform::from_xyz(room.cx, room.cy, Z_BG + 0.1),
        StateScoped(GameState::Playing),
    ));

    // Floor
    commands.spawn((
        Sprite {
            color: C_FLOOR,
            custom_size: Some(Vec2::new(room.floor_w(), room.floor_h())),
            ..default()
        },
        Transform::from_xyz(room.cx, room.cy, Z_FLOOR),
        DungeonFloor,
        StateScoped(GameState::Playing),
    ));

    // Door archway (coloured rectangle)
    let (door_x, door_y) = room.exit_door();
    commands.spawn((
        Sprite {
            color: C_DOOR,
            custom_size: Some(Vec2::new(40.0, 20.0)),
            ..default()
        },
        Transform::from_xyz(door_x, door_y, Z_FLOOR + 0.1),
        StateScoped(GameState::Playing),
    ));

    // ── HUD background ────────────────────────────────────────────────────────
    commands.spawn((
        Sprite {
            color: C_HUD_BG,
            custom_size: Some(Vec2::new(SCREEN_W, HUD_H)),
            ..default()
        },
        Transform::from_xyz(0.0, HUD_CY, Z_HUD),
        StateScoped(GameState::Playing),
    ));

    // HUD top border
    commands.spawn((
        Sprite {
            color: C_HUD_BORDER,
            custom_size: Some(Vec2::new(SCREEN_W, 2.0)),
            ..default()
        },
        Transform::from_xyz(0.0, -150.0, Z_HUD + 0.1),
        StateScoped(GameState::Playing),
    ));

    // HP bar background
    commands.spawn((
        Sprite {
            color: C_HP_BG,
            custom_size: Some(Vec2::new(200.0, 16.0)),
            ..default()
        },
        Transform::from_xyz(-200.0, HUD_CY + 20.0, Z_HUD + 0.2),
        StateScoped(GameState::Playing),
    ));

    // HP bar fill (updated each frame)
    commands.spawn((
        Sprite {
            color: C_HP_BAR,
            custom_size: Some(Vec2::new(200.0, 16.0)),
            ..default()
        },
        Transform::from_xyz(-200.0, HUD_CY + 20.0, Z_HUD + 0.3),
        HpBarFill,
        StateScoped(GameState::Playing),
    ));

    // HP label
    commands.spawn((
        Text2d::new("HP"),
        TextFont { font_size: 11.0, ..default() },
        TextColor(Color::srgb(0.800, 0.267, 0.267)),
        Transform::from_xyz(-308.0, HUD_CY + 20.0, Z_HUD_TEXT),
        StateScoped(GameState::Playing),
    ));

    // MP bar background
    commands.spawn((
        Sprite {
            color: C_MP_BG,
            custom_size: Some(Vec2::new(200.0, 16.0)),
            ..default()
        },
        Transform::from_xyz(-200.0, HUD_CY, Z_HUD + 0.2),
        StateScoped(GameState::Playing),
    ));

    // MP bar fill
    commands.spawn((
        Sprite {
            color: C_MP_BAR,
            custom_size: Some(Vec2::new(200.0, 16.0)),
            ..default()
        },
        Transform::from_xyz(-200.0, HUD_CY, Z_HUD + 0.3),
        MpBarFill,
        StateScoped(GameState::Playing),
    ));

    // MP label
    commands.spawn((
        Text2d::new("MP"),
        TextFont { font_size: 11.0, ..default() },
        TextColor(Color::srgb(0.267, 0.400, 0.800)),
        Transform::from_xyz(-308.0, HUD_CY, Z_HUD_TEXT),
        StateScoped(GameState::Playing),
    ));

    // Ability bar background
    commands.spawn((
        Sprite {
            color: Color::srgb(0.15, 0.15, 0.20),
            custom_size: Some(Vec2::new(100.0, 16.0)),
            ..default()
        },
        Transform::from_xyz(0.0, HUD_CY + 20.0, Z_HUD + 0.2),
        StateScoped(GameState::Playing),
    ));

    // Ability bar fill
    commands.spawn((
        Sprite {
            color: C_ABILITY_RDY,
            custom_size: Some(Vec2::new(100.0, 16.0)),
            ..default()
        },
        Transform::from_xyz(0.0, HUD_CY + 20.0, Z_HUD + 0.3),
        AbilityBarFill,
        StateScoped(GameState::Playing),
    ));

    // Gold text
    commands.spawn((
        Text2d::new("0 GP"),
        TextFont { font_size: 16.0, ..default() },
        TextColor(C_GOLD),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(200.0, HUD_CY + 20.0, Z_HUD_TEXT),
        GoldText,
        StateScoped(GameState::Playing),
    ));

    // HUD flash text
    commands.spawn((
        Text2d::new(""),
        TextFont { font_size: 14.0, ..default() },
        TextColor(Color::srgb(1.0, 0.933, 0.267)),
        TextLayout::new_with_justify(JustifyText::Center),
        Transform::from_xyz(0.0, HUD_CY - 25.0, Z_HUD_TEXT),
        HudFlashText,
        StateScoped(GameState::Playing),
    ));
}

fn setup_playing(
    mut commands: Commands,
    selected: Res<SelectedClass>,
    mut dungeon: ResMut<DungeonRes>,
    mut gold: ResMut<Gold>,
    mut run_stats: ResMut<RunStats>,
    mut rng: ResMut<Rng>,
) {
    let class = selected.0.unwrap_or(PlayerClass::Warrior);
    gold.0 = 0;
    *run_stats = RunStats::default();
    dungeon.0 = DungeonMap::new();

    let (spawn_x, spawn_y) = dungeon.0.current().entry_spawn();

    // Spawn player
    let player_color = match class {
        PlayerClass::Warrior => C_WARRIOR_BODY,
        PlayerClass::Magician => C_MAGICIAN_BODY,
    };
    commands.spawn((
        Sprite {
            color: player_color,
            custom_size: Some(Vec2::new(24.0, 40.0)),
            ..default()
        },
        Transform::from_xyz(spawn_x, spawn_y, Z_PLAYER),
        PlayerMarker,
        PlayerClassComp(class),
        Health::new(class.base_hp()),
        Mana::new(class.base_mana()),
        Cooldowns::default(),
        Equipment::default(),
        Potions::default(),
        DamageBonus(0.0),
        MoveTarget(Vec2::new(spawn_x, spawn_y)),
        StateScoped(GameState::Playing),
    ));

    // Spawn enemies for current room
    spawn_room_enemies(&mut commands, &dungeon.0, &mut rng);
}

fn spawn_room_enemies(commands: &mut Commands, dungeon: &DungeonMap, rng: &mut Rng) {
    let room = dungeon.current();
    let positions = [
        (room.cx - 150.0, room.cy + 60.0),
        (room.cx + 150.0, room.cy + 60.0),
        (room.cx + 150.0, room.cy - 60.0),
    ];
    for (ex, ey) in positions {
        commands.spawn((
            Sprite {
                color: C_ENEMY_BODY,
                custom_size: Some(Vec2::new(24.0, 36.0)),
                ..default()
            },
            Transform::from_xyz(ex, ey, Z_ENEMY),
            EnemyMarker,
            Health::new(30.0),
            EnemyAI {
                state: EnemyState::Patrolling,
                home: Vec2::new(ex, ey),
                attack_timer: 0.0,
            },
            StateScoped(GameState::Playing),
        ));
    }
}

// ── Input ─────────────────────────────────────────────────────────────────────

fn player_click_input(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut player_q: Query<(Entity, &Transform, &mut MoveTarget, &mut AttackTarget), (With<PlayerMarker>, Without<EnemyMarker>)>,
    enemy_q: Query<(Entity, &Transform), (With<EnemyMarker>, Without<PlayerMarker>)>,
    mut commands: Commands,
) {
    if !mouse.just_pressed(MouseButton::Left) { return; }
    let Ok(win) = windows.get_single() else { return; };
    let Ok((cam, cam_tf)) = camera_q.get_single() else { return; };
    let Some(cursor) = win.cursor_position() else { return; };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return; };

    // Click in HUD area — ignore
    if world.y < -150.0 { return; }

    // Check if clicking on an enemy
    const ATTACK_R: f32 = 45.0;
    let mut closest: Option<(Entity, f32)> = None;
    for (entity, tf) in &enemy_q {
        let d = tf.translation.truncate().distance(world);
        if d < ATTACK_R {
            if closest.map_or(true, |(_, bd)| d < bd) {
                closest = Some((entity, d));
            }
        }
    }

    for (player_entity, _player_tf, mut move_target, mut attack_target) in &mut player_q {
        if let Some((enemy_entity, _)) = closest {
            *attack_target = AttackTarget(enemy_entity);
            commands.entity(player_entity).insert(AttackTarget(enemy_entity));
        } else {
            move_target.0 = world;
            commands.entity(player_entity).remove::<AttackTarget>();
        }
    }
}

fn player_keyboard_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut player_q: Query<(
        &Transform,
        &mut Cooldowns,
        &PlayerClassComp,
        &Equipment,
        &mut Mana,
        &mut Potions,
        &mut Health,
    ), With<PlayerMarker>>,
    mut enemy_q: Query<(Entity, &Transform, &mut EnemyAI, &mut Health), (With<EnemyMarker>, Without<PlayerMarker>)>,
    mut next_state: ResMut<NextState<GameState>>,
    mut ev_damage: EventWriter<DamageEvent>,
    mut hud_flash: ResMut<HudFlash>,
    mut gold: ResMut<Gold>,
    mut run_stats: ResMut<RunStats>,
    mut rng: ResMut<Rng>,
    mut shake: ResMut<ScreenShakeRes>,
    mut commands: Commands,
) {
    if keys.just_pressed(KeyCode::Escape) {
        next_state.set(GameState::Title);
        return;
    }

    let Ok((player_tf, mut cooldowns, class_comp, equipment, mut mana, mut potions, mut health)) =
        player_q.get_single_mut() else { return; };

    // Potions
    if keys.just_pressed(KeyCode::Digit1) {
        if potions.hp == 0 || health.current >= health.max {
            hud_flash.show("No HP potion!");
        } else {
            potions.hp -= 1;
            health.current = (health.current + 40.0).min(health.max);
        }
    }
    if keys.just_pressed(KeyCode::Digit2) {
        if potions.mp == 0 || mana.current >= mana.max {
            hud_flash.show("No MP potion!");
        } else {
            potions.mp -= 1;
            mana.current = (mana.current + 40.0).min(mana.max);
        }
    }

    // Ability — Space
    if keys.just_pressed(KeyCode::Space) {
        if cooldowns.ability > 0.0 {
            hud_flash.show("Ability on cooldown!");
            return;
        }
        let px = player_tf.translation.x;
        let py = player_tf.translation.y;
        let dmg_bonus = equipment.total_damage_bonus();

        match class_comp.0 {
            PlayerClass::Warrior => {
                // Cleave: AoE 80 px, cone approximated as circle for v1
                const CLEAVE_R: f32 = 80.0;
                for (enemy_entity, enemy_tf, _ai, _hp) in &enemy_q {
                    let d = enemy_tf.translation.truncate().distance(Vec2::new(px, py));
                    if d <= CLEAVE_R {
                        let mut seed = rng.0;
                        let dmg = 15.0 + (xorshift(&mut seed) % 6) as f32 + dmg_bonus;
                        rng.0 = seed;
                        ev_damage.send(DamageEvent { target: enemy_entity, amount: dmg, from_player: true });
                    }
                }
                cooldowns.ability = class_comp.0.ability_cooldown_max();
                shake.trigger(3.0);
            }
            PlayerClass::Magician => {
                // Frost Nova: freeze all enemies within 70 px
                const NOVA_R: f32 = 70.0;
                let mut hit_any = false;
                for (enemy_entity, enemy_tf, mut ai, _hp) in &mut enemy_q {
                    let d = enemy_tf.translation.truncate().distance(Vec2::new(px, py));
                    if d <= NOVA_R {
                        ai.state = EnemyState::Patrolling; // brief state reset
                        commands.entity(enemy_entity).insert(Frozen(2.0));
                        hit_any = true;
                    }
                }
                if hit_any { shake.trigger(2.0); }
                cooldowns.ability = class_comp.0.ability_cooldown_max();
            }
        }
    }
}

// ── Cooldowns & movement ──────────────────────────────────────────────────────

fn tick_cooldowns(
    time: Res<Time>,
    mut player_q: Query<&mut Cooldowns, With<PlayerMarker>>,
    mut frozen_q: Query<(Entity, &mut Frozen)>,
    mut commands: Commands,
) {
    let dt = time.delta_secs();
    for mut cd in &mut player_q {
        cd.attack  = (cd.attack  - dt).max(0.0);
        cd.ability = (cd.ability - dt).max(0.0);
    }
    for (entity, mut frozen) in &mut frozen_q {
        frozen.0 -= dt;
        if frozen.0 <= 0.0 {
            commands.entity(entity).remove::<Frozen>();
        }
    }
}

const PLAYER_SPEED: f32 = 180.0;
const ATTACK_RANGE: f32 = 48.0;

fn player_movement(
    time: Res<Time>,
    mut player_q: Query<(&mut Transform, &MoveTarget, Option<&AttackTarget>), With<PlayerMarker>>,
    enemy_q: Query<&Transform, (With<EnemyMarker>, Without<PlayerMarker>)>,
    dungeon: Res<DungeonRes>,
) {
    let dt = time.delta_secs();
    let Ok((mut tf, move_target, attack_target)) = player_q.get_single_mut() else { return; };

    if let Some(AttackTarget(enemy_entity)) = attack_target {
        if let Ok(enemy_tf) = enemy_q.get(*enemy_entity) {
            let to_enemy = enemy_tf.translation.truncate() - tf.translation.truncate();
            if to_enemy.length() > ATTACK_RANGE - 4.0 {
                let dir = to_enemy.normalize_or_zero();
                let new_pos = tf.translation.truncate() + dir * PLAYER_SPEED * dt;
                tf.translation.x = new_pos.x.clamp(
                    dungeon.0.current().floor_x_min() + 12.0,
                    dungeon.0.current().floor_x_max() - 12.0,
                );
                tf.translation.y = new_pos.y.clamp(
                    dungeon.0.current().floor_y_min() + 20.0,
                    dungeon.0.current().floor_y_max() - 20.0,
                );
            }
        }
        return;
    }

    let to_target = move_target.0 - tf.translation.truncate();
    if to_target.length() > 4.0 {
        let dir = to_target.normalize_or_zero();
        let new_pos = tf.translation.truncate() + dir * PLAYER_SPEED * dt;
        tf.translation.x = new_pos.x.clamp(
            dungeon.0.current().floor_x_min() + 12.0,
            dungeon.0.current().floor_x_max() - 12.0,
        );
        tf.translation.y = new_pos.y.clamp(
            dungeon.0.current().floor_y_min() + 20.0,
            dungeon.0.current().floor_y_max() - 20.0,
        );
    }
}

// ── Enemy AI ──────────────────────────────────────────────────────────────────

const ENEMY_SPEED: f32 = 60.0;
const AGGRO_RADIUS: f32 = 200.0;
const MELEE_RANGE: f32 = 36.0;
const ENEMY_ATTACK_CD: f32 = 1.2;

fn enemy_ai(
    time: Res<Time>,
    player_q: Query<(Entity, &Transform, &Health), With<PlayerMarker>>,
    mut enemy_q: Query<(Entity, &mut Transform, &mut EnemyAI, &Health), (With<EnemyMarker>, Without<PlayerMarker>, Without<Frozen>)>,
    mut ev_damage: EventWriter<DamageEvent>,
    mut rng: ResMut<Rng>,
    mut shake: ResMut<ScreenShakeRes>,
    dungeon: Res<DungeonRes>,
) {
    let dt = time.delta_secs();
    let Ok((player_entity, player_tf, player_health)) = player_q.get_single() else { return; };
    if player_health.is_dead() { return; }

    let px = player_tf.translation.x;
    let py = player_tf.translation.y;
    let floor = dungeon.0.current();

    for (_entity, mut tf, mut ai, _hp) in &mut enemy_q {
        let ex = tf.translation.x;
        let ey = tf.translation.y;
        let d = cdist(ex, ey, px, py);

        match ai.state {
            EnemyState::Patrolling => {
                if d < AGGRO_RADIUS { ai.state = EnemyState::Chasing; }
            }
            EnemyState::Chasing => {
                if d > AGGRO_RADIUS * 1.5 {
                    ai.state = EnemyState::Patrolling;
                } else if d <= MELEE_RANGE {
                    ai.state = EnemyState::Attacking;
                } else {
                    let dir = Vec2::new(px - ex, py - ey).normalize_or_zero();
                    let new_x = (ex + dir.x * ENEMY_SPEED * dt)
                        .clamp(floor.floor_x_min() + 12.0, floor.floor_x_max() - 12.0);
                    let new_y = (ey + dir.y * ENEMY_SPEED * dt)
                        .clamp(floor.floor_y_min() + 20.0, floor.floor_y_max() - 20.0);
                    tf.translation.x = new_x;
                    tf.translation.y = new_y;
                }
            }
            EnemyState::Attacking => {
                if d > MELEE_RANGE * 1.4 {
                    ai.state = EnemyState::Chasing;
                } else {
                    ai.attack_timer -= dt;
                    if ai.attack_timer <= 0.0 {
                        ai.attack_timer = ENEMY_ATTACK_CD;
                        let mut seed = rng.0;
                        let dmg = enemy_damage_roll(&mut seed);
                        rng.0 = seed;
                        ev_damage.send(DamageEvent {
                            target: player_entity,
                            amount: dmg,
                            from_player: false,
                        });
                        shake.trigger(1.5);
                    }
                }
            }
        }
    }
}

// ── Player auto-attack ────────────────────────────────────────────────────────

fn player_auto_attack(
    mut player_q: Query<(&Transform, &mut Cooldowns, &PlayerClassComp, &Equipment, &AttackTarget), With<PlayerMarker>>,
    enemy_q: Query<&Transform, (With<EnemyMarker>, Without<PlayerMarker>)>,
    mut ev_damage: EventWriter<DamageEvent>,
    mut rng: ResMut<Rng>,
    mut shake: ResMut<ScreenShakeRes>,
) {
    let Ok((player_tf, mut cooldowns, class_comp, equipment, attack_target)) =
        player_q.get_single_mut() else { return; };

    if cooldowns.attack > 0.0 { return; }

    let Ok(enemy_tf) = enemy_q.get(attack_target.0) else { return; };
    let d = player_tf.translation.truncate().distance(enemy_tf.translation.truncate());
    if d > ATTACK_RANGE { return; }

    let mut seed = rng.0;
    let dmg = player_damage_roll(&mut seed, class_comp.0, equipment.total_damage_bonus());
    rng.0 = seed;

    ev_damage.send(DamageEvent {
        target: attack_target.0,
        amount: dmg,
        from_player: true,
    });

    cooldowns.attack = 0.6;
    shake.trigger(1.0);

    #[cfg(target_arch = "wasm32")]
    crate::audio::play_hit();
}

// ── Damage / death ────────────────────────────────────────────────────────────

fn apply_damage(
    mut ev_damage: EventReader<DamageEvent>,
    mut hp_q: Query<(&mut Health, &Transform)>,
    mut commands: Commands,
    mut rng: ResMut<Rng>,
) {
    for ev in ev_damage.read() {
        let Ok((mut hp, tf)) = hp_q.get_mut(ev.target) else { continue; };
        hp.current = (hp.current - ev.amount).max(0.0);

        // Spawn damage number
        let offset_x = (rng.range_u32(20) as f32) - 10.0;
        commands.spawn((
            Text2d::new(format!("{:.0}", ev.amount)),
            TextFont { font_size: 14.0, ..default() },
            TextColor(C_DAMAGE),
            TextLayout::new_with_justify(JustifyText::Center),
            Transform::from_xyz(
                tf.translation.x + offset_x,
                tf.translation.y + 20.0,
                Z_VFX,
            ),
            DamageNumberComp { value: ev.amount as u32, timer: 0.5 },
        ));

        // Blood particles
        for _ in 0..6 {
            let angle = rng.f32() * std::f32::consts::TAU;
            let speed = rng.range_f32(40.0, 120.0);
            commands.spawn((
                Sprite {
                    color: C_PARTICLE,
                    custom_size: Some(Vec2::new(4.0, 4.0)),
                    ..default()
                },
                Transform::from_xyz(tf.translation.x, tf.translation.y, Z_VFX),
                ParticleComp {
                    vx: angle.cos() * speed,
                    vy: angle.sin() * speed,
                    timer: 0.5,
                    max_time: 0.5,
                },
            ));
        }
    }
}

fn check_deaths(
    enemy_q: Query<(Entity, &Health), (With<EnemyMarker>, Without<PlayerMarker>)>,
    player_q: Query<(Entity, &Health), (With<PlayerMarker>, Without<EnemyMarker>)>,
    mut commands: Commands,
    mut dungeon: ResMut<DungeonRes>,
    mut run_stats: ResMut<RunStats>,
    mut gold: ResMut<Gold>,
    mut rng: ResMut<Rng>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    // Check enemy deaths
    let mut all_dead = true;
    let mut living_count = 0u32;
    for (entity, hp) in &enemy_q {
        if hp.is_dead() {
            commands.entity(entity).despawn();
            run_stats.kills += 1;
            let gold_drop = 5 + rng.range_u32(11);
            gold.0 += gold_drop;
            run_stats.gold_earned += gold_drop;
            #[cfg(target_arch = "wasm32")]
            crate::audio::play_death();
        } else {
            all_dead = false;
            living_count += 1;
        }
    }

    // Mark room cleared when all enemies are gone
    if living_count == 0 && enemy_q.iter().count() > 0 {
        let room = dungeon.0.current_room;
        dungeon.0.rooms_cleared[room] = true;
    }

    // Check player death
    let Ok((_, player_hp)) = player_q.get_single() else { return; };
    if player_hp.is_dead() {
        next_state.set(GameState::Dead);
    }
}

// ── VFX systems ───────────────────────────────────────────────────────────────

fn damage_number_tick(
    time: Res<Time>,
    mut q: Query<(Entity, &mut DamageNumberComp, &mut Transform, &mut TextColor)>,
    mut commands: Commands,
) {
    let dt = time.delta_secs();
    for (entity, mut dn, mut tf, mut color) in &mut q {
        dn.timer -= dt;
        tf.translation.y += 30.0 * dt;
        let alpha = (dn.timer / 0.5).clamp(0.0, 1.0);
        color.0 = Color::srgba(1.0, 0.933, 0.267, alpha);
        if dn.timer <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}

fn particle_tick(
    time: Res<Time>,
    mut q: Query<(Entity, &mut ParticleComp, &mut Transform, &mut Sprite)>,
    mut commands: Commands,
) {
    let dt = time.delta_secs();
    for (entity, mut p, mut tf, mut sprite) in &mut q {
        p.timer -= dt;
        p.vy -= 120.0 * dt; // gravity
        p.vx *= (0.92_f32).powf(dt * 60.0);
        p.vy *= (0.92_f32).powf(dt * 60.0);
        tf.translation.x += p.vx * dt;
        tf.translation.y += p.vy * dt;
        let alpha = (p.timer / p.max_time).clamp(0.0, 1.0);
        sprite.color = Color::srgba(0.800, 0.133, 0.133, alpha);
        if p.timer <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}

fn screen_shake_tick(
    time: Res<Time>,
    mut shake: ResMut<ScreenShakeRes>,
    mut rng: ResMut<Rng>,
    mut camera_q: Query<&mut Transform, With<Camera2d>>,
) {
    let dt = time.delta_secs();
    shake.tick(&mut rng, dt);
    if let Ok(mut cam_tf) = camera_q.get_single_mut() {
        cam_tf.translation.x = shake.offset.x;
        cam_tf.translation.y = shake.offset.y;
    }
}

// ── Room transition ───────────────────────────────────────────────────────────

fn room_transition_check(
    player_q: Query<&Transform, With<PlayerMarker>>,
    mut dungeon: ResMut<DungeonRes>,
    mut transition: ResMut<PendingTransition>,
) {
    if transition.active { return; }
    let Ok(tf) = player_q.get_single() else { return; };
    let px = tf.translation.x;
    let py = tf.translation.y;
    if let Some(target) = dungeon.0.check_transition(px, py) {
        transition.active = true;
        transition.target_room = target;
        transition.phase = TransitionPhase::FadeOut;
        transition.timer = 0.0;

        #[cfg(target_arch = "wasm32")]
        crate::audio::play_transition();
    }
}

fn fade_overlay_tick(
    time: Res<Time>,
    mut transition: ResMut<PendingTransition>,
    mut dungeon: ResMut<DungeonRes>,
    mut player_q: Query<(&mut Transform, &mut MoveTarget), With<PlayerMarker>>,
    mut commands: Commands,
    existing_overlays: Query<Entity, With<FadeOverlay>>,
    mut run_stats: ResMut<RunStats>,
) {
    if !transition.active { return; }
    let dt = time.delta_secs();
    transition.timer += dt;

    const HALF: f32 = 0.27;

    match transition.phase {
        TransitionPhase::FadeOut => {
            let alpha = (transition.timer / HALF).clamp(0.0, 1.0);
            // Spawn/update overlay
            for e in &existing_overlays { commands.entity(e).despawn(); }
            commands.spawn((
                Sprite {
                    color: Color::srgba(0.0, 0.0, 0.0, alpha),
                    custom_size: Some(Vec2::new(SCREEN_W, SCREEN_H)),
                    ..default()
                },
                Transform::from_xyz(0.0, 0.0, Z_OVERLAY),
                FadeOverlay { timer: transition.timer, fading_in: false },
            ));
            if transition.timer >= HALF {
                transition.phase = TransitionPhase::Switch;
                transition.timer = 0.0;
            }
        }
        TransitionPhase::Switch => {
            // Actually switch the room
            let target = transition.target_room;
            dungeon.0.switch_room(target);
            run_stats.rooms_cleared += 1;

            let (spawn_x, spawn_y) = dungeon.0.current().entry_spawn();
            if let Ok((mut player_tf, mut move_target)) = player_q.get_single_mut() {
                player_tf.translation.x = spawn_x;
                player_tf.translation.y = spawn_y;
                move_target.0 = Vec2::new(spawn_x, spawn_y);
            }

            transition.phase = TransitionPhase::FadeIn;
            transition.timer = 0.0;
        }
        TransitionPhase::FadeIn => {
            let alpha = 1.0 - (transition.timer / HALF).clamp(0.0, 1.0);
            for e in &existing_overlays { commands.entity(e).despawn(); }
            commands.spawn((
                Sprite {
                    color: Color::srgba(0.0, 0.0, 0.0, alpha),
                    custom_size: Some(Vec2::new(SCREEN_W, SCREEN_H)),
                    ..default()
                },
                Transform::from_xyz(0.0, 0.0, Z_OVERLAY),
                FadeOverlay { timer: transition.timer, fading_in: true },
            ));
            if transition.timer >= HALF {
                for e in &existing_overlays { commands.entity(e).despawn(); }
                transition.active = false;
            }
        }
    }
}

// ── HUD live updates ──────────────────────────────────────────────────────────

pub fn update_hud(
    player_q: Query<(&Health, &Mana, &Cooldowns, &PlayerClassComp), With<PlayerMarker>>,
    mut hp_bar_q: Query<(&mut Sprite, &mut Transform), (With<HpBarFill>, Without<MpBarFill>, Without<AbilityBarFill>)>,
    mut mp_bar_q: Query<(&mut Sprite, &mut Transform), (With<MpBarFill>, Without<HpBarFill>, Without<AbilityBarFill>)>,
    mut ab_bar_q: Query<&mut Sprite, (With<AbilityBarFill>, Without<HpBarFill>, Without<MpBarFill>)>,
    mut gold_text_q: Query<&mut Text2d, (With<GoldText>, Without<HudFlashText>)>,
    mut flash_text_q: Query<&mut Text2d, (With<HudFlashText>, Without<GoldText>)>,
    gold: Res<Gold>,
    mut hud_flash: ResMut<HudFlash>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    hud_flash.timer = (hud_flash.timer - dt).max(0.0);

    let Ok((health, mana, cooldowns, class_comp)) = player_q.get_single() else { return; };
    let max_cd = class_comp.0.ability_cooldown_max();

    // HP bar width scaling
    if let Ok((mut sprite, mut tf)) = hp_bar_q.get_single_mut() {
        let frac = health.fraction();
        let full_w = 200.0;
        let w = full_w * frac;
        sprite.custom_size = Some(Vec2::new(w, 16.0));
        tf.translation.x = -200.0 - (full_w - w) / 2.0;
    }

    // MP bar width scaling
    if let Ok((mut sprite, mut tf)) = mp_bar_q.get_single_mut() {
        let frac = (mana.current / mana.max).clamp(0.0, 1.0);
        let full_w = 200.0;
        let w = full_w * frac;
        sprite.custom_size = Some(Vec2::new(w, 16.0));
        tf.translation.x = -200.0 - (full_w - w) / 2.0;
    }

    // Ability bar
    if let Ok(mut sprite) = ab_bar_q.get_single_mut() {
        let frac = if max_cd > 0.0 {
            1.0 - (cooldowns.ability / max_cd)
        } else { 1.0 };
        let frac = frac.clamp(0.0, 1.0);
        sprite.color = if frac >= 1.0 { C_ABILITY_RDY } else { C_ABILITY_CD };
        sprite.custom_size = Some(Vec2::new(100.0 * frac, 16.0));
    }

    // Gold text
    if let Ok(mut text) = gold_text_q.get_single_mut() {
        **text = format!("{} GP", gold.0);
    }

    // Flash text
    if let Ok(mut text) = flash_text_q.get_single_mut() {
        if hud_flash.timer > 0.0 {
            **text = hud_flash.text.clone();
        } else {
            **text = String::new();
        }
    }
}
