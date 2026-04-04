use bevy::prelude::*;
use crate::entities::{PlayerClass, EnemyState, ItemKind};

// ── Screen layout constants (Bevy Y-up, origin = screen center) ───────────────
// Window: 800 × 500
// Play area: y ∈ [-150, 250]  (400 px tall)
// HUD strip: y ∈ [-250, -150] (100 px tall)
// Dungeon room: center (0, 60), 700 × 320
// Floor (inset 20 px walls): center (0, 60), 660 × 280, y ∈ [-80, 200]

pub const SCREEN_W: f32 = 800.0;
pub const SCREEN_H: f32 = 500.0;
pub const ROOM_W: f32 = 700.0;
pub const ROOM_H: f32 = 320.0;
pub const ROOM_CX: f32 = 0.0;
pub const ROOM_CY: f32 = 60.0;
pub const WALL_T: f32 = 20.0;
pub const FLOOR_W: f32 = ROOM_W - 2.0 * WALL_T;   // 660
pub const FLOOR_H: f32 = ROOM_H - 2.0 * WALL_T;   // 280
pub const FLOOR_X_MIN: f32 = ROOM_CX - FLOOR_W / 2.0; // -330
pub const FLOOR_X_MAX: f32 = ROOM_CX + FLOOR_W / 2.0; // 330
pub const FLOOR_Y_MIN: f32 = ROOM_CY - FLOOR_H / 2.0; // -80
pub const FLOOR_Y_MAX: f32 = ROOM_CY + FLOOR_H / 2.0; // 200
pub const HUD_CY: f32 = -200.0;
pub const HUD_H: f32 = 100.0;

// Z-layers
pub const Z_BG: f32 = 0.0;
pub const Z_FLOOR: f32 = 1.0;
pub const Z_ITEM: f32 = 2.0;
pub const Z_ENEMY: f32 = 3.0;
pub const Z_PLAYER: f32 = 3.5;
pub const Z_VFX: f32 = 4.0;
pub const Z_HUD: f32 = 5.0;
pub const Z_HUD_TEXT: f32 = 6.0;
pub const Z_OVERLAY: f32 = 10.0;

// ── Markers ────────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct PlayerMarker;

#[derive(Component)]
pub struct EnemyMarker;

// ── Player / shared stats ──────────────────────────────────────────────────────

#[derive(Component)]
pub struct PlayerClassComp(pub PlayerClass);

#[derive(Component)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

impl Health {
    pub fn new(max: f32) -> Self { Self { current: max, max } }
    pub fn fraction(&self) -> f32 { (self.current / self.max).clamp(0.0, 1.0) }
    pub fn is_dead(&self) -> bool { self.current <= 0.0 }
}

#[derive(Component)]
pub struct Mana {
    pub current: f32,
    pub max: f32,
}

impl Mana {
    pub fn new(max: f32) -> Self { Self { current: max, max } }
}

#[derive(Component, Default)]
pub struct Cooldowns {
    pub attack: f32,
    pub ability: f32,
}

#[derive(Component)]
pub struct MoveTarget(pub Vec2);

#[derive(Component)]
pub struct AttackTarget(pub Entity);

#[derive(Component)]
pub struct DamageBonus(pub f32);

#[derive(Component, Default)]
pub struct Equipment(pub [Option<ItemKind>; 3]);

impl Equipment {
    pub fn total_damage_bonus(&self) -> f32 {
        self.0.iter()
            .filter_map(|s| *s)
            .map(|k| k.damage_bonus())
            .sum()
    }
}

#[derive(Component, Default)]
pub struct Potions { pub hp: u8, pub mp: u8 }

// ── Enemy ─────────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct EnemyAI {
    pub state: EnemyState,
    pub home: Vec2,
    pub attack_timer: f32,
}

#[derive(Component)]
pub struct Frozen(pub f32);

// ── Items ─────────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct ItemOnFloor(pub ItemKind);

// ── VFX ───────────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct DamageNumberComp {
    pub value: u32,
    pub timer: f32,       // countdown from 0.5s
}

#[derive(Component)]
pub struct ParticleComp {
    pub vx: f32,
    pub vy: f32,
    pub timer: f32,
    pub max_time: f32,
}

// ── HUD live update markers ────────────────────────────────────────────────────

#[derive(Component)]
pub struct HpBarFill;

#[derive(Component)]
pub struct MpBarFill;

#[derive(Component)]
pub struct AbilityBarFill;

#[derive(Component)]
pub struct GoldText;

#[derive(Component)]
pub struct HudFlashText;

// ── Dungeon visual markers ────────────────────────────────────────────────────

#[derive(Component)]
pub struct DungeonFloor;

// ── Transition fade overlay ────────────────────────────────────────────────────

#[derive(Component)]
pub struct FadeOverlay {
    pub timer: f32,      // 0..1
    pub fading_in: bool, // true = black→clear, false = clear→black
}
