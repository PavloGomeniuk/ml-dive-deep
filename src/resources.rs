use bevy::prelude::*;
use crate::entities::PlayerClass;

/// xorshift32 RNG wrapped as a Resource.
#[derive(Resource)]
pub struct Rng(pub u32);

impl Rng {
    pub fn new(seed: u32) -> Self { Self(seed.max(1)) }

    pub fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Random f32 in [0, 1)
    pub fn f32(&mut self) -> f32 { self.next() as f32 / u32::MAX as f32 }

    /// Random u32 in [0, range)
    pub fn range_u32(&mut self, range: u32) -> u32 { self.next() % range }

    /// Random f32 in [min, max)
    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + self.f32() * (max - min)
    }
}

/// Player's gold.
#[derive(Resource, Default)]
pub struct Gold(pub u32);

/// Screen shake state.
#[derive(Resource, Default)]
pub struct ScreenShakeRes {
    pub intensity: f32,
    pub offset: Vec2,
}

impl ScreenShakeRes {
    pub fn trigger(&mut self, intensity: f32) {
        self.intensity = self.intensity.max(intensity);
    }

    pub fn tick(&mut self, rng: &mut Rng, dt: f32) {
        if self.intensity < 0.05 {
            self.intensity = 0.0;
            self.offset = Vec2::ZERO;
            return;
        }
        let angle = rng.f32() * std::f32::consts::TAU;
        self.offset = Vec2::new(angle.cos(), angle.sin()) * self.intensity;
        self.intensity -= self.intensity * 8.0 * dt; // exponential decay
    }
}

/// Flashed message in the HUD (e.g. "nothing to equip").
#[derive(Resource, Default)]
pub struct HudFlash {
    pub text: String,
    pub timer: f32,
}

impl HudFlash {
    pub fn show(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.timer = 1.2; // seconds
    }
}

/// Chosen class from CharacterSelect screen.
#[derive(Resource, Default)]
pub struct SelectedClass(pub Option<PlayerClass>);

/// Stats for the death screen.
#[derive(Resource, Default)]
pub struct RunStats {
    pub kills: u32,
    pub rooms_cleared: u32,
    pub gold_earned: u32,
}

/// Which card is highlighted on the CharSelect screen (0=Warrior, 1=Magician).
#[derive(Resource, Default)]
pub struct CharSelectHighlight(pub Option<usize>);

/// Pending room transition.
#[derive(Resource, Default)]
pub struct PendingTransition {
    pub active: bool,
    pub target_room: usize,
    pub phase: TransitionPhase,
    pub timer: f32,
}

#[derive(Default, PartialEq)]
pub enum TransitionPhase {
    #[default]
    FadeOut,  // clear→black
    Switch,   // switch room at midpoint
    FadeIn,   // black→clear
}
