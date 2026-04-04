use crate::entities::{Enemy, PlayerClass};

pub fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

pub fn distance(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

pub enum ClickAction {
    Attack(usize),
    Move(f32, f32),
}

pub fn click_disambiguation(cx: f32, cy: f32, enemies: &[Enemy]) -> ClickAction {
    const ATTACK_RADIUS: f32 = 60.0;
    let mut closest_dist = f32::MAX;
    let mut closest_idx = None;

    for (i, enemy) in enemies.iter().enumerate() {
        if !enemy.alive {
            continue;
        }
        let d = distance(cx, cy, enemy.x, enemy.y);
        if d < ATTACK_RADIUS && d < closest_dist {
            closest_dist = d;
            closest_idx = Some(i);
        }
    }

    match closest_idx {
        Some(i) => ClickAction::Attack(i),
        None => ClickAction::Move(cx, cy),
    }
}

pub fn player_damage_roll(rng: &mut u32, class: PlayerClass, damage_bonus: f32) -> f32 {
    let base = match class {
        PlayerClass::Warrior => 15.0 + (xorshift(rng) % 6) as f32,
        PlayerClass::Magician => 12.0 + (xorshift(rng) % 7) as f32,
    };
    base + damage_bonus
}

pub fn enemy_damage_roll(rng: &mut u32) -> f32 {
    8.0 + (xorshift(rng) % 5) as f32
}

/// Warrior Cleave: AoE hit on all enemies within 80px. Returns list of (enemy_idx, damage).
pub fn warrior_cleave(
    px: f32,
    py: f32,
    enemies: &[Enemy],
    rng: &mut u32,
    damage_bonus: f32,
) -> Vec<(usize, f32)> {
    const CLEAVE_RADIUS: f32 = 80.0;
    let mut hits = Vec::new();
    for (i, enemy) in enemies.iter().enumerate() {
        if !enemy.alive {
            continue;
        }
        let d = distance(px, py, enemy.x, enemy.y);
        if d <= CLEAVE_RADIUS {
            let dmg = 15.0 + (xorshift(rng) % 6) as f32 + damage_bonus;
            hits.push((i, dmg));
        }
    }
    hits
}

/// Magician Frost Nova: AoE freeze on all enemies within 70px.
/// Returns list of enemy indices that were frozen.
pub fn magician_frost_nova(
    px: f32,
    py: f32,
    enemies: &[Enemy],
) -> Vec<usize> {
    const NOVA_RADIUS: f32 = 70.0;
    let mut frozen = Vec::new();
    for (i, enemy) in enemies.iter().enumerate() {
        if !enemy.alive {
            continue;
        }
        let d = distance(px, py, enemy.x, enemy.y);
        if d <= NOVA_RADIUS {
            frozen.push(i);
        }
    }
    frozen
}

#[derive(Default)]
pub struct HitStop {
    pub frames_remaining: u8,
}

impl HitStop {
    pub fn trigger(&mut self) {
        self.frames_remaining = 4;
    }

    /// Returns true if frozen this tick (and decrements), false if running.
    pub fn tick(&mut self) -> bool {
        if self.frames_remaining > 0 {
            self.frames_remaining -= 1;
            true
        } else {
            false
        }
    }

    pub fn is_active(&self) -> bool {
        self.frames_remaining > 0
    }
}

#[derive(Default)]
pub struct ScreenShake {
    pub intensity: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

impl ScreenShake {
    pub fn trigger(&mut self) {
        self.intensity = 3.0;
    }

    pub fn update(&mut self, rng: &mut u32) {
        if self.intensity < 0.1 {
            self.intensity = 0.0;
            self.offset_x = 0.0;
            self.offset_y = 0.0;
            return;
        }
        let angle_bits = xorshift(rng);
        let angle = (angle_bits % 628) as f32 / 100.0;
        self.offset_x = self.intensity * angle.cos();
        self.offset_y = self.intensity * angle.sin();
        self.intensity *= 0.75;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::Enemy;

    fn make_enemies(positions: &[(f32, f32)]) -> Vec<Enemy> {
        positions.iter().map(|&(x, y)| Enemy::new(x, y)).collect()
    }

    #[test]
    fn distance_within_melee_range() {
        assert!(distance(100.0, 100.0, 140.0, 100.0) <= 50.0);
    }

    #[test]
    fn distance_outside_melee_range() {
        assert!(distance(100.0, 100.0, 200.0, 100.0) > 50.0);
    }

    #[test]
    fn warrior_damage_in_range() {
        let mut rng: u32 = 12345;
        for _ in 0..100 {
            let dmg = player_damage_roll(&mut rng, PlayerClass::Warrior, 0.0);
            assert!(dmg >= 15.0 && dmg <= 20.0, "damage {} out of [15,20]", dmg);
        }
    }

    #[test]
    fn magician_damage_in_range() {
        let mut rng: u32 = 12345;
        for _ in 0..100 {
            let dmg = player_damage_roll(&mut rng, PlayerClass::Magician, 0.0);
            assert!(dmg >= 12.0 && dmg <= 18.0, "damage {} out of [12,18]", dmg);
        }
    }

    #[test]
    fn enemy_damage_always_in_range() {
        let mut rng: u32 = 99999;
        for _ in 0..100 {
            let dmg = enemy_damage_roll(&mut rng);
            assert!(dmg >= 8.0 && dmg <= 12.0, "damage {} out of [8,12]", dmg);
        }
    }

    #[test]
    fn cleave_hits_nearby_enemies() {
        let enemies = make_enemies(&[(50.0, 50.0), (500.0, 500.0)]);
        let mut rng: u32 = 1;
        let hits = warrior_cleave(50.0, 50.0, &enemies, &mut rng, 0.0);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);
    }

    #[test]
    fn frost_nova_freezes_nearby() {
        let enemies = make_enemies(&[(60.0, 60.0), (500.0, 500.0)]);
        let frozen = magician_frost_nova(50.0, 50.0, &enemies);
        assert_eq!(frozen, vec![0]);
    }

    #[test]
    fn hitstop_frames_positive_returns_frozen() {
        let mut hs = HitStop { frames_remaining: 4 };
        assert!(hs.tick());
        assert_eq!(hs.frames_remaining, 3);
    }

    #[test]
    fn hitstop_frames_zero_returns_running() {
        let mut hs = HitStop { frames_remaining: 0 };
        assert!(!hs.tick());
    }

    #[test]
    fn click_disambiguation_within_radius_attacks() {
        let enemies = make_enemies(&[(170.0, 160.0)]);
        match click_disambiguation(150.0, 150.0, &enemies) {
            ClickAction::Attack(i) => assert_eq!(i, 0),
            ClickAction::Move(_, _) => panic!("expected Attack"),
        }
    }

    #[test]
    fn click_disambiguation_outside_all_enemies_moves() {
        let enemies = make_enemies(&[(100.0, 100.0), (120.0, 110.0)]);
        match click_disambiguation(500.0, 500.0, &enemies) {
            ClickAction::Move(x, y) => {
                assert!((x - 500.0).abs() < 0.001);
                assert!((y - 500.0).abs() < 0.001);
            }
            ClickAction::Attack(_) => panic!("expected Move"),
        }
    }
}
