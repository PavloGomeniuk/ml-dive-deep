#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EnemyState {
    Patrolling,
    Chasing,
    Attacking,
}

pub struct Player {
    pub x: f32,
    pub y: f32,
    pub hp: f32,
    pub max_hp: f32,
    pub attack_cooldown: f32,
    pub move_target_x: f32,
    pub move_target_y: f32,
    pub moving: bool,
    pub attack_target: Option<usize>,
}

impl Player {
    pub fn new(x: f32, y: f32) -> Self {
        Player {
            x,
            y,
            hp: 100.0,
            max_hp: 100.0,
            attack_cooldown: 0.0,
            move_target_x: x,
            move_target_y: y,
            moving: false,
            attack_target: None,
        }
    }
}

pub struct Enemy {
    pub x: f32,
    pub y: f32,
    pub hp: f32,
    pub max_hp: f32,
    pub state: EnemyState,
    pub attack_cooldown: f32,
    pub home_x: f32,
    pub home_y: f32,
    pub alive: bool,
}

impl Enemy {
    pub fn new(x: f32, y: f32) -> Self {
        Enemy {
            x,
            y,
            hp: 30.0,
            max_hp: 30.0,
            state: EnemyState::Patrolling,
            attack_cooldown: 0.0,
            home_x: x,
            home_y: y,
            alive: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_starts_at_full_hp() {
        let p = Player::new(400.0, 200.0);
        assert_eq!(p.hp, 100.0);
        assert_eq!(p.max_hp, 100.0);
    }

    #[test]
    fn enemy_starts_alive_at_full_hp() {
        let e = Enemy::new(200.0, 150.0);
        assert!(e.alive);
        assert_eq!(e.hp, 30.0);
        assert_eq!(e.max_hp, 30.0);
        assert_eq!(e.state, EnemyState::Patrolling);
    }
}
