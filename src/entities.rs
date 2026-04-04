#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EnemyState {
    Patrolling,
    Chasing,
    Attacking,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PlayerClass {
    Warrior,
    Magician,
}

impl PlayerClass {
    pub fn base_hp(&self) -> f32 {
        match self {
            PlayerClass::Warrior => 150.0,
            PlayerClass::Magician => 90.0,
        }
    }

    pub fn base_mana(&self) -> f32 {
        match self {
            PlayerClass::Warrior => 30.0,
            PlayerClass::Magician => 100.0,
        }
    }

    pub fn ability_cooldown_max(&self) -> f32 {
        match self {
            PlayerClass::Warrior => 4.0,
            PlayerClass::Magician => 3.0,
        }
    }

    pub fn ability_name(&self) -> &'static str {
        match self {
            PlayerClass::Warrior => "Cleave",
            PlayerClass::Magician => "Frost Nova",
        }
    }
}

/// Equipment slot indices: 0=Weapon, 1=Armor, 2=Ring
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ItemKind {
    Sword,
    Staff,
    Tome,
}

impl ItemKind {
    pub fn name(&self) -> &'static str {
        match self {
            ItemKind::Sword => "Sword",
            ItemKind::Staff => "Staff",
            ItemKind::Tome => "Tome",
        }
    }

    pub fn damage_bonus(&self) -> f32 {
        match self {
            ItemKind::Sword => 5.0,
            ItemKind::Staff => 3.0,
            ItemKind::Tome => 0.0,
        }
    }

    pub fn slot(&self) -> usize {
        match self {
            ItemKind::Sword => 0,
            ItemKind::Staff => 1,
            ItemKind::Tome => 2,
        }
    }
}

pub struct Player {
    pub x: f32,
    pub y: f32,
    pub hp: f32,
    pub max_hp: f32,
    pub mana: f32,
    pub max_mana: f32,
    pub class: PlayerClass,
    pub attack_cooldown: f32,
    pub ability_cooldown: f32,
    pub move_target_x: f32,
    pub move_target_y: f32,
    pub moving: bool,
    pub attack_target: Option<usize>,
    pub equipment: [Option<ItemKind>; 3],
}

impl Player {
    pub fn new(x: f32, y: f32) -> Self {
        Player::new_with_class(x, y, PlayerClass::Warrior)
    }

    pub fn new_with_class(x: f32, y: f32, class: PlayerClass) -> Self {
        let max_hp = class.base_hp();
        let max_mana = class.base_mana();
        Player {
            x,
            y,
            hp: max_hp,
            max_hp,
            mana: max_mana,
            max_mana,
            class,
            attack_cooldown: 0.0,
            ability_cooldown: 0.0,
            move_target_x: x,
            move_target_y: y,
            moving: false,
            attack_target: None,
            equipment: [None, None, None],
        }
    }

    pub fn damage_bonus(&self) -> f32 {
        self.equipment.iter().filter_map(|e| *e).map(|k| k.damage_bonus()).sum()
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
    pub frozen_timer: f32,
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
            frozen_timer: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warrior_has_more_hp_than_magician() {
        let w = Player::new_with_class(0.0, 0.0, PlayerClass::Warrior);
        let m = Player::new_with_class(0.0, 0.0, PlayerClass::Magician);
        assert!(w.max_hp > m.max_hp);
    }

    #[test]
    fn magician_has_more_mana_than_warrior() {
        let w = Player::new_with_class(0.0, 0.0, PlayerClass::Warrior);
        let m = Player::new_with_class(0.0, 0.0, PlayerClass::Magician);
        assert!(m.max_mana > w.max_mana);
    }

    #[test]
    fn player_starts_at_full_hp() {
        let p = Player::new(400.0, 200.0);
        assert_eq!(p.hp, p.max_hp);
    }

    #[test]
    fn enemy_starts_alive_at_full_hp() {
        let e = Enemy::new(200.0, 150.0);
        assert!(e.alive);
        assert_eq!(e.hp, 30.0);
        assert_eq!(e.max_hp, 30.0);
        assert_eq!(e.state, EnemyState::Patrolling);
    }

    #[test]
    fn sword_damage_bonus_positive() {
        let mut p = Player::new(0.0, 0.0);
        p.equipment[0] = Some(ItemKind::Sword);
        assert!(p.damage_bonus() > 0.0);
    }

    // Regression: ISSUE-002 — Staff and Sword must occupy different equipment slots
    // Found by /qa on 2026-04-04
    // Report: .gstack/qa-reports/qa-report-localhost-2026-04-04.md
    #[test]
    fn item_slots_are_distinct() {
        assert_ne!(ItemKind::Sword.slot(), ItemKind::Staff.slot());
        assert_ne!(ItemKind::Sword.slot(), ItemKind::Tome.slot());
        assert_ne!(ItemKind::Staff.slot(), ItemKind::Tome.slot());
    }

    #[test]
    fn all_item_slots_in_bounds() {
        let slots = [ItemKind::Sword.slot(), ItemKind::Staff.slot(), ItemKind::Tome.slot()];
        for s in &slots {
            assert!(*s < 3, "slot {} out of bounds for equipment[3]", s);
        }
    }
}
