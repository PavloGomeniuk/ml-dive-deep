use crate::entities::Enemy;
use crate::items::{Chest, spawn_chest_loot};

pub struct Room {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub index: usize,
}

impl Room {
    pub fn floor_x(&self) -> f32 { self.x + 20.0 }
    pub fn floor_y(&self) -> f32 { self.y + 20.0 }
    pub fn floor_w(&self) -> f32 { self.w - 40.0 }
    pub fn floor_h(&self) -> f32 { self.h - 40.0 }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.floor_x()
            && px <= self.floor_x() + self.floor_w()
            && py >= self.floor_y()
            && py <= self.floor_y() + self.floor_h()
    }

    /// Center of the door for this room.
    /// Room 0: door at bottom-center of floor.
    /// Room 1: door at top-center of floor (to go back).
    pub fn door_center(&self) -> (f32, f32) {
        let cx = self.floor_x() + self.floor_w() / 2.0;
        match self.index {
            0 => (cx, self.floor_y() + self.floor_h() - 5.0),
            _ => (cx, self.floor_y() + 5.0),
        }
    }

    /// Spawn point for player when arriving into this room.
    pub fn entry_spawn(&self) -> (f32, f32) {
        let cx = self.floor_x() + self.floor_w() / 2.0;
        match self.index {
            // Arriving from room 1: spawn near bottom, away from door
            0 => (cx, self.floor_y() + self.floor_h() - 40.0),
            // Arriving from room 0: spawn near top, away from door
            _ => (cx, self.floor_y() + 40.0),
        }
    }
}

pub struct DungeonMap {
    pub rooms: Vec<Room>,
    pub current_room: usize,
    /// Per-room enemy snapshots. Loaded into gs.enemies on room switch.
    pub enemy_cache: Vec<Vec<Enemy>>,
    /// Per-room chest snapshots. Loaded into gs.chests on room switch.
    pub chest_cache: Vec<Vec<Chest>>,
}

fn make_room0_chests(rng: &mut u32) -> Vec<Chest> {
    vec![
        Chest::new(560.0, 260.0, spawn_chest_loot(rng)),
        Chest::new(160.0, 80.0, spawn_chest_loot(rng)),
    ]
}

fn make_room1_chests(rng: &mut u32) -> Vec<Chest> {
    vec![
        Chest::new(320.0, 240.0, spawn_chest_loot(rng)),
    ]
}

impl DungeonMap {
    pub fn new() -> Self {
        Self::new_with_seed(12345)
    }

    pub fn new_with_seed(seed: u32) -> Self {
        let mut rng = seed.max(1);

        let rooms = vec![
            Room { x: 50.0, y: 30.0, w: 700.0, h: 320.0, index: 0 },
            Room { x: 50.0, y: 30.0, w: 700.0, h: 320.0, index: 1 },
        ];

        let room0_enemies = vec![
            Enemy::new(200.0, 150.0),
            Enemy::new(600.0, 150.0),
            Enemy::new(600.0, 290.0),
        ];
        let room1_enemies = vec![
            Enemy::new(250.0, 120.0),
            Enemy::new(550.0, 200.0),
            Enemy::new(380.0, 280.0),
            Enemy::new(480.0, 100.0),
        ];

        let room0_chests = make_room0_chests(&mut rng);
        let room1_chests = make_room1_chests(&mut rng);

        DungeonMap {
            rooms,
            current_room: 0,
            enemy_cache: vec![room0_enemies, room1_enemies],
            chest_cache: vec![room0_chests, room1_chests],
        }
    }

    pub fn current_room(&self) -> &Room {
        &self.rooms[self.current_room]
    }

    /// Switch to room at `idx`, saturating-clamped to valid range.
    pub fn switch_room(&mut self, idx: usize) {
        let idx = idx.min(self.rooms.len().saturating_sub(1));
        self.current_room = idx;
    }

    /// Target room index if player (px, py) is in the door trigger zone (30px radius).
    pub fn check_transition(&self, px: f32, py: f32) -> Option<usize> {
        const TRIGGER_RADIUS: f32 = 30.0;
        let room = self.current_room();
        let (dx, dy) = room.door_center();
        let dist = ((px - dx) * (px - dx) + (py - dy) * (py - dy)).sqrt();
        if dist <= TRIGGER_RADIUS {
            let target = if self.current_room == 0 { 1 } else { 0 };
            Some(target)
        } else {
            None
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn room0_door_at_bottom() {
        let map = DungeonMap::new();
        let (_, dy) = map.rooms[0].door_center();
        let floor_bottom = map.rooms[0].floor_y() + map.rooms[0].floor_h();
        assert!(dy < floor_bottom);
        assert!(dy > floor_bottom - 10.0);
    }

    #[test]
    fn room1_door_at_top() {
        let map = DungeonMap::new();
        let (_, dy) = map.rooms[1].door_center();
        let floor_top = map.rooms[1].floor_y();
        assert!(dy > floor_top);
        assert!(dy < floor_top + 10.0);
    }

    #[test]
    fn transition_triggered_in_range() {
        let map = DungeonMap::new();
        let (dx, dy) = map.rooms[0].door_center();
        assert!(map.check_transition(dx, dy).is_some());
    }

    #[test]
    fn transition_not_triggered_far_away() {
        let map = DungeonMap::new();
        assert!(map.check_transition(10.0, 10.0).is_none());
    }

    #[test]
    fn switch_room_bounds_check() {
        let mut map = DungeonMap::new();
        map.switch_room(usize::MAX);
        assert_eq!(map.current_room, 1);
    }

    #[test]
    fn room0_has_chests() {
        let map = DungeonMap::new();
        assert!(!map.chest_cache[0].is_empty());
    }

    #[test]
    fn room1_has_chests() {
        let map = DungeonMap::new();
        assert!(!map.chest_cache[1].is_empty());
    }
}
