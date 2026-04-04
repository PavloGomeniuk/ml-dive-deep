use crate::entities::Enemy;

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
}

impl DungeonMap {
    pub fn new() -> Self {
        let rooms = vec![
            Room { x: 50.0, y: 30.0, w: 700.0, h: 320.0, index: 0 },
            Room { x: 50.0, y: 30.0, w: 700.0, h: 320.0, index: 1 },
        ];

        let room0_enemies = vec![
            Enemy::new(200.0, 150.0),
            Enemy::new(600.0, 150.0),
            Enemy::new(400.0, 300.0),
        ];
        let room1_enemies = vec![
            Enemy::new(250.0, 120.0),
            Enemy::new(550.0, 200.0),
            Enemy::new(380.0, 280.0),
            Enemy::new(480.0, 100.0),
        ];

        DungeonMap {
            rooms,
            current_room: 0,
            enemy_cache: vec![room0_enemies, room1_enemies],
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
            // Determine target room
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
        // Should not panic, should saturate to last valid index
        map.switch_room(usize::MAX);
        assert_eq!(map.current_room, 1);
    }
}
