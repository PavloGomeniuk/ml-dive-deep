//! Dungeon room layout — pure data, no Bevy types.
//! Coordinates are in Bevy world-space (Y-up, origin at screen centre).
//! Canvas coords map as: bevy_x = canvas_x - 400, bevy_y = 250 - canvas_y.

const WALL_THICKNESS: f32 = 20.0;

pub struct Room {
    /// Centre of the room in Bevy world coords.
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub index: usize,
}

impl Room {
    pub fn floor_w(&self) -> f32 { self.w - 2.0 * WALL_THICKNESS }
    pub fn floor_h(&self) -> f32 { self.h - 2.0 * WALL_THICKNESS }
    pub fn floor_x_min(&self) -> f32 { self.cx - self.floor_w() / 2.0 }
    pub fn floor_x_max(&self) -> f32 { self.cx + self.floor_w() / 2.0 }
    pub fn floor_y_min(&self) -> f32 { self.cy - self.floor_h() / 2.0 }
    pub fn floor_y_max(&self) -> f32 { self.cy + self.floor_h() / 2.0 }

    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.floor_x_min()
            && x <= self.floor_x_max()
            && y >= self.floor_y_min()
            && y <= self.floor_y_max()
    }

    /// Door centre position.
    /// Room 0: exit door at floor bottom (lower Y — forward progression).
    /// Room N>0: exit door at floor top (higher Y — return door).
    pub fn exit_door(&self) -> (f32, f32) {
        match self.index {
            0 => (self.cx, self.floor_y_min() + 5.0),
            _ => (self.cx, self.floor_y_max() - 5.0),
        }
    }

    /// Where the player spawns when arriving into this room.
    pub fn entry_spawn(&self) -> (f32, f32) {
        match self.index {
            // Entering from room 1: player appears near bottom
            0 => (self.cx, self.floor_y_min() + 50.0),
            // Entering from room 0: player appears near top
            _ => (self.cx, self.floor_y_max() - 50.0),
        }
    }
}

/// Dungeon map: ordered list of rooms + current room index.
/// Per-room cleared state is tracked here (enemies defeated).
pub struct DungeonMap {
    pub rooms: Vec<Room>,
    pub current_room: usize,
    /// true once all enemies in the room are dead (unlocks exit door).
    pub rooms_cleared: Vec<bool>,
}

impl DungeonMap {
    pub fn new() -> Self {
        // Room layout in Bevy coords — two rooms stacked vertically.
        // Same spatial footprint as v2 (700×320 at canvas (50,30)).
        let rooms = vec![
            Room { cx: 0.0, cy: 60.0, w: 700.0, h: 320.0, index: 0 },
            Room { cx: 0.0, cy: 60.0, w: 700.0, h: 320.0, index: 1 },
        ];
        let room_count = rooms.len();
        DungeonMap {
            rooms,
            current_room: 0,
            rooms_cleared: vec![false; room_count],
        }
    }

    pub fn current(&self) -> &Room {
        &self.rooms[self.current_room]
    }

    /// Switch room, clamped to valid range.
    pub fn switch_room(&mut self, idx: usize) {
        self.current_room = idx.min(self.rooms.len().saturating_sub(1));
    }

    /// Returns the target room if the player is within 30 px of the exit door.
    pub fn check_transition(&self, px: f32, py: f32) -> Option<usize> {
        if !self.rooms_cleared[self.current_room] {
            return None; // door locked until room is cleared
        }
        const TRIGGER_R: f32 = 30.0;
        let (dx, dy) = self.current().exit_door();
        let dist = ((px - dx).powi(2) + (py - dy).powi(2)).sqrt();
        if dist <= TRIGGER_R {
            let target = if self.current_room + 1 < self.rooms.len() {
                self.current_room + 1
            } else {
                0
            };
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
    fn room0_exit_at_floor_bottom() {
        let map = DungeonMap::new();
        let (_, dy) = map.rooms[0].exit_door();
        assert!(dy < map.rooms[0].cy); // door is below centre
        assert!(dy >= map.rooms[0].floor_y_min());
    }

    #[test]
    fn room1_exit_at_floor_top() {
        let map = DungeonMap::new();
        let (_, dy) = map.rooms[1].exit_door();
        assert!(dy > map.rooms[1].cy); // door is above centre
        assert!(dy <= map.rooms[1].floor_y_max());
    }

    #[test]
    fn transition_locked_if_room_not_cleared() {
        let map = DungeonMap::new();
        let (dx, dy) = map.rooms[0].exit_door();
        // Not cleared — should return None even when standing on door
        assert!(map.check_transition(dx, dy).is_none());
    }

    #[test]
    fn transition_triggers_when_cleared_and_in_range() {
        let mut map = DungeonMap::new();
        map.rooms_cleared[0] = true;
        let (dx, dy) = map.rooms[0].exit_door();
        assert!(map.check_transition(dx, dy).is_some());
    }

    #[test]
    fn transition_returns_none_far_from_door() {
        let mut map = DungeonMap::new();
        map.rooms_cleared[0] = true;
        assert!(map.check_transition(999.0, 999.0).is_none());
    }

    #[test]
    fn switch_room_bounds_check() {
        let mut map = DungeonMap::new();
        map.switch_room(usize::MAX);
        assert_eq!(map.current_room, map.rooms.len() - 1);
    }

    #[test]
    fn room_contains_own_centre() {
        let map = DungeonMap::new();
        let r = &map.rooms[0];
        assert!(r.contains(r.cx, r.cy));
    }
}
