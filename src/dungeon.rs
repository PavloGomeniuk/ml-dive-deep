pub struct Room {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
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
}

pub struct Dungeon {
    pub room: Room,
}

impl Dungeon {
    pub fn new_single_room() -> Self {
        Dungeon {
            room: Room {
                x: 50.0,
                y: 30.0,
                w: 700.0,
                h: 320.0,
            },
        }
    }
}
