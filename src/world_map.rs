/// The location the player is currently in or traveling to.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Location {
    Desert,
    City,
}

// WorldMap layout: 800×400 game area split into two clickable zones.
// Desert: left half (x 80-340, y 80-340)
// City:   right half (x 460-720, y 80-340)

const DESERT_X: f32 = 80.0;
const DESERT_Y: f32 = 80.0;
const DESERT_W: f32 = 260.0;
const DESERT_H: f32 = 260.0;

const CITY_X: f32 = 460.0;
const CITY_Y: f32 = 80.0;
const CITY_W: f32 = 260.0;
const CITY_H: f32 = 260.0;

/// Merchant sprite position in the City scene.
pub const MERCHANT_X: f32 = 400.0;
pub const MERCHANT_Y: f32 = 220.0;
pub const MERCHANT_CLICK_RADIUS: f32 = 40.0;

pub fn hit_test_desert(x: f32, y: f32) -> bool {
    x >= DESERT_X && x <= DESERT_X + DESERT_W
        && y >= DESERT_Y && y <= DESERT_Y + DESERT_H
}

pub fn hit_test_city(x: f32, y: f32) -> bool {
    x >= CITY_X && x <= CITY_X + CITY_W
        && y >= CITY_Y && y <= CITY_Y + CITY_H
}

pub fn hit_test_merchant(x: f32, y: f32) -> bool {
    let dx = x - MERCHANT_X;
    let dy = y - MERCHANT_Y;
    (dx * dx + dy * dy).sqrt() <= MERCHANT_CLICK_RADIUS
}

/// Bounding rects exposed for rendering
pub fn desert_rect() -> (f32, f32, f32, f32) {
    (DESERT_X, DESERT_Y, DESERT_W, DESERT_H)
}

pub fn city_rect() -> (f32, f32, f32, f32) {
    (CITY_X, CITY_Y, CITY_W, CITY_H)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_test_desert_center() {
        let (x, y, w, h) = desert_rect();
        assert!(hit_test_desert(x + w / 2.0, y + h / 2.0));
    }

    #[test]
    fn hit_test_city_center() {
        let (x, y, w, h) = city_rect();
        assert!(hit_test_city(x + w / 2.0, y + h / 2.0));
    }

    #[test]
    fn desert_and_city_do_not_overlap() {
        // A point in the desert should not be in the city
        let (dx, dy, dw, dh) = desert_rect();
        let center_x = dx + dw / 2.0;
        let center_y = dy + dh / 2.0;
        assert!(!hit_test_city(center_x, center_y));
    }

    #[test]
    fn merchant_click_within_radius() {
        assert!(hit_test_merchant(MERCHANT_X, MERCHANT_Y));
        assert!(hit_test_merchant(MERCHANT_X + 20.0, MERCHANT_Y));
    }

    #[test]
    fn merchant_click_outside_radius() {
        assert!(!hit_test_merchant(MERCHANT_X + 50.0, MERCHANT_Y));
    }
}
