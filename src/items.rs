use crate::combat::{xorshift, distance};
use crate::entities::ItemKind;

pub struct Item {
    pub x: f32,
    pub y: f32,
    pub kind: ItemKind,
    /// Frames the name label stays visible (starts at 180 = 3s @ 60fps)
    pub label_life: u32,
}

impl Item {
    pub fn new(x: f32, y: f32, kind: ItemKind) -> Self {
        Item { x, y, kind, label_life: 180 }
    }
}

/// Drop a random item at position using the game RNG.
pub fn spawn_loot(x: f32, y: f32, rng: &mut u32) -> Item {
    let r = xorshift(rng) % 3;
    let kind = match r {
        0 => ItemKind::Sword,
        1 => ItemKind::Staff,
        _ => ItemKind::Tome,
    };
    Item::new(x, y, kind)
}

/// Returns the index of the nearest item within 30px of (px, py), or None.
pub fn nearest_item(items: &[Item], px: f32, py: f32) -> Option<usize> {
    const PICKUP_RANGE: f32 = 30.0;
    let mut best_dist = f32::MAX;
    let mut best_idx = None;
    for (i, item) in items.iter().enumerate() {
        let d = distance(px, py, item.x, item.y);
        if d < PICKUP_RANGE && d < best_dist {
            best_dist = d;
            best_idx = Some(i);
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_item_within_range() {
        let items = vec![Item::new(100.0, 100.0, ItemKind::Sword)];
        assert_eq!(nearest_item(&items, 110.0, 100.0), Some(0));
    }

    #[test]
    fn nearest_item_out_of_range() {
        let items = vec![Item::new(100.0, 100.0, ItemKind::Sword)];
        assert_eq!(nearest_item(&items, 200.0, 200.0), None);
    }

    #[test]
    fn spawn_loot_valid_kind() {
        let mut rng: u32 = 42;
        let item = spawn_loot(0.0, 0.0, &mut rng);
        assert!(matches!(item.kind, ItemKind::Sword | ItemKind::Staff | ItemKind::Tome));
    }
}
