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

pub enum ChestLoot {
    Item(ItemKind),
    Gold(u32),
}

pub struct Chest {
    pub x: f32,
    pub y: f32,
    pub opened: bool,
    loot: Option<ChestLoot>,
}

impl Chest {
    pub fn new(x: f32, y: f32, loot: ChestLoot) -> Self {
        Chest { x, y, opened: false, loot: Some(loot) }
    }

    /// Open the chest. Returns the loot on first call, None on subsequent calls.
    pub fn open(&mut self) -> Option<ChestLoot> {
        if self.opened {
            return None;
        }
        self.opened = true;
        self.loot.take()
    }
}

/// Spawn a random loot item: 20% HP potion, 20% MP potion, 60% equipment.
pub fn spawn_loot(x: f32, y: f32, rng: &mut u32) -> Item {
    let r = xorshift(rng) % 10;
    let kind = match r {
        0 | 1 => ItemKind::HpPotion,
        2 | 3 => ItemKind::MpPotion,
        4 | 5 | 6 => ItemKind::Sword,
        7 | 8 => ItemKind::Staff,
        _ => ItemKind::Tome,
    };
    Item::new(x, y, kind)
}

/// Spawn a random chest loot: 50% gold (10-30g), 50% random item.
pub fn spawn_chest_loot(rng: &mut u32) -> ChestLoot {
    if xorshift(rng) % 2 == 0 {
        let gold = 10 + (xorshift(rng) % 21);
        ChestLoot::Gold(gold)
    } else {
        let r = xorshift(rng) % 5;
        let kind = match r {
            0 => ItemKind::Sword,
            1 => ItemKind::Staff,
            2 => ItemKind::Tome,
            3 => ItemKind::HpPotion,
            _ => ItemKind::MpPotion,
        };
        ChestLoot::Item(kind)
    }
}

/// Returns the index of the nearest item within 45px of (px, py), or None.
pub fn nearest_item(items: &[Item], px: f32, py: f32) -> Option<usize> {
    const PICKUP_RANGE: f32 = 45.0;
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

/// Returns the index of the nearest unopened chest within 45px of (px, py), or None.
pub fn nearest_chest(chests: &[Chest], px: f32, py: f32) -> Option<usize> {
    const INTERACT_RANGE: f32 = 45.0;
    let mut best_dist = f32::MAX;
    let mut best_idx = None;
    for (i, chest) in chests.iter().enumerate() {
        if chest.opened {
            continue;
        }
        let d = distance(px, py, chest.x, chest.y);
        if d < INTERACT_RANGE && d < best_dist {
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
        assert!(matches!(
            item.kind,
            ItemKind::Sword | ItemKind::Staff | ItemKind::Tome
            | ItemKind::HpPotion | ItemKind::MpPotion
        ));
    }

    #[test]
    fn chest_open_once_returns_loot() {
        let mut rng: u32 = 1;
        let loot = spawn_chest_loot(&mut rng);
        let mut chest = Chest::new(100.0, 100.0, loot);
        assert!(!chest.opened);
        let result = chest.open();
        assert!(result.is_some());
        assert!(chest.opened);
    }

    #[test]
    fn chest_open_twice_returns_none() {
        let mut rng: u32 = 1;
        let loot = spawn_chest_loot(&mut rng);
        let mut chest = Chest::new(100.0, 100.0, loot);
        chest.open(); // first open
        let second = chest.open(); // should be None
        assert!(second.is_none());
    }

    #[test]
    fn nearest_chest_within_range() {
        let mut rng: u32 = 1;
        let loot = spawn_chest_loot(&mut rng);
        let chests = vec![Chest::new(100.0, 100.0, loot)];
        assert_eq!(nearest_chest(&chests, 110.0, 100.0), Some(0));
    }

    #[test]
    fn nearest_chest_opened_is_skipped() {
        let mut rng: u32 = 1;
        let loot = spawn_chest_loot(&mut rng);
        let mut chests = vec![Chest::new(100.0, 100.0, loot)];
        chests[0].open();
        // chest is opened — nearest_chest should return None even though in range
        assert!(nearest_chest(&chests, 110.0, 100.0).is_none());
    }

    #[test]
    fn spawn_chest_loot_produces_gold_or_item() {
        let mut rng: u32 = 99999;
        for _ in 0..20 {
            let loot = spawn_chest_loot(&mut rng);
            match loot {
                ChestLoot::Gold(g) => assert!(g >= 10 && g <= 30),
                ChestLoot::Item(k) => assert!(matches!(
                    k,
                    ItemKind::Sword | ItemKind::Staff | ItemKind::Tome
                    | ItemKind::HpPotion | ItemKind::MpPotion
                )),
            }
        }
    }
}
