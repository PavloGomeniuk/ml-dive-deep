use crate::entities::{ItemKind, PlayerClass};

pub struct ShopItem {
    pub kind: ItemKind,
    pub price: u32,
}

impl ShopItem {
    fn new(kind: ItemKind, price: u32) -> Self {
        ShopItem { kind, price }
    }
}

/// Items available to a Warrior: sword + both potions.
pub fn warrior_shop_items() -> Vec<ShopItem> {
    vec![
        ShopItem::new(ItemKind::Sword, 30),
        ShopItem::new(ItemKind::HpPotion, 15),
        ShopItem::new(ItemKind::MpPotion, 15),
    ]
}

/// Items available to a Magician: staff, tome + both potions.
pub fn magician_shop_items() -> Vec<ShopItem> {
    vec![
        ShopItem::new(ItemKind::Staff, 25),
        ShopItem::new(ItemKind::Tome, 20),
        ShopItem::new(ItemKind::HpPotion, 15),
        ShopItem::new(ItemKind::MpPotion, 15),
    ]
}

pub fn shop_items_for_class(class: PlayerClass) -> Vec<ShopItem> {
    match class {
        PlayerClass::Warrior => warrior_shop_items(),
        PlayerClass::Magician => magician_shop_items(),
    }
}

pub enum BuyResult {
    /// Item was purchased and added to inventory.
    Purchased,
    /// Not enough gold.
    NotEnoughGold,
    /// Equipment slot already occupied by same item type.
    SlotFull,
}

/// Attempt to buy a shop item. Deducts gold and adds to player inventory on success.
/// Returns BuyResult describing what happened.
pub fn try_buy(
    gold: &mut u32,
    equipment: &mut [Option<ItemKind>; 3],
    hp_potions: &mut u8,
    mp_potions: &mut u8,
    item: &ShopItem,
) -> BuyResult {
    if *gold < item.price {
        return BuyResult::NotEnoughGold;
    }
    match item.kind {
        ItemKind::HpPotion => {
            *gold -= item.price;
            *hp_potions = hp_potions.saturating_add(1);
            BuyResult::Purchased
        }
        ItemKind::MpPotion => {
            *gold -= item.price;
            *mp_potions = mp_potions.saturating_add(1);
            BuyResult::Purchased
        }
        equipment_kind => {
            if let Some(slot) = equipment_kind.slot() {
                *gold -= item.price;
                equipment[slot] = Some(equipment_kind);
                BuyResult::Purchased
            } else {
                BuyResult::NotEnoughGold // unreachable but safe fallback
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_equipment() -> [Option<ItemKind>; 3] {
        [None, None, None]
    }

    #[test]
    fn buy_with_sufficient_gold_succeeds() {
        let mut gold = 50u32;
        let mut eq = default_equipment();
        let mut hp = 0u8;
        let mut mp = 0u8;
        let item = ShopItem::new(ItemKind::Sword, 30);
        let result = try_buy(&mut gold, &mut eq, &mut hp, &mut mp, &item);
        assert!(matches!(result, BuyResult::Purchased));
        assert_eq!(gold, 20);
        assert_eq!(eq[0], Some(ItemKind::Sword));
    }

    #[test]
    fn buy_with_insufficient_gold_is_noop() {
        let mut gold = 5u32;
        let mut eq = default_equipment();
        let mut hp = 0u8;
        let mut mp = 0u8;
        let item = ShopItem::new(ItemKind::Sword, 30);
        let result = try_buy(&mut gold, &mut eq, &mut hp, &mut mp, &item);
        assert!(matches!(result, BuyResult::NotEnoughGold));
        // u32 must NOT have wrapped — critical regression guard
        assert_eq!(gold, 5, "gold must not underflow on failed purchase");
        assert_eq!(eq[0], None);
    }

    #[test]
    fn buy_hp_potion_increments_stack() {
        let mut gold = 50u32;
        let mut eq = default_equipment();
        let mut hp = 0u8;
        let mut mp = 0u8;
        let item = ShopItem::new(ItemKind::HpPotion, 15);
        try_buy(&mut gold, &mut eq, &mut hp, &mut mp, &item);
        assert_eq!(hp, 1);
        assert_eq!(gold, 35);
    }

    #[test]
    fn buy_mp_potion_increments_stack() {
        let mut gold = 50u32;
        let mut eq = default_equipment();
        let mut hp = 0u8;
        let mut mp = 0u8;
        let item = ShopItem::new(ItemKind::MpPotion, 15);
        try_buy(&mut gold, &mut eq, &mut hp, &mut mp, &item);
        assert_eq!(mp, 1);
        assert_eq!(gold, 35);
    }

    #[test]
    fn warrior_shop_has_sword_and_potions() {
        let items = warrior_shop_items();
        assert!(items.iter().any(|i| i.kind == ItemKind::Sword));
        assert!(items.iter().any(|i| i.kind == ItemKind::HpPotion));
        assert!(items.iter().any(|i| i.kind == ItemKind::MpPotion));
    }

    #[test]
    fn magician_shop_has_staff_and_potions() {
        let items = magician_shop_items();
        assert!(items.iter().any(|i| i.kind == ItemKind::Staff));
        assert!(items.iter().any(|i| i.kind == ItemKind::HpPotion));
        assert!(items.iter().any(|i| i.kind == ItemKind::MpPotion));
    }

    #[test]
    fn buy_with_exact_gold_succeeds() {
        let mut gold = 15u32;
        let mut eq = default_equipment();
        let mut hp = 0u8;
        let mut mp = 0u8;
        let item = ShopItem::new(ItemKind::HpPotion, 15);
        let result = try_buy(&mut gold, &mut eq, &mut hp, &mut mp, &item);
        assert!(matches!(result, BuyResult::Purchased));
        assert_eq!(gold, 0);
    }
}
