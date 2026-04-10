use crate::deck::{Card, Deck, Rank, Suit};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DurakState {
    Dealing,
    PlayerAttacks,   // attacker's turn to play a card
    PlayerDefends,   // defender's turn
    EndTurn,         // attacker called end — resolve table
    Victory(usize),  // player index (0 or 1) who won (other is the "fool")
}

#[derive(Debug, Clone)]
pub struct DurakGame {
    pub deck: Deck,
    pub trump: Suit,
    pub trump_card: Card,   // bottom card of deck shown as trump indicator
    pub hands: [Vec<Card>; 2],
    pub table: Vec<(Card, Option<Card>)>, // (attack_card, defend_card)
    pub attacker: usize,    // 0 or 1
    pub state: DurakState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DurakError {
    NotYourTurn,
    CardNotInHand,
    TableFull,
    InvalidAttack,   // rank not on table when table is non-empty
    InvalidDefense,  // card doesn't beat attack card
    NoCardsToDefend,
    AttackNotOver,   // tried to end attack while undefended cards remain
    GameOver,
}

impl DurakGame {
    pub fn new() -> Self {
        let mut deck = Deck::new_36();
        deck.shuffle();

        let trump_card = *deck.bottom_card().expect("deck not empty at start");
        let trump = trump_card.suit;

        let hand0 = deck.deal(6);
        let hand1 = deck.deal(6);

        // Player with lowest trump card attacks first
        let attacker = lowest_trump_holder(&hand0, &hand1, trump);

        DurakGame {
            deck,
            trump,
            trump_card,
            hands: [hand0, hand1],
            table: Vec::new(),
            attacker,
            state: DurakState::PlayerAttacks,
        }
    }

    pub fn defender(&self) -> usize {
        1 - self.attacker
    }

    /// Attacker plays a card onto the table.
    pub fn attack(&mut self, attacker_idx: usize, card: Card) -> Result<(), DurakError> {
        if self.state != DurakState::PlayerAttacks {
            return Err(DurakError::NotYourTurn);
        }
        if attacker_idx != self.attacker {
            return Err(DurakError::NotYourTurn);
        }
        if self.table.len() >= 6 {
            return Err(DurakError::TableFull);
        }
        // When table already has cards, new attack card must match a rank on the table
        if !self.table.is_empty() && !self.rank_on_table(card.rank) {
            return Err(DurakError::InvalidAttack);
        }
        let pos = self.hands[attacker_idx]
            .iter()
            .position(|c| *c == card)
            .ok_or(DurakError::CardNotInHand)?;

        self.hands[attacker_idx].remove(pos);
        self.table.push((card, None));
        self.state = DurakState::PlayerDefends;
        Ok(())
    }

    /// Defender plays a card to beat an attack card.
    pub fn defend(
        &mut self,
        defender_idx: usize,
        attack_card: Card,
        defend_card: Card,
    ) -> Result<(), DurakError> {
        if self.state != DurakState::PlayerDefends {
            return Err(DurakError::NotYourTurn);
        }
        if defender_idx != self.defender() {
            return Err(DurakError::NotYourTurn);
        }
        // Find the undefended attack card on the table
        let table_idx = self
            .table
            .iter()
            .position(|(atk, def)| *atk == attack_card && def.is_none())
            .ok_or(DurakError::NoCardsToDefend)?;

        // Validate the defense
        if !self.beats(attack_card, defend_card) {
            return Err(DurakError::InvalidDefense);
        }

        let pos = self.hands[defender_idx]
            .iter()
            .position(|c| *c == defend_card)
            .ok_or(DurakError::CardNotInHand)?;

        self.hands[defender_idx].remove(pos);
        self.table[table_idx].1 = Some(defend_card);

        // All cards on table defended → attacker can add more or end turn
        if self.all_defended() {
            self.state = DurakState::PlayerAttacks;
        }
        Ok(())
    }

    /// Defender takes all table cards (gives up defending).
    pub fn take_cards(&mut self, defender_idx: usize) -> Result<Vec<Card>, DurakError> {
        if defender_idx != self.defender() {
            return Err(DurakError::NotYourTurn);
        }
        let mut taken: Vec<Card> = Vec::new();
        for (atk, def) in self.table.drain(..) {
            taken.push(atk);
            if let Some(d) = def {
                taken.push(d);
            }
        }
        self.hands[defender_idx].extend(taken.iter().cloned());

        // Defender becomes defender again next round (attacker stays attacker)
        // Refill hands; attacker refills first
        self.refill_hands();
        self.state = DurakState::PlayerAttacks;
        self.check_victory();
        Ok(taken)
    }

    /// Attacker ends their attack. All table cards must be defended.
    pub fn end_attack(&mut self, attacker_idx: usize) -> Result<(), DurakError> {
        if self.state != DurakState::PlayerAttacks {
            return Err(DurakError::AttackNotOver);
        }
        if attacker_idx != self.attacker {
            return Err(DurakError::NotYourTurn);
        }
        if !self.all_defended() {
            return Err(DurakError::AttackNotOver);
        }
        // Clear the table (cards go to discard, not back to deck)
        self.table.clear();

        // Swap roles: defender becomes attacker
        self.attacker = self.defender();

        // Refill hands; old attacker refills first, then old defender (now attacker)
        self.refill_hands();
        self.state = DurakState::PlayerAttacks;
        self.check_victory();
        Ok(())
    }

    fn all_defended(&self) -> bool {
        self.table.iter().all(|(_, def)| def.is_some())
    }

    fn rank_on_table(&self, rank: Rank) -> bool {
        self.table.iter().any(|(atk, def)| {
            atk.rank == rank || def.as_ref().map_or(false, |d| d.rank == rank)
        })
    }

    /// Does `defend_card` beat `attack_card` under current trump?
    fn beats(&self, attack: Card, defend: Card) -> bool {
        if attack.suit == defend.suit {
            defend.rank.value() > attack.rank.value()
        } else if defend.suit == self.trump {
            // Trump beats any non-trump
            attack.suit != self.trump
        } else {
            false
        }
    }

    /// Refill both hands up to 6 cards (attacker first, then defender).
    fn refill_hands(&mut self) {
        let attacker = self.attacker;
        let defender = self.defender();
        for &player in &[attacker, defender] {
            let need = 6usize.saturating_sub(self.hands[player].len());
            if need > 0 {
                let cards = self.deck.deal(need);
                self.hands[player].extend(cards);
            }
        }
    }

    fn check_victory(&mut self) {
        // Win condition: hand empty AND deck empty
        for i in 0..2 {
            if self.hands[i].is_empty() && self.deck.is_empty() {
                self.state = DurakState::Victory(i);
                return;
            }
        }
    }
}

/// Returns index of player holding the lowest trump. Defaults to player 0.
fn lowest_trump_holder(hand0: &[Card], hand1: &[Card], trump: Suit) -> usize {
    fn lowest_trump(hand: &[Card], trump: Suit) -> Option<u8> {
        hand.iter()
            .filter(|c| c.suit == trump)
            .map(|c| c.rank.value())
            .min()
    }
    match (lowest_trump(hand0, trump), lowest_trump(hand1, trump)) {
        (None, None) => 0,
        (Some(_), None) => 0,
        (None, Some(_)) => 1,
        (Some(v0), Some(v1)) => if v0 <= v1 { 0 } else { 1 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    fn game_with_hands(
        hand0: Vec<Card>,
        hand1: Vec<Card>,
        trump: Suit,
        trump_card: Card,
    ) -> DurakGame {
        DurakGame {
            deck: Deck::new_36(),
            trump,
            trump_card,
            hands: [hand0, hand1],
            table: Vec::new(),
            attacker: 0,
            state: DurakState::PlayerAttacks,
        }
    }

    #[test]
    fn new_game_deals_6_cards_each() {
        let game = DurakGame::new();
        assert_eq!(game.hands[0].len(), 6);
        assert_eq!(game.hands[1].len(), 6);
        assert_eq!(game.deck.remaining(), 24); // 36 - 12
    }

    #[test]
    fn new_game_trump_is_bottom_card_suit() {
        let game = DurakGame::new();
        assert_eq!(game.trump, game.trump_card.suit);
    }

    #[test]
    fn attack_valid_card_removes_from_hand() {
        let c = card(Rank::Seven, Suit::Hearts);
        let mut game = game_with_hands(
            vec![c, card(Rank::Eight, Suit::Clubs)],
            vec![card(Rank::Nine, Suit::Spades), card(Rank::Ten, Suit::Diamonds)],
            Suit::Spades,
            card(Rank::Six, Suit::Spades),
        );
        game.attack(0, c).unwrap();
        assert_eq!(game.hands[0].len(), 1);
        assert_eq!(game.table.len(), 1);
        assert_eq!(game.state, DurakState::PlayerDefends);
    }

    #[test]
    fn attack_card_not_in_hand_errors() {
        let c = card(Rank::Ace, Suit::Spades);
        let mut game = game_with_hands(
            vec![card(Rank::Seven, Suit::Hearts)],
            vec![card(Rank::Nine, Suit::Spades)],
            Suit::Clubs,
            card(Rank::Six, Suit::Clubs),
        );
        assert_eq!(game.attack(0, c), Err(DurakError::CardNotInHand));
    }

    #[test]
    fn attack_table_full_errors() {
        let suits = [Suit::Hearts, Suit::Clubs, Suit::Diamonds];
        let ranks = [Rank::Six, Rank::Seven, Rank::Eight, Rank::Nine, Rank::Ten, Rank::Jack];
        let mut game = game_with_hands(
            vec![card(Rank::Queen, Suit::Spades)],
            vec![card(Rank::King, Suit::Spades)],
            Suit::Spades,
            card(Rank::Six, Suit::Spades),
        );
        // Fill table with 6 attack cards manually
        for i in 0..6 {
            let atk = Card::new(ranks[i], suits[i % 3]);
            let def = Card::new(ranks[(i + 1) % 6], suits[(i + 1) % 3]);
            game.table.push((atk, Some(def)));
        }
        game.state = DurakState::PlayerAttacks;
        let c = card(Rank::Queen, Suit::Spades);
        assert_eq!(game.attack(0, c), Err(DurakError::TableFull));
    }

    #[test]
    fn attack_rank_not_on_table_errors_when_table_nonempty() {
        let atk = card(Rank::Seven, Suit::Hearts);
        let def = card(Rank::Nine, Suit::Hearts);
        let mut game = game_with_hands(
            vec![card(Rank::King, Suit::Clubs)],
            vec![card(Rank::Ace, Suit::Spades)],
            Suit::Spades,
            card(Rank::Six, Suit::Spades),
        );
        // Table has 7/9 — new attack must match rank 7 or 9
        game.table.push((atk, Some(def)));
        game.state = DurakState::PlayerAttacks;
        let new_atk = card(Rank::King, Suit::Clubs); // King rank not on table
        assert_eq!(game.attack(0, new_atk), Err(DurakError::InvalidAttack));
    }

    #[test]
    fn defend_higher_same_suit_beats() {
        let atk = card(Rank::Seven, Suit::Hearts);
        let def_card = card(Rank::King, Suit::Hearts);
        let mut game = game_with_hands(
            vec![card(Rank::Six, Suit::Clubs)],
            vec![def_card, card(Rank::Eight, Suit::Clubs)],
            Suit::Spades,
            card(Rank::Six, Suit::Spades),
        );
        game.table.push((atk, None));
        game.state = DurakState::PlayerDefends;
        game.defend(1, atk, def_card).unwrap();
        assert!(game.table[0].1.is_some());
    }

    #[test]
    fn defend_trump_beats_non_trump() {
        let atk = card(Rank::Ace, Suit::Hearts);   // high non-trump
        let def_card = card(Rank::Six, Suit::Spades); // lowest trump
        let mut game = game_with_hands(
            vec![card(Rank::Six, Suit::Clubs)],
            vec![def_card],
            Suit::Spades,
            card(Rank::Six, Suit::Spades),
        );
        game.table.push((atk, None));
        game.state = DurakState::PlayerDefends;
        game.defend(1, atk, def_card).unwrap();
        assert!(game.table[0].1.is_some());
    }

    #[test]
    fn defend_lower_same_suit_errors() {
        let atk = card(Rank::King, Suit::Hearts);
        let def_card = card(Rank::Seven, Suit::Hearts);
        let mut game = game_with_hands(
            vec![card(Rank::Six, Suit::Clubs)],
            vec![def_card],
            Suit::Spades,
            card(Rank::Six, Suit::Spades),
        );
        game.table.push((atk, None));
        game.state = DurakState::PlayerDefends;
        assert_eq!(game.defend(1, atk, def_card), Err(DurakError::InvalidDefense));
    }

    #[test]
    fn take_cards_adds_table_to_defender_hand() {
        let atk = card(Rank::Ace, Suit::Hearts);
        let mut game = game_with_hands(
            vec![],
            vec![card(Rank::Six, Suit::Clubs)],
            Suit::Spades,
            card(Rank::Six, Suit::Spades),
        );
        game.table.push((atk, None));
        game.state = DurakState::PlayerDefends;
        let taken = game.take_cards(1).unwrap();
        assert_eq!(taken.len(), 1);
        assert!(game.hands[1].contains(&atk));
        assert!(game.table.is_empty());
    }

    #[test]
    fn end_attack_clears_table_and_swaps_roles() {
        let atk = card(Rank::Seven, Suit::Hearts);
        let def_card = card(Rank::King, Suit::Hearts);
        let mut game = game_with_hands(
            vec![card(Rank::Six, Suit::Clubs)],
            vec![],
            Suit::Spades,
            card(Rank::Six, Suit::Spades),
        );
        game.table.push((atk, Some(def_card)));
        game.state = DurakState::PlayerAttacks;
        game.end_attack(0).unwrap();
        assert!(game.table.is_empty());
        assert_eq!(game.attacker, 1); // roles swapped
    }

    #[test]
    fn end_attack_with_undefended_errors() {
        let atk = card(Rank::Seven, Suit::Hearts);
        let mut game = game_with_hands(
            vec![card(Rank::Six, Suit::Clubs)],
            vec![card(Rank::Eight, Suit::Hearts)],
            Suit::Spades,
            card(Rank::Six, Suit::Spades),
        );
        game.table.push((atk, None)); // undefended
        game.state = DurakState::PlayerAttacks;
        assert_eq!(game.end_attack(0), Err(DurakError::AttackNotOver));
    }

    #[test]
    fn victory_when_hand_and_deck_empty() {
        // Player 0 empties hand, deck is empty
        let mut game = game_with_hands(
            vec![card(Rank::Seven, Suit::Hearts)],
            vec![card(Rank::King, Suit::Hearts)],
            Suit::Spades,
            card(Rank::Six, Suit::Spades),
        );
        // Empty the deck
        game.deck = {
            let mut d = Deck::new_36();
            let _ = d.deal(36);
            d
        };
        game.table.push((card(Rank::Seven, Suit::Hearts), Some(card(Rank::King, Suit::Hearts))));
        game.hands[0].clear();
        game.hands[1].clear();
        // Manually drain to simulate end condition
        game.check_victory();
        // Both hands empty and deck empty — player 0 wins (checked first)
        assert!(matches!(game.state, DurakState::Victory(0)));
    }
}
