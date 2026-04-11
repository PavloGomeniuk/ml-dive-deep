use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Suit {
    Spades,
    Hearts,
    Diamonds,
    Clubs,
}

impl Suit {
    pub fn symbol(&self) -> char {
        match self {
            Suit::Spades => '♠',
            Suit::Hearts => '♥',
            Suit::Diamonds => '♦',
            Suit::Clubs => '♣',
        }
    }

    pub fn is_red(&self) -> bool {
        matches!(self, Suit::Hearts | Suit::Diamonds)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Rank {
    // Full 52-card ranks (Two-Five needed for Texas Hold'em)
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

impl Rank {
    /// Numeric value for comparison. Ace is highest (14). Two is lowest (2).
    pub fn value(&self) -> u8 {
        match self {
            Rank::Two => 2,
            Rank::Three => 3,
            Rank::Four => 4,
            Rank::Five => 5,
            Rank::Six => 6,
            Rank::Seven => 7,
            Rank::Eight => 8,
            Rank::Nine => 9,
            Rank::Ten => 10,
            Rank::Jack => 11,
            Rank::Queen => 12,
            Rank::King => 13,
            Rank::Ace => 14,
        }
    }

    pub fn display(&self) -> &'static str {
        match self {
            Rank::Two => "2",
            Rank::Three => "3",
            Rank::Four => "4",
            Rank::Five => "5",
            Rank::Six => "6",
            Rank::Seven => "7",
            Rank::Eight => "8",
            Rank::Nine => "9",
            Rank::Ten => "10",
            Rank::Jack => "J",
            Rank::Queen => "Q",
            Rank::King => "K",
            Rank::Ace => "A",
        }
    }

    /// Blackjack value: face cards = 10, ace = 11 (caller handles soft ace)
    pub fn blackjack_value(&self) -> u8 {
        match self {
            Rank::Two => 2,
            Rank::Three => 3,
            Rank::Four => 4,
            Rank::Five => 5,
            Rank::Six => 6,
            Rank::Seven => 7,
            Rank::Eight => 8,
            Rank::Nine => 9,
            Rank::Ten | Rank::Jack | Rank::Queen | Rank::King => 10,
            Rank::Ace => 11,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl Card {
    pub fn new(rank: Rank, suit: Suit) -> Self {
        Card { rank, suit }
    }

    pub fn display(&self) -> String {
        format!("{}{}", self.rank.display(), self.suit.symbol())
    }
}

#[derive(Debug, Clone)]
pub struct Deck {
    cards: Vec<Card>,
}

impl Deck {
    /// Standard 36-card Durak deck: 6s through Aces
    pub fn new_36() -> Self {
        let suits = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs];
        let ranks = [
            Rank::Six,
            Rank::Seven,
            Rank::Eight,
            Rank::Nine,
            Rank::Ten,
            Rank::Jack,
            Rank::Queen,
            Rank::King,
            Rank::Ace,
        ];
        let mut cards = Vec::with_capacity(36);
        for &suit in &suits {
            for &rank in &ranks {
                cards.push(Card::new(rank, suit));
            }
        }
        Deck { cards }
    }

    /// 36-card deck shuffled fresh — used for Blackjack (6-Ace variant)
    pub fn new_52() -> Self {
        // We use the 36-card (6-Ace) deck for Blackjack. Standard enough.
        let mut deck = Self::new_36();
        deck.shuffle();
        deck
    }

    /// True 52-card deck (Two-Ace, all suits) for Texas Hold'em poker.
    pub fn new_52_full() -> Self {
        let suits = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs];
        let ranks = [
            Rank::Two, Rank::Three, Rank::Four, Rank::Five, Rank::Six,
            Rank::Seven, Rank::Eight, Rank::Nine, Rank::Ten,
            Rank::Jack, Rank::Queen, Rank::King, Rank::Ace,
        ];
        let mut cards = Vec::with_capacity(52);
        for &suit in &suits {
            for &rank in &ranks {
                cards.push(Card::new(rank, suit));
            }
        }
        let mut deck = Deck { cards };
        deck.shuffle();
        deck
    }

    pub fn shuffle(&mut self) {
        self.cards.shuffle(&mut thread_rng());
    }

    /// Deal n cards from the top of the deck. Returns fewer if deck runs low.
    pub fn deal(&mut self, n: usize) -> Vec<Card> {
        let take = n.min(self.cards.len());
        self.cards.drain(..take).collect()
    }

    /// Deal exactly one card. Returns None if deck is empty.
    pub fn deal_one(&mut self) -> Option<Card> {
        if self.cards.is_empty() {
            None
        } else {
            Some(self.cards.remove(0))
        }
    }

    pub fn remaining(&self) -> usize {
        self.cards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Peek at the bottom card (trump indicator in Durak)
    pub fn bottom_card(&self) -> Option<&Card> {
        self.cards.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn deck_36_has_36_unique_cards() {
        let deck = Deck::new_36();
        assert_eq!(deck.cards.len(), 36);
        let unique: HashSet<_> = deck.cards.iter().collect();
        assert_eq!(unique.len(), 36, "deck has duplicate cards");
    }

    #[test]
    fn deck_36_all_suits_all_ranks() {
        let deck = Deck::new_36();
        let suits = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs];
        let ranks = [
            Rank::Six, Rank::Seven, Rank::Eight, Rank::Nine, Rank::Ten,
            Rank::Jack, Rank::Queen, Rank::King, Rank::Ace,
        ];
        for &suit in &suits {
            for &rank in &ranks {
                assert!(
                    deck.cards.contains(&Card::new(rank, suit)),
                    "missing {:?} {:?}", rank, suit
                );
            }
        }
    }

    #[test]
    fn deal_returns_n_cards_and_removes_from_deck() {
        let mut deck = Deck::new_36();
        let hand = deck.deal(6);
        assert_eq!(hand.len(), 6);
        assert_eq!(deck.remaining(), 30);
    }

    #[test]
    fn deal_more_than_remaining_returns_all() {
        let mut deck = Deck::new_36();
        let _ = deck.deal(34);
        let last = deck.deal(10); // only 2 remain
        assert_eq!(last.len(), 2);
        assert!(deck.is_empty());
    }

    #[test]
    fn deal_one_reduces_deck() {
        let mut deck = Deck::new_36();
        let card = deck.deal_one();
        assert!(card.is_some());
        assert_eq!(deck.remaining(), 35);
    }

    #[test]
    fn deal_one_empty_returns_none() {
        let mut deck = Deck::new_36();
        let _ = deck.deal(36);
        assert!(deck.deal_one().is_none());
    }

    #[test]
    fn rank_ordering_is_correct() {
        assert!(Rank::Ace.value() > Rank::King.value());
        assert!(Rank::King.value() > Rank::Six.value());
        assert!(Rank::Ten.value() == 10);
    }
}
