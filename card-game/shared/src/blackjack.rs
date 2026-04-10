use crate::deck::{Card, Deck, Rank};
use crate::messages::Outcome;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BlackjackState {
    Dealing,
    PlayerTurn,
    DealerTurn,
    Payout,
}

#[derive(Debug, Clone)]
pub struct BlackjackGame {
    pub deck: Deck,
    pub player_hand: Vec<Card>,
    pub dealer_hand: Vec<Card>, // dealer_hand[0] is face-down until DealerTurn
    pub state: BlackjackState,
}

impl BlackjackGame {
    pub fn new() -> Self {
        let mut deck = Deck::new_52(); // already shuffled
        let player_hand = vec![deck.deal_one().unwrap(), deck.deal_one().unwrap()];
        let dealer_hand = vec![deck.deal_one().unwrap(), deck.deal_one().unwrap()];

        BlackjackGame {
            deck,
            player_hand,
            dealer_hand,
            state: BlackjackState::PlayerTurn,
        }
    }

    /// Hit: deal one card to player.
    pub fn player_hit(&mut self) -> Result<Card, &'static str> {
        if self.state != BlackjackState::PlayerTurn {
            return Err("not player's turn");
        }
        let card = self.deck.deal_one().ok_or("deck empty")?;
        self.player_hand.push(card);
        Ok(card)
    }

    /// Stand: advance to dealer's turn.
    pub fn player_stand(&mut self) -> Result<(), &'static str> {
        if self.state != BlackjackState::PlayerTurn {
            return Err("not player's turn");
        }
        self.state = BlackjackState::DealerTurn;
        Ok(())
    }

    /// Play out the dealer's hand (hits below 17, stands at 17+).
    /// Returns all cards the dealer drew.
    pub fn dealer_play(&mut self) -> Vec<Card> {
        self.state = BlackjackState::DealerTurn;
        let mut drawn = Vec::new();
        while hand_value(&self.dealer_hand) < 17 {
            if let Some(card) = self.deck.deal_one() {
                self.dealer_hand.push(card);
                drawn.push(card);
            } else {
                break;
            }
        }
        self.state = BlackjackState::Payout;
        drawn
    }

    /// Determine the final outcome for the player.
    pub fn determine_outcome(&self) -> Outcome {
        let player = hand_value(&self.player_hand);
        let dealer = hand_value(&self.dealer_hand);
        determine_winner(player, dealer, &self.player_hand)
    }
}

/// Calculate the best hand value (handles soft aces).
pub fn hand_value(hand: &[Card]) -> u8 {
    let mut total: u16 = 0;
    let mut aces = 0u8;

    for card in hand {
        let v = card.rank.blackjack_value();
        if card.rank == Rank::Ace {
            aces += 1;
        }
        total += v as u16;
    }

    // Downgrade aces from 11→1 as needed
    while total > 21 && aces > 0 {
        total -= 10;
        aces -= 1;
    }

    total.min(255) as u8
}

/// True if hand is a natural blackjack (Ace + 10-value in exactly 2 cards).
pub fn is_blackjack(hand: &[Card]) -> bool {
    if hand.len() != 2 {
        return false;
    }
    let has_ace = hand.iter().any(|c| c.rank == Rank::Ace);
    let has_ten = hand.iter().any(|c| c.rank.blackjack_value() == 10);
    has_ace && has_ten
}

/// Determine outcome given player score, dealer score, and player hand (for BJ check).
pub fn determine_winner(player: u8, dealer: u8, player_hand: &[Card]) -> Outcome {
    if player > 21 {
        return Outcome::Lose;
    }
    if dealer > 21 {
        return Outcome::Win;
    }
    if is_blackjack(player_hand) && dealer == 21 {
        // Both could be blackjack — check separately (caller has dealer_hand)
        return Outcome::Blackjack; // Natural BJ beats dealer 21 from multiple cards
    }
    if is_blackjack(player_hand) {
        return Outcome::Blackjack;
    }
    match player.cmp(&dealer) {
        std::cmp::Ordering::Greater => Outcome::Win,
        std::cmp::Ordering::Less => Outcome::Lose,
        std::cmp::Ordering::Equal => Outcome::Push,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::{Card, Rank, Suit};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn hand_value_hard_total() {
        let hand = vec![card(Rank::Seven, Suit::Hearts), card(Rank::Eight, Suit::Clubs)];
        assert_eq!(hand_value(&hand), 15);
    }

    #[test]
    fn hand_value_soft_ace_as_11() {
        let hand = vec![card(Rank::Ace, Suit::Spades), card(Rank::Seven, Suit::Hearts)];
        assert_eq!(hand_value(&hand), 18); // soft 18
    }

    #[test]
    fn hand_value_ace_downgraded_to_1() {
        let hand = vec![
            card(Rank::Ace, Suit::Spades),
            card(Rank::King, Suit::Hearts),
            card(Rank::Seven, Suit::Clubs),
        ];
        assert_eq!(hand_value(&hand), 18); // A=1, K=10, 7=7 → 18
    }

    #[test]
    fn hand_value_bust() {
        let hand = vec![
            card(Rank::King, Suit::Hearts),
            card(Rank::Queen, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ];
        assert_eq!(hand_value(&hand), 27);
    }

    #[test]
    fn hand_value_21_three_cards() {
        let hand = vec![
            card(Rank::Seven, Suit::Hearts),
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Seven, Suit::Diamonds),
        ];
        assert_eq!(hand_value(&hand), 21);
    }

    #[test]
    fn hand_value_two_aces() {
        let hand = vec![card(Rank::Ace, Suit::Spades), card(Rank::Ace, Suit::Hearts)];
        assert_eq!(hand_value(&hand), 12); // 11 + 1
    }

    #[test]
    fn blackjack_detection_ace_and_king() {
        let hand = vec![card(Rank::Ace, Suit::Spades), card(Rank::King, Suit::Hearts)];
        assert!(is_blackjack(&hand));
    }

    #[test]
    fn blackjack_not_three_cards() {
        let hand = vec![
            card(Rank::Ace, Suit::Spades),
            card(Rank::Nine, Suit::Hearts),
            card(Rank::Ace, Suit::Clubs),
        ];
        assert!(!is_blackjack(&hand));
    }

    #[test]
    fn player_bust_dealer_wins() {
        // Player busts regardless of dealer score
        let hand = vec![card(Rank::King, Suit::Hearts), card(Rank::Queen, Suit::Clubs)];
        assert_eq!(determine_winner(22, 18, &hand), Outcome::Lose);
        assert_eq!(determine_winner(22, 26, &hand), Outcome::Lose); // both bust: player loses
    }

    #[test]
    fn dealer_bust_player_wins() {
        let hand = vec![card(Rank::Nine, Suit::Hearts), card(Rank::King, Suit::Clubs)];
        assert_eq!(determine_winner(19, 22, &hand), Outcome::Win);
    }

    #[test]
    fn higher_score_wins() {
        let hand = vec![card(Rank::Nine, Suit::Hearts), card(Rank::King, Suit::Clubs)];
        assert_eq!(determine_winner(19, 18, &hand), Outcome::Win);
        assert_eq!(determine_winner(17, 19, &hand), Outcome::Lose);
    }

    #[test]
    fn push_on_equal_scores() {
        let hand = vec![card(Rank::Nine, Suit::Hearts), card(Rank::King, Suit::Clubs)];
        assert_eq!(determine_winner(19, 19, &hand), Outcome::Push);
    }

    #[test]
    fn dealer_plays_hits_below_17() {
        let mut game = BlackjackGame::new();
        // Force a low dealer hand by overwriting
        game.dealer_hand = vec![card(Rank::Six, Suit::Hearts), card(Rank::Seven, Suit::Clubs)]; // 13
        game.state = BlackjackState::DealerTurn;
        game.dealer_play();
        // Dealer must have hit at least once
        assert!(hand_value(&game.dealer_hand) >= 17 || game.deck.is_empty());
    }

    #[test]
    fn dealer_stands_at_17() {
        let mut game = BlackjackGame::new();
        game.dealer_hand = vec![card(Rank::King, Suit::Hearts), card(Rank::Seven, Suit::Clubs)]; // 17
        game.state = BlackjackState::DealerTurn;
        let drawn = game.dealer_play();
        assert_eq!(drawn.len(), 0); // no cards drawn
    }

    #[test]
    fn new_game_deals_2_cards_each() {
        let game = BlackjackGame::new();
        assert_eq!(game.player_hand.len(), 2);
        assert_eq!(game.dealer_hand.len(), 2);
    }
}
