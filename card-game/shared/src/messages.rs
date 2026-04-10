use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::deck::Card;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum GameType {
    Durak,
    Blackjack,
}

impl Default for GameType {
    fn default() -> Self {
        GameType::Durak
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum GameAction {
    // Durak actions
    DurakAttack { card: Card },
    DurakDefend { attack_card: Card, defend_card: Card },
    DurakTakeCards,
    DurakEndAttack,
    // Blackjack actions
    BlackjackHit,
    BlackjackStand,
}

/// Messages sent from client → server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    Join { username: String },
    Rejoin { player_id: Uuid, room_id: Uuid },
    InvitePlayer { target_id: Uuid, game: GameType },
    AcceptInvite { room_id: Uuid },
    DeclineInvite { room_id: Uuid },
    PlayBot { game: GameType },
    GameMove { room_id: Uuid, action: GameAction },
    ChatMessage { text: String },
}

/// Lightweight player info sent to clients in lobby updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LobbyPlayer {
    pub id: Uuid,
    pub username: String,
    pub available: bool,
}

/// Outcome of a Blackjack hand
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Outcome {
    Win,
    Lose,
    Push,
    Blackjack,
}

/// Messages sent from server → client
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    // Connection lifecycle
    Welcome { player_id: Uuid },
    LobbyUpdate { players: Vec<LobbyPlayer> },

    // Invitation flow
    IncomingInvite { from: String, from_id: Uuid, room_id: Uuid, game: GameType },
    InviteDeclined { by: String },

    // Game lifecycle
    GameStarted {
        room_id: Uuid,
        game: GameType,
        your_hand: Vec<Card>,
        opponent_name: String,
        trump: Option<Card>,
        deck_remaining: u8,
        you_attack_first: bool,
    },

    // Durak events
    CardAttacked { card: Card },
    CardDefended { attack_card: Card, defend_card: Card },
    CardsTaken { cards: Vec<Card> },
    TurnEnded {
        you_attack: bool,
        your_new_cards: Vec<Card>,
        deck_remaining: u8,
    },
    OpponentHandCount { count: u8 },

    // Blackjack events
    DealerCard { card: Card, hidden: bool },
    PlayerCard { card: Card },
    DealerRevealed { card: Card },
    HandResult { outcome: Outcome, your_score: u8, dealer_score: u8 },

    // Shared
    GameOver { winner: Option<Uuid>, reason: String },
    ChatReceived { from: String, text: String },
    Error { msg: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::{Card, Rank, Suit};

    fn round_trip_client<T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug>(
        msg: T,
    ) {
        let json = serde_json::to_string(&msg).expect("serialize failed");
        let back: T = serde_json::from_str(&json).expect("deserialize failed");
        // We can't derive PartialEq on ClientMessage/ServerMessage easily due to Card,
        // so just check re-serialization is stable
        let json2 = serde_json::to_string(&back).expect("re-serialize failed");
        assert_eq!(json, json2, "round-trip not stable for {:?}", msg);
    }

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn client_join_round_trip() {
        let msg = ClientMessage::Join { username: "Alice".into() };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ClientMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn client_rejoin_round_trip() {
        let msg = ClientMessage::Rejoin {
            player_id: Uuid::new_v4(),
            room_id: Uuid::new_v4(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ClientMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn client_game_move_durak_attack_round_trip() {
        let msg = ClientMessage::GameMove {
            room_id: Uuid::new_v4(),
            action: GameAction::DurakAttack {
                card: card(Rank::Ace, Suit::Spades),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ClientMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn client_game_move_durak_defend_round_trip() {
        let msg = ClientMessage::GameMove {
            room_id: Uuid::new_v4(),
            action: GameAction::DurakDefend {
                attack_card: card(Rank::Seven, Suit::Hearts),
                defend_card: card(Rank::King, Suit::Hearts),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ClientMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn client_game_move_take_cards_round_trip() {
        let msg = ClientMessage::GameMove {
            room_id: Uuid::new_v4(),
            action: GameAction::DurakTakeCards,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ClientMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn client_game_move_blackjack_hit_round_trip() {
        let msg = ClientMessage::GameMove {
            room_id: Uuid::new_v4(),
            action: GameAction::BlackjackHit,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ClientMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn client_chat_round_trip() {
        let msg = ClientMessage::ChatMessage { text: "gg wp".into() };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ClientMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn server_welcome_round_trip() {
        let msg = ServerMessage::Welcome { player_id: Uuid::new_v4() };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ServerMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn server_lobby_update_round_trip() {
        let msg = ServerMessage::LobbyUpdate {
            players: vec![
                LobbyPlayer { id: Uuid::new_v4(), username: "Bob".into(), available: true },
            ],
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ServerMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn server_game_over_round_trip() {
        let msg = ServerMessage::GameOver {
            winner: Some(Uuid::new_v4()),
            reason: "opponent disconnected".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ServerMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn server_error_round_trip() {
        let msg = ServerMessage::Error { msg: "Username taken".into() };
        let json = serde_json::to_string(&msg).unwrap();
        let back: ServerMessage = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }
}
