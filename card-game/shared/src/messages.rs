use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::deck::Card;
use crate::poker::HandRank;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum GameType {
    Durak,
    Blackjack,
    TexasPoker,
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
    // Poker actions
    PokerFold,
    PokerCheck,
    PokerCall,
    PokerRaise { amount: u32 },
}

/// Messages sent from client → server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    Join { username: String },
    Rejoin { player_id: Uuid, room_id: Uuid },
    InvitePlayer { target_id: Uuid, game: GameType },
    /// Invite an additional player to an existing pending room (poker multi-invite)
    InviteToRoom { room_id: Uuid, target_id: Uuid },
    AcceptInvite { room_id: Uuid },
    DeclineInvite { room_id: Uuid },
    PlayBot { game: GameType },
    GameMove { room_id: Uuid, action: GameAction },
    /// Forfeit the current game. In poker: fold permanently (game continues if >1 active).
    ForfeitGame { room_id: Uuid },
    /// WebRTC signaling — server relays to target player (same room or lounge).
    VoiceSignal { to: Uuid, signal_type: String, payload: String },
    ChatMessage { text: String },
    /// Join the global lounge (max 10 players, must not already be in a game).
    JoinLounge,
    /// Leave the lounge and return to lobby.
    LeaveLounge,
    /// Send a text message to all lounge members.
    LoungeChat { text: String },
}

/// Lightweight player info sent to clients in lobby updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LobbyPlayer {
    pub id: Uuid,
    pub username: String,
    pub available: bool,
    /// Some(GameType) if currently in a game, None if in lobby
    pub game_type: Option<GameType>,
}

/// Per-seat info sent in PokerGameStarted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PokerSeatInfo {
    pub id: Uuid,
    pub name: String,
    pub chips: u32,
    pub is_bot: bool,
}

/// Per-player info in PokerStateUpdate (no hole cards revealed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PokerPlayerInfo {
    pub id: Uuid,
    pub chips: u32,
    pub bet: u32,
    pub folded: bool,
    pub active: bool,
    pub all_in: bool,
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

    // Game lifecycle (Durak + Blackjack)
    GameStarted {
        room_id: Uuid,
        game: GameType,
        your_hand: Vec<Card>,
        opponent_name: String,
        opponent_id: Option<Uuid>,   // Some for PvP, None vs bot
        trump: Option<Card>,
        deck_remaining: u8,
        you_attack_first: bool,
    },

    // Poker game lifecycle
    PokerGameStarted {
        room_id: Uuid,
        your_hole_cards: [Card; 2],
        your_seat: usize,
        players: Vec<PokerSeatInfo>,
        dealer_seat: usize,
        small_blind: u32,
        big_blind: u32,
    },
    PokerStateUpdate {
        room_id: Uuid,
        community_cards: Vec<Card>,
        pot: u32,
        current_bet: u32,
        your_chips: u32,
        your_bet: u32,
        action_player_id: Uuid,
        dealer_seat: usize,
        players_info: Vec<PokerPlayerInfo>,
    },
    PokerShowdown {
        hands: Vec<(Uuid, Vec<Card>, HandRank)>,
        winner_id: Uuid,
        pot_won: u32,
    },
    PokerPlayerFolded { player_id: Uuid },

    // PokerPlayerInfo embedded in PokerStateUpdate — includes all_in: bool
    // (see PokerPlayerInfo struct above)

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

    // Voice relay (same-room or same-lounge)
    VoiceSignalRelayed { from: Uuid, signal_type: String, payload: String },

    // Lounge (group chat + voice/video, max 10 players)
    /// Sent to the joining player with current member list and lounge room ID.
    LoungeJoined { room_id: Uuid, members: Vec<LobbyPlayer> },
    /// Broadcast to all lounge members when someone joins or leaves.
    LoungeUpdate { members: Vec<LobbyPlayer> },
    /// A lounge text message broadcast to all members.
    LoungeChatReceived { from: String, text: String },

    // Shared
    GameOver { winner: Option<Uuid>, reason: String },
    ChatReceived { from: String, text: String },
    Error { msg: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::{Card, Rank, Suit};

    fn round_trip_client<T: Serialize + for<'de> Deserialize<'de> + std::fmt::Debug>(
        msg: T,
    ) {
        let json = serde_json::to_string(&msg).expect("serialize failed");
        let back: T = serde_json::from_str(&json).expect("deserialize failed");
        let json2 = serde_json::to_string(&back).expect("re-serialize failed");
        assert_eq!(json, json2, "round-trip not stable for {:?}", msg);
    }

    fn card(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn client_join_round_trip() {
        let msg = ClientMessage::Join { username: "Alice".into() };
        round_trip_client(msg);
    }

    #[test]
    fn client_rejoin_round_trip() {
        let msg = ClientMessage::Rejoin {
            player_id: Uuid::new_v4(),
            room_id: Uuid::new_v4(),
        };
        round_trip_client(msg);
    }

    #[test]
    fn client_game_move_durak_attack_round_trip() {
        let msg = ClientMessage::GameMove {
            room_id: Uuid::new_v4(),
            action: GameAction::DurakAttack {
                card: card(Rank::Ace, Suit::Spades),
            },
        };
        round_trip_client(msg);
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
        round_trip_client(msg);
    }

    #[test]
    fn client_game_move_take_cards_round_trip() {
        let msg = ClientMessage::GameMove {
            room_id: Uuid::new_v4(),
            action: GameAction::DurakTakeCards,
        };
        round_trip_client(msg);
    }

    #[test]
    fn client_game_move_blackjack_hit_round_trip() {
        let msg = ClientMessage::GameMove {
            room_id: Uuid::new_v4(),
            action: GameAction::BlackjackHit,
        };
        round_trip_client(msg);
    }

    #[test]
    fn client_chat_round_trip() {
        let msg = ClientMessage::ChatMessage { text: "gg wp".into() };
        round_trip_client(msg);
    }

    // Test plan item 22: PokerGameStarted round-trip
    #[test]
    fn server_poker_game_started_round_trip() {
        let msg = ServerMessage::PokerGameStarted {
            room_id: Uuid::new_v4(),
            your_hole_cards: [card(Rank::Ace, Suit::Spades), card(Rank::King, Suit::Hearts)],
            your_seat: 0,
            players: vec![
                PokerSeatInfo { id: Uuid::new_v4(), name: "Alice".into(), chips: 1000, is_bot: false },
                PokerSeatInfo { id: Uuid::new_v4(), name: "Bob".into(), chips: 1000, is_bot: true },
            ],
            dealer_seat: 0,
            small_blind: 10,
            big_blind: 20,
        };
        round_trip_client(msg);
    }

    // Test plan item 23: VoiceSignal round-trip
    #[test]
    fn client_voice_signal_round_trip() {
        let msg = ClientMessage::VoiceSignal {
            to: Uuid::new_v4(),
            signal_type: "offer".into(),
            payload: r#"{"sdp":"v=0..."}"#.into(),
        };
        round_trip_client(msg);
    }

    // Test plan item 24: ForfeitGame round-trip
    #[test]
    fn client_forfeit_game_round_trip() {
        let msg = ClientMessage::ForfeitGame { room_id: Uuid::new_v4() };
        round_trip_client(msg);
    }

    // Test plan item 25: PokerStateUpdate round-trip
    #[test]
    fn server_poker_state_update_round_trip() {
        let msg = ServerMessage::PokerStateUpdate {
            room_id: Uuid::new_v4(),
            community_cards: vec![
                card(Rank::Ace, Suit::Spades),
                card(Rank::King, Suit::Hearts),
                card(Rank::Queen, Suit::Diamonds),
            ],
            pot: 120,
            current_bet: 40,
            your_chips: 880,
            your_bet: 40,
            action_player_id: Uuid::new_v4(),
            dealer_seat: 0,
            players_info: vec![
                PokerPlayerInfo {
                    id: Uuid::new_v4(),
                    chips: 880,
                    bet: 40,
                    folded: false,
                    active: true,
                    all_in: false,
                },
            ],
        };
        round_trip_client(msg);
    }

    // Test plan item 26: LobbyPlayer with game_type round-trip
    #[test]
    fn lobby_player_with_poker_game_type_round_trip() {
        let msg = ServerMessage::LobbyUpdate {
            players: vec![
                LobbyPlayer {
                    id: Uuid::new_v4(),
                    username: "Bob".into(),
                    available: false,
                    game_type: Some(GameType::TexasPoker),
                },
            ],
        };
        round_trip_client(msg);
    }

    #[test]
    fn server_welcome_round_trip() {
        let msg = ServerMessage::Welcome { player_id: Uuid::new_v4() };
        round_trip_client(msg);
    }

    #[test]
    fn server_lobby_update_round_trip() {
        let msg = ServerMessage::LobbyUpdate {
            players: vec![
                LobbyPlayer { id: Uuid::new_v4(), username: "Bob".into(), available: true, game_type: None },
            ],
        };
        round_trip_client(msg);
    }

    #[test]
    fn server_game_over_round_trip() {
        let msg = ServerMessage::GameOver {
            winner: Some(Uuid::new_v4()),
            reason: "opponent disconnected".into(),
        };
        round_trip_client(msg);
    }

    #[test]
    fn server_error_round_trip() {
        let msg = ServerMessage::Error { msg: "Username taken".into() };
        round_trip_client(msg);
    }
}
