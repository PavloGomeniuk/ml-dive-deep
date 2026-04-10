use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use shared::messages::*;
use shared::durak::DurakGame;
use shared::blackjack::BlackjackGame;
use std::collections::{HashMap, HashSet};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::bot;

// ── State types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PlayerStatus {
    Lobby,
    InvitePending,
    InGame(Uuid),
}

pub struct PlayerInfo {
    pub id: Uuid,
    pub username: String,
    pub status: PlayerStatus,
    pub tx: mpsc::Sender<String>, // serialized ServerMessage JSON
}

pub enum GameInstance {
    Durak(DurakGame),
    Blackjack(BlackjackGame),
}

pub struct GameRoom {
    pub id: Uuid,
    pub game_type: GameType,
    pub players: [Uuid; 2],  // [player_index_0, player_index_1]
    pub is_bot: [bool; 2],   // true if that slot is a bot
    pub game: GameInstance,
}

pub struct AppState {
    pub players: HashMap<Uuid, PlayerInfo>,
    pub rooms: HashMap<Uuid, GameRoom>,
    pub lobby: HashSet<Uuid>,
    pub usernames: HashSet<String>,
    // pending invites: room_id → (inviter_id, invitee_id, game_type)
    pub pending_invites: HashMap<Uuid, (Uuid, Uuid, GameType)>,
}

impl AppState {
    pub fn new() -> Self {
        AppState {
            players: HashMap::new(),
            rooms: HashMap::new(),
            lobby: HashSet::new(),
            usernames: HashSet::new(),
            pending_invites: HashMap::new(),
        }
    }
}

// ── Connection entry point ───────────────────────────────────────────────────

pub async fn handle_connection(
    socket: WebSocket,
    state: crate::AppStateHandle,
) {
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::channel::<String>(64);

    // Forward outgoing messages to socket
    let write_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sink.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    let player_id = Uuid::new_v4();

    // Process incoming messages
    while let Some(Ok(msg)) = stream.next().await {
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        let client_msg: ClientMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(_) => {
                let _ = tx.send(error_json("Invalid message format")).await;
                continue;
            }
        };

        handle_message(client_msg, player_id, &tx, &state).await;
    }

    // Cleanup on disconnect
    cleanup_player(player_id, &state).await;
    write_task.abort();
}

async fn handle_message(
    msg: ClientMessage,
    player_id: Uuid,
    tx: &mpsc::Sender<String>,
    state: &crate::AppStateHandle,
) {
    match msg {
        ClientMessage::Join { username } => {
            handle_join(player_id, username, tx.clone(), state).await;
        }
        ClientMessage::Rejoin { player_id: stored_id, room_id } => {
            handle_rejoin(player_id, stored_id, room_id, tx.clone(), state).await;
        }
        ClientMessage::InvitePlayer { target_id, game } => {
            handle_invite(player_id, target_id, game, state).await;
        }
        ClientMessage::AcceptInvite { room_id } => {
            handle_accept_invite(player_id, room_id, state).await;
        }
        ClientMessage::DeclineInvite { room_id } => {
            handle_decline_invite(player_id, room_id, state).await;
        }
        ClientMessage::PlayBot { game } => {
            handle_play_bot(player_id, game, state).await;
        }
        ClientMessage::GameMove { room_id, action } => {
            handle_game_move(player_id, room_id, action, state).await;
        }
        ClientMessage::ChatMessage { text } => {
            if text.trim().is_empty() {
                return;
            }
            handle_chat(player_id, text, state).await;
        }
    }
}

// ── Handler functions ────────────────────────────────────────────────────────

async fn handle_join(
    player_id: Uuid,
    username: String,
    tx: mpsc::Sender<String>,
    state: &crate::AppStateHandle,
) {
    // Validate username
    let username = username.trim().to_string();
    if username.is_empty() {
        let _ = tx.send(error_json("Username cannot be empty")).await;
        return;
    }
    if username.len() > 32 {
        let _ = tx.send(error_json("Username too long (max 32 chars)")).await;
        return;
    }

    let mut s = state.write().await;

    if s.usernames.contains(&username) {
        let _ = tx.send(error_json("Username already taken — try another")).await;
        return;
    }

    s.usernames.insert(username.clone());
    s.players.insert(
        player_id,
        PlayerInfo {
            id: player_id,
            username,
            status: PlayerStatus::Lobby,
            tx: tx.clone(),
        },
    );
    s.lobby.insert(player_id);

    // Send welcome
    let welcome = ServerMessage::Welcome { player_id };
    let _ = tx.send(to_json(&welcome)).await;

    // Broadcast lobby update to everyone
    broadcast_lobby_update(&s);
}

async fn handle_rejoin(
    new_conn_id: Uuid,
    stored_id: Uuid,
    room_id: Uuid,
    tx: mpsc::Sender<String>,
    state: &crate::AppStateHandle,
) {
    let mut s = state.write().await;

    // Check if the old session and room still exist
    if !s.players.contains_key(&stored_id) || !s.rooms.contains_key(&room_id) {
        // Session expired — treat as new connection, send error so client shows username modal
        let _ = tx.send(error_json("Session expired — please rejoin")).await;
        return;
    }

    // Swap the sender for the existing player entry
    if let Some(player) = s.players.get_mut(&stored_id) {
        player.tx = tx.clone();
    }

    let welcome = ServerMessage::Welcome { player_id: stored_id };
    let _ = tx.send(to_json(&welcome)).await;
}

async fn handle_invite(
    inviter_id: Uuid,
    target_id: Uuid,
    game: GameType,
    state: &crate::AppStateHandle,
) {
    let mut s = state.write().await;

    let inviter_name = match s.players.get(&inviter_id) {
        Some(p) => p.username.clone(),
        None => return,
    };

    // Check target exists and is available
    let target = match s.players.get(&target_id) {
        Some(p) => p,
        None => {
            send_to(&s, inviter_id, &ServerMessage::Error { msg: "Player not found".into() });
            return;
        }
    };
    if target.status != PlayerStatus::Lobby {
        send_to(&s, inviter_id, &ServerMessage::Error { msg: "Player is busy".into() });
        return;
    }

    let room_id = Uuid::new_v4();
    s.pending_invites.insert(room_id, (inviter_id, target_id, game.clone()));

    // Mark inviter as invite-pending
    if let Some(p) = s.players.get_mut(&inviter_id) {
        p.status = PlayerStatus::InvitePending;
    }

    send_to(
        &s,
        target_id,
        &ServerMessage::IncomingInvite {
            from: inviter_name,
            from_id: inviter_id,
            room_id,
            game,
        },
    );
}

async fn handle_accept_invite(
    accepter_id: Uuid,
    room_id: Uuid,
    state: &crate::AppStateHandle,
) {
    let mut s = state.write().await;

    let (inviter_id, invitee_id, game_type) = match s.pending_invites.remove(&room_id) {
        Some(inv) => inv,
        None => {
            send_to(&s, accepter_id, &ServerMessage::Error { msg: "Invite expired".into() });
            return;
        }
    };

    if accepter_id != invitee_id {
        return; // wrong player
    }

    start_game(&mut s, room_id, game_type, inviter_id, invitee_id, false, false);
}

async fn handle_decline_invite(
    decliner_id: Uuid,
    room_id: Uuid,
    state: &crate::AppStateHandle,
) {
    let mut s = state.write().await;

    let (inviter_id, _, _) = match s.pending_invites.remove(&room_id) {
        Some(inv) => inv,
        None => return,
    };

    let decliner_name = s.players.get(&decliner_id)
        .map(|p| p.username.clone())
        .unwrap_or_default();

    // Send InviteDeclined BEFORE resetting inviter status and broadcasting lobby update.
    // This guarantees the targeted message is queued in the inviter's channel first,
    // so InviteDeclined always arrives before the subsequent LobbyUpdate.
    send_to(&s, inviter_id, &ServerMessage::InviteDeclined { by: decliner_name });

    // Now reset inviter status and broadcast so the lobby list reflects them as available.
    if let Some(p) = s.players.get_mut(&inviter_id) {
        p.status = PlayerStatus::Lobby;
    }
    broadcast_lobby_update(&s);
}

async fn handle_play_bot(
    player_id: Uuid,
    game: GameType,
    state: &crate::AppStateHandle,
) {
    let room_id = Uuid::new_v4();
    let bot_id = Uuid::new_v4();
    {
        let mut s = state.write().await;
        start_game(&mut s, room_id, game, player_id, bot_id, false, true);
    }
    // If the bot is the attacker (player index 1 is bot, so bot_idx=1 but attacker could be 0 or 1)
    // Check and fire bot's first move after releasing the write lock.
    trigger_bot_move_if_needed(room_id, state).await;
}

/// After game starts or after a human move, trigger the bot's next move if it's the bot's turn.
async fn trigger_bot_move_if_needed(room_id: Uuid, state: &crate::AppStateHandle) {
    let bot_action = {
        let s = state.read().await;
        let room = match s.rooms.get(&room_id) {
            Some(r) => r,
            None => return,
        };
        let bot_idx = if room.is_bot[1] { Some(1usize) } else if room.is_bot[0] { Some(0) } else { return };
        let bidx = bot_idx.unwrap();
        if let GameInstance::Durak(g) = &room.game {
            let bot_should_move =
                (g.attacker == bidx && matches!(g.state, shared::durak::DurakState::PlayerAttacks))
                || (g.defender() == bidx && matches!(g.state, shared::durak::DurakState::PlayerDefends));
            if bot_should_move {
                bot::durak_move(g, bidx).map(|a| (a, bidx))
            } else {
                None
            }
        } else {
            None
        }
    };

    if let Some((action, bot_idx)) = bot_action {
        let mut s = state.write().await;
        handle_durak_move(&mut s, room_id, bot_idx, action).await;
    }
}

fn start_game(
    s: &mut AppState,
    room_id: Uuid,
    game_type: GameType,
    player0_id: Uuid,
    player1_id: Uuid,
    player0_is_bot: bool,
    player1_is_bot: bool,
) {
    // Remove both from lobby
    s.lobby.remove(&player0_id);
    s.lobby.remove(&player1_id);

    // Update statuses
    for id in [player0_id, player1_id] {
        if let Some(p) = s.players.get_mut(&id) {
            p.status = PlayerStatus::InGame(room_id);
        }
    }

    let (game_instance, your_hand_0, your_hand_1, trump, deck_remaining, p0_attacks) =
        match &game_type {
            GameType::Durak => {
                let g = DurakGame::new();
                let h0 = g.hands[0].clone();
                let h1 = g.hands[1].clone();
                let trump = Some(g.trump_card);
                let remaining = g.deck.remaining() as u8;
                let p0_atk = g.attacker == 0;
                (GameInstance::Durak(g), h0, h1, trump, remaining, p0_atk)
            }
            GameType::Blackjack => {
                let g = BlackjackGame::new();
                let h0 = g.player_hand.clone();
                let h1 = vec![]; // bot/opponent hand not sent
                let remaining = g.deck.remaining() as u8;
                (GameInstance::Blackjack(g), h0, h1, None, remaining, true)
            }
        };

    let room = GameRoom {
        id: room_id,
        game_type: game_type.clone(),
        players: [player0_id, player1_id],
        is_bot: [player0_is_bot, player1_is_bot],
        game: game_instance,
    };
    s.rooms.insert(room_id, room);

    let p0_name = s.players.get(&player0_id).map(|p| p.username.clone()).unwrap_or_default();
    let p1_name = s.players.get(&player1_id).map(|p| p.username.clone()).unwrap_or_default();

    // Send GameStarted to player 0
    if !player0_is_bot {
        send_to(
            s,
            player0_id,
            &ServerMessage::GameStarted {
                room_id,
                game: game_type.clone(),
                your_hand: your_hand_0,
                opponent_name: if player1_is_bot { "Bot".into() } else { p1_name.clone() },
                trump,
                deck_remaining,
                you_attack_first: p0_attacks,
            },
        );
    }

    // Send GameStarted to player 1
    if !player1_is_bot {
        send_to(
            s,
            player1_id,
            &ServerMessage::GameStarted {
                room_id,
                game: game_type.clone(),
                your_hand: your_hand_1,
                opponent_name: p0_name,
                trump,
                deck_remaining,
                you_attack_first: !p0_attacks,
            },
        );
    }

    broadcast_lobby_update(s);

    // If player 0 is human and bot attacks first (player index 1 = bot, attacker idx 0 = player0)
    // We handle bot-goes-first case after game start.
}

async fn handle_game_move(
    player_id: Uuid,
    room_id: Uuid,
    action: GameAction,
    state: &crate::AppStateHandle,
) {
    let mut s = state.write().await;

    let room = match s.rooms.get_mut(&room_id) {
        Some(r) => r,
        None => {
            send_to(&s, player_id, &ServerMessage::Error { msg: "Room not found".into() });
            return;
        }
    };

    // Determine player's index in this room
    let player_idx = match room.players.iter().position(|&id| id == player_id) {
        Some(i) => i,
        None => {
            send_to(&s, player_id, &ServerMessage::Error { msg: "You are not in this game".into() });
            return;
        }
    };

    match &action {
        GameAction::DurakAttack { .. }
        | GameAction::DurakDefend { .. }
        | GameAction::DurakTakeCards
        | GameAction::DurakEndAttack => {
            handle_durak_move(&mut s, room_id, player_idx, action).await;
        }
        GameAction::BlackjackHit | GameAction::BlackjackStand => {
            handle_blackjack_move(&mut s, room_id, player_idx, action).await;
        }
    }
}

async fn handle_durak_move(
    s: &mut AppState,
    room_id: Uuid,
    player_idx: usize,
    initial_action: GameAction,
) {
    let mut current_action = initial_action;
    let mut current_player_idx = player_idx;

    // Loop handles the human move then bot response(s) without async recursion
    loop {
        let room = match s.rooms.get_mut(&room_id) {
            Some(r) => r,
            None => return,
        };
        let is_bot = room.is_bot;
        let game = match &mut room.game {
            GameInstance::Durak(g) => g,
            _ => return,
        };

        let result = match current_action.clone() {
            GameAction::DurakAttack { card } => game.attack(current_player_idx, card),
            GameAction::DurakDefend { attack_card, defend_card } => {
                game.defend(current_player_idx, attack_card, defend_card)
            }
            GameAction::DurakTakeCards => {
                match game.take_cards(current_player_idx) {
                    Ok(cards) => {
                        broadcast_room(s, room_id, &ServerMessage::CardsTaken { cards });
                        check_durak_victory(s, room_id);
                        break;
                    }
                    Err(e) => Err(e),
                }
            }
            GameAction::DurakEndAttack => game.end_attack(current_player_idx),
            _ => return,
        };

        match result {
            Ok(()) => {
                // Build broadcast message and send it
                let room = s.rooms.get(&room_id).unwrap();
                let game = match &room.game {
                    GameInstance::Durak(g) => g,
                    _ => return,
                };

                let msg = match &current_action {
                    GameAction::DurakAttack { card } => ServerMessage::CardAttacked { card: *card },
                    GameAction::DurakDefend { attack_card, defend_card } => {
                        ServerMessage::CardDefended { attack_card: *attack_card, defend_card: *defend_card }
                    }
                    GameAction::DurakEndAttack => {
                        ServerMessage::TurnEnded {
                            you_attack: game.attacker == 0,
                            your_new_cards: vec![],
                            deck_remaining: game.deck.remaining() as u8,
                        }
                    }
                    _ => break,
                };
                broadcast_room(s, room_id, &msg);
                check_durak_victory(s, room_id);

                // Check if game ended
                if !s.rooms.contains_key(&room_id) {
                    break;
                }

                // Check if a bot needs to move
                let room = s.rooms.get(&room_id).unwrap();
                let bot_idx = if is_bot[1] { Some(1usize) } else if is_bot[0] { Some(0) } else { None };
                if let Some(bidx) = bot_idx {
                    if let GameInstance::Durak(g) = &room.game {
                        let bot_should_move =
                            (g.attacker == bidx && matches!(g.state, shared::durak::DurakState::PlayerAttacks))
                            || (g.defender() == bidx && matches!(g.state, shared::durak::DurakState::PlayerDefends));

                        if bot_should_move {
                            if let Some(ba) = bot::durak_move(g, bidx) {
                                current_action = ba;
                                current_player_idx = bidx;
                                continue; // loop: apply bot move
                            }
                        }
                    }
                }
                break; // no bot move needed
            }
            Err(e) => {
                let player_id = s.rooms.get(&room_id).unwrap().players[current_player_idx];
                send_to(s, player_id, &ServerMessage::Error { msg: format!("{:?}", e) });
                break;
            }
        }
    }
}

async fn handle_blackjack_move(
    s: &mut AppState,
    room_id: Uuid,
    player_idx: usize,
    action: GameAction,
) {
    let room = s.rooms.get_mut(&room_id).unwrap();
    let player_id = room.players[player_idx];
    let game = match &mut room.game {
        GameInstance::Blackjack(g) => g,
        _ => return,
    };

    match action {
        GameAction::BlackjackHit => {
            match game.player_hit() {
                Ok(card) => {
                    let score = shared::blackjack::hand_value(&game.player_hand);
                    send_to(s, player_id, &ServerMessage::PlayerCard { card });
                    if score > 21 {
                        // Bust — game over
                        let dealer_score = shared::blackjack::hand_value(
                            &s.rooms.get(&room_id).unwrap().game.as_blackjack().unwrap().dealer_hand
                        );
                        let reveal = s.rooms.get(&room_id).unwrap().game.as_blackjack().unwrap().dealer_hand[0];
                        send_to(s, player_id, &ServerMessage::DealerRevealed { card: reveal });
                        send_to(s, player_id, &ServerMessage::HandResult {
                            outcome: shared::messages::Outcome::Lose,
                            your_score: score,
                            dealer_score,
                        });
                        end_game(s, room_id, None, "Player bust".into());
                    }
                }
                Err(e) => {
                    send_to(s, player_id, &ServerMessage::Error { msg: e.into() });
                }
            }
        }
        GameAction::BlackjackStand => {
            let room = s.rooms.get_mut(&room_id).unwrap();
            let game = match &mut room.game {
                GameInstance::Blackjack(g) => g,
                _ => return,
            };
            let drawn = game.dealer_play();
            let dealer_score = shared::blackjack::hand_value(&game.dealer_hand);
            let player_score = shared::blackjack::hand_value(&game.player_hand);
            let outcome = game.determine_outcome();
            let reveal_card = game.dealer_hand[0];

            for card in drawn {
                send_to(s, player_id, &ServerMessage::DealerCard { card, hidden: false });
            }
            send_to(s, player_id, &ServerMessage::DealerRevealed { card: reveal_card });
            send_to(s, player_id, &ServerMessage::HandResult {
                outcome: outcome.clone(),
                your_score: player_score,
                dealer_score,
            });

            let winner = match &outcome {
                shared::messages::Outcome::Win | shared::messages::Outcome::Blackjack => Some(player_id),
                shared::messages::Outcome::Lose => None,
                shared::messages::Outcome::Push => None,
            };
            end_game(s, room_id, winner, format!("{:?}", outcome));
        }
        _ => {}
    }
}

fn check_durak_victory(s: &mut AppState, room_id: Uuid) {
    let room = match s.rooms.get(&room_id) {
        Some(r) => r,
        None => return,
    };
    if let GameInstance::Durak(g) = &room.game {
        if let shared::durak::DurakState::Victory(winner_idx) = g.state {
            let winner_id = room.players[winner_idx];
            let reason = "Hand and deck empty — opponent is the Fool!".into();
            end_game(s, room_id, Some(winner_id), reason);
        }
    }
}

fn end_game(s: &mut AppState, room_id: Uuid, winner: Option<Uuid>, reason: String) {
    let room = match s.rooms.remove(&room_id) {
        Some(r) => r,
        None => return,
    };
    for (idx, &pid) in room.players.iter().enumerate() {
        if room.is_bot[idx] {
            continue;
        }
        if let Some(p) = s.players.get_mut(&pid) {
            p.status = PlayerStatus::Lobby;
            s.lobby.insert(pid);
        }
        send_to(s, pid, &ServerMessage::GameOver { winner, reason: reason.clone() });
    }
    broadcast_lobby_update(s);
}

async fn handle_chat(player_id: Uuid, text: String, state: &crate::AppStateHandle) {
    let s = state.read().await;
    let player = match s.players.get(&player_id) {
        Some(p) => p,
        None => return,
    };
    let username = player.username.clone();
    let room_id = match &player.status {
        PlayerStatus::InGame(rid) => *rid,
        _ => return, // no chat outside games for now
    };

    drop(s); // release read lock

    let s = state.read().await;
    if let Some(room) = s.rooms.get(&room_id) {
        let msg = ServerMessage::ChatReceived { from: username, text };
        for (idx, &pid) in room.players.iter().enumerate() {
            if !room.is_bot[idx] {
                send_to(&s, pid, &msg);
            }
        }
    }
}

// ── Cleanup on disconnect ────────────────────────────────────────────────────

pub async fn cleanup_player(player_id: Uuid, state: &crate::AppStateHandle) {
    let mut s = state.write().await;

    let player = match s.players.remove(&player_id) {
        Some(p) => p,
        None => return,
    };
    s.usernames.remove(&player.username);
    s.lobby.remove(&player_id);

    // If in a game, notify opponent and end the room
    if let PlayerStatus::InGame(room_id) = player.status {
        let room = s.rooms.remove(&room_id);
        if let Some(room) = room {
            for (idx, &pid) in room.players.iter().enumerate() {
                if pid == player_id || room.is_bot[idx] {
                    continue;
                }
                // Opponent wins by walkover
                send_to(
                    &s,
                    pid,
                    &ServerMessage::GameOver {
                        winner: Some(pid),
                        reason: format!("{} disconnected", player.username),
                    },
                );
                if let Some(p) = s.players.get_mut(&pid) {
                    p.status = PlayerStatus::Lobby;
                    s.lobby.insert(pid);
                }
            }
        }
    }

    // Clean up any pending invites this player was part of
    s.pending_invites.retain(|_, (inviter, invitee, _)| {
        *inviter != player_id && *invitee != player_id
    });

    broadcast_lobby_update(&s);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn send_to(s: &AppState, player_id: Uuid, msg: &ServerMessage) {
    if let Some(player) = s.players.get(&player_id) {
        let json = to_json(msg);
        let _ = player.tx.try_send(json);
    }
}

fn broadcast_room(s: &AppState, room_id: Uuid, msg: &ServerMessage) {
    if let Some(room) = s.rooms.get(&room_id) {
        for (idx, &pid) in room.players.iter().enumerate() {
            if !room.is_bot[idx] {
                send_to(s, pid, msg);
            }
        }
    }
}

fn broadcast_lobby_update(s: &AppState) {
    let players: Vec<LobbyPlayer> = s.players.values().map(|p| LobbyPlayer {
        id: p.id,
        username: p.username.clone(),
        available: p.status == PlayerStatus::Lobby,
    }).collect();

    let msg = ServerMessage::LobbyUpdate { players };
    // Only send to players currently in the lobby — in-game players don't need lobby updates
    // and mixing lobby updates into game message streams causes ordering confusion.
    for player in s.players.values() {
        if player.status == PlayerStatus::Lobby {
            let _ = player.tx.try_send(to_json(&msg));
        }
    }
}

fn to_json(msg: &ServerMessage) -> String {
    serde_json::to_string(msg).unwrap_or_default()
}

fn error_json(msg: &str) -> String {
    to_json(&ServerMessage::Error { msg: msg.into() })
}

// Helper to get blackjack game from room
impl GameInstance {
    fn as_blackjack(&self) -> Option<&BlackjackGame> {
        match self {
            GameInstance::Blackjack(g) => Some(g),
            _ => None,
        }
    }
}
