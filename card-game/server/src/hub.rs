use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use shared::messages::*;
use shared::durak::DurakGame;
use shared::blackjack::BlackjackGame;
use shared::poker::{TexasPokerGame, PokerAction, PokerError};
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
    TexasPoker(TexasPokerGame),
}

pub struct GameRoom {
    pub id: Uuid,
    pub game_type: GameType,
    pub players: Vec<Uuid>,  // all player IDs
    pub is_bot: Vec<bool>,   // true if that slot is a bot
    pub game: GameInstance,
}

/// Multi-player invite state: tracks who has accepted, who still needs to respond.
pub struct PendingRoom {
    pub initiator_id: Uuid,
    pub game_type: GameType,
    pub accepted: Vec<(Uuid, bool)>,   // (id, is_bot) — committed players
    pub pending: Vec<Uuid>,            // still waiting to accept/decline
    pub capacity: usize,
}

pub struct AppState {
    pub players: HashMap<Uuid, PlayerInfo>,
    pub rooms: HashMap<Uuid, GameRoom>,
    pub lobby: HashSet<Uuid>,
    pub usernames: HashSet<String>,
    // room_id → pending room state
    pub pending_rooms: HashMap<Uuid, PendingRoom>,
}

impl AppState {
    pub fn new() -> Self {
        AppState {
            players: HashMap::new(),
            rooms: HashMap::new(),
            lobby: HashSet::new(),
            usernames: HashSet::new(),
            pending_rooms: HashMap::new(),
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
        ClientMessage::InviteToRoom { room_id, target_id } => {
            handle_invite_to_room(player_id, room_id, target_id, state).await;
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
        ClientMessage::ForfeitGame { room_id } => {
            handle_forfeit_game(player_id, room_id, state).await;
        }
        ClientMessage::VoiceSignal { to, signal_type, payload } => {
            handle_voice_signal(player_id, to, signal_type, payload, state).await;
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

    let welcome = ServerMessage::Welcome { player_id };
    let _ = tx.send(to_json(&welcome)).await;

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

    if !s.players.contains_key(&stored_id) || !s.rooms.contains_key(&room_id) {
        let _ = tx.send(error_json("Session expired — please rejoin")).await;
        return;
    }

    if let Some(player) = s.players.get_mut(&stored_id) {
        player.tx = tx.clone();
    }

    let welcome = ServerMessage::Welcome { player_id: stored_id };
    let _ = tx.send(to_json(&welcome)).await;
    let _ = new_conn_id; // new socket ID not used further
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

    // Determine capacity: poker = 4, others = 2
    let capacity = if matches!(game, GameType::TexasPoker) { 4 } else { 2 };

    s.pending_rooms.insert(room_id, PendingRoom {
        initiator_id: inviter_id,
        game_type: game.clone(),
        accepted: vec![(inviter_id, false)],
        pending: vec![target_id],
        capacity,
    });

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

async fn handle_invite_to_room(
    inviter_id: Uuid,
    room_id: Uuid,
    target_id: Uuid,
    state: &crate::AppStateHandle,
) {
    let mut s = state.write().await;

    // Validate pending room and membership
    let (is_member, already_invited) = match s.pending_rooms.get(&room_id) {
        Some(pr) => (
            pr.accepted.iter().any(|(id, _)| *id == inviter_id),
            pr.pending.contains(&target_id) || pr.accepted.iter().any(|(id, _)| *id == target_id),
        ),
        None => {
            send_to(&s, inviter_id, &ServerMessage::Error { msg: "Room not found or already started".into() });
            return;
        }
    };

    if !is_member {
        send_to(&s, inviter_id, &ServerMessage::Error { msg: "You are not in this room".into() });
        return;
    }
    if already_invited {
        send_to(&s, inviter_id, &ServerMessage::Error { msg: "Player already invited".into() });
        return;
    }

    // Check target is available
    let target_available = s.players.get(&target_id)
        .map(|p| p.status == PlayerStatus::Lobby)
        .unwrap_or(false);
    if !target_available {
        send_to(&s, inviter_id, &ServerMessage::Error { msg: "Player is busy".into() });
        return;
    }

    let inviter_name = s.players.get(&inviter_id).map(|p| p.username.clone()).unwrap_or_default();
    let game = s.pending_rooms.get(&room_id).map(|pr| pr.game_type.clone()).unwrap();

    s.pending_rooms.get_mut(&room_id).unwrap().pending.push(target_id);

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

    let pr = match s.pending_rooms.get_mut(&room_id) {
        Some(pr) => pr,
        None => {
            send_to(&s, accepter_id, &ServerMessage::Error { msg: "Invite expired".into() });
            return;
        }
    };

    // Remove from pending
    if let Some(pos) = pr.pending.iter().position(|&id| id == accepter_id) {
        pr.pending.remove(pos);
    } else {
        return; // not in pending list
    }

    pr.accepted.push((accepter_id, false));

    // Check if we have enough players to start
    if pr.accepted.len() >= pr.capacity || pr.pending.is_empty() {
        let pr = s.pending_rooms.remove(&room_id).unwrap();
        let players: Vec<(Uuid, bool)> = pr.accepted;
        start_game(&mut s, room_id, pr.game_type, players);
    }
    // else: still waiting for more accepts
}

async fn handle_decline_invite(
    decliner_id: Uuid,
    room_id: Uuid,
    state: &crate::AppStateHandle,
) {
    let mut s = state.write().await;

    let pr = match s.pending_rooms.remove(&room_id) {
        Some(pr) => pr,
        None => return,
    };

    let decliner_name = s.players.get(&decliner_id)
        .map(|p| p.username.clone())
        .unwrap_or_default();

    let initiator_id = pr.initiator_id;

    // Notify all accepted players that invite was declined
    for (pid, _) in &pr.accepted {
        send_to(&s, *pid, &ServerMessage::InviteDeclined { by: decliner_name.clone() });
        if let Some(p) = s.players.get_mut(pid) {
            p.status = PlayerStatus::Lobby;
        }
    }

    // Reset initiator status
    if let Some(p) = s.players.get_mut(&initiator_id) {
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
    let (bot_ids, players): (Vec<Uuid>, Vec<(Uuid, bool)>) = match &game {
        GameType::TexasPoker => {
            // 1 human + 3 bots
            let b1 = Uuid::new_v4();
            let b2 = Uuid::new_v4();
            let b3 = Uuid::new_v4();
            (
                vec![b1, b2, b3],
                vec![(player_id, false), (b1, true), (b2, true), (b3, true)],
            )
        }
        _ => {
            let bot_id = Uuid::new_v4();
            (vec![bot_id], vec![(player_id, false), (bot_id, true)])
        }
    };

    {
        let mut s = state.write().await;
        // Register bot placeholder entries so send_to doesn't fail lookups
        for bot_id in &bot_ids {
            let (bot_tx, _bot_rx) = mpsc::channel::<String>(1);
            s.players.insert(*bot_id, PlayerInfo {
                id: *bot_id,
                username: "Bot".into(),
                status: PlayerStatus::Lobby,
                tx: bot_tx,
            });
        }
        start_game(&mut s, room_id, game, players);
    }

    trigger_bot_move_if_needed(room_id, state).await;
}

async fn handle_forfeit_game(
    player_id: Uuid,
    room_id: Uuid,
    state: &crate::AppStateHandle,
) {
    let need_bot_trigger = {
        let mut s = state.write().await;

        let room = match s.rooms.get(&room_id) {
            Some(r) => r,
            None => return,
        };

        if !room.players.contains(&player_id) {
            return;
        }

        match &room.game {
            GameInstance::Durak(_) | GameInstance::Blackjack(_) => {
                let human_opponents: Vec<Uuid> = room.players.iter().enumerate()
                    .filter(|(i, &pid)| pid != player_id && !room.is_bot[*i])
                    .map(|(_, &pid)| pid)
                    .collect();
                let winner = human_opponents.first().copied();
                let forfeiter_name = s.players.get(&player_id)
                    .map(|p| p.username.clone())
                    .unwrap_or_default();
                end_game(&mut s, room_id, winner, format!("{} forfeited", forfeiter_name));
                false
            }
            GameInstance::TexasPoker(_) => {
                let player_idx = room.players.iter().position(|&id| id == player_id);
                if let Some(idx) = player_idx {
                    let room = s.rooms.get_mut(&room_id).unwrap();
                    if let GameInstance::TexasPoker(g) = &mut room.game {
                        g.forfeit_player(player_id);
                        let active_count = g.players.iter().filter(|p| p.active).count();
                        if active_count <= 1 {
                            let awards = g.award_pot();
                            let winner = awards.first().map(|(id, _)| *id);
                            let pot_won = awards.first().map(|(_, chips)| *chips).unwrap_or(0);
                            let community = g.community_cards.clone();
                            let hands: Vec<(Uuid, Vec<shared::deck::Card>, shared::poker::HandRank)> =
                                g.players.iter().filter(|p| !p.folded).map(|p| {
                                    let (rank, best) = shared::poker::evaluate_hand(&p.hole_cards, &community);
                                    (p.id, best, rank)
                                }).collect();
                            if let Some(w) = winner {
                                let showdown_msg = ServerMessage::PokerShowdown { hands, winner_id: w, pot_won };
                                broadcast_room(&s, room_id, &showdown_msg);
                            }
                            let forfeiter_name = s.players.get(&player_id)
                                .map(|p| p.username.clone())
                                .unwrap_or_default();
                            end_game(&mut s, room_id, winner, format!("{} forfeited", forfeiter_name));
                            false
                        } else {
                            // Fold this player; bots will need to continue playing
                            let msg = ServerMessage::PokerPlayerFolded { player_id };
                            broadcast_room(&s, room_id, &msg);
                            send_poker_state_update(&mut s, room_id, idx);
                            true // need bot trigger
                        }
                    } else { false }
                } else { false }
            }
        }
    }; // write lock released here

    if need_bot_trigger {
        trigger_bot_move_if_needed(room_id, state).await;
    }
}

async fn handle_voice_signal(
    from_id: Uuid,
    to_id: Uuid,
    signal_type: String,
    payload: String,
    state: &crate::AppStateHandle,
) {
    let s = state.read().await;

    // Validate both players are in the same room
    let from_room = match s.players.get(&from_id) {
        Some(p) => match &p.status {
            PlayerStatus::InGame(rid) => *rid,
            _ => return,
        },
        None => return,
    };
    let to_room = match s.players.get(&to_id) {
        Some(p) => match &p.status {
            PlayerStatus::InGame(rid) => *rid,
            _ => return,
        },
        None => return,
    };

    if from_room != to_room {
        return; // security: same-room only
    }

    send_to(&s, to_id, &ServerMessage::VoiceSignalRelayed { from: from_id, signal_type, payload });
}

/// After game starts or after a human move, trigger the bot's next move if it's the bot's turn.
async fn trigger_bot_move_if_needed(room_id: Uuid, state: &crate::AppStateHandle) {
    loop {
        let bot_action = {
            let s = state.read().await;
            let room = match s.rooms.get(&room_id) {
                Some(r) => r,
                None => return,
            };

            match &room.game {
                GameInstance::Durak(g) => {
                    // Find a bot whose turn it is
                    let bot_idx = room.is_bot.iter().enumerate()
                        .find(|(_, &is_bot)| is_bot)
                        .map(|(i, _)| i);
                    if let Some(bidx) = bot_idx {
                        let bot_should_move =
                            (g.attacker == bidx && matches!(g.state, shared::durak::DurakState::PlayerAttacks))
                            || (g.defender() == bidx && matches!(g.state, shared::durak::DurakState::PlayerDefends));
                        if bot_should_move {
                            bot::durak_move(g, bidx).map(|a| (a, bidx, None::<Uuid>))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                GameInstance::TexasPoker(g) => {
                    let action_player_id = g.action_player_id();
                    if let Some(apid) = action_player_id {
                        let pidx = room.players.iter().position(|&id| id == apid);
                        if let Some(idx) = pidx {
                            if room.is_bot[idx] {
                                let action = bot::poker_move(g);
                                Some((GameAction::PokerCall, 0, Some(apid))) // placeholder; replaced below
                                    .map(|_| (GameAction::PokerCall, idx, Some(apid)))
                                    .map(|(_, i, pid)| (poker_action_to_game_action(&action), i, pid))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        match bot_action {
            Some((GameAction::DurakAttack { .. } | GameAction::DurakDefend { .. }
                  | GameAction::DurakTakeCards | GameAction::DurakEndAttack, bot_idx, _)) => {
                let mut s = state.write().await;
                handle_durak_move(&mut s, room_id, bot_idx, bot_action.unwrap().0).await;
                break;
            }
            Some((action, _, Some(bot_player_id))) => {
                // Poker bot move
                let mut s = state.write().await;
                handle_poker_move(&mut s, room_id, bot_player_id, action).await;
                // Check if next action is also a bot — loop again
                let is_next_bot = {
                    let room = s.rooms.get(&room_id);
                    room.and_then(|r| {
                        if let GameInstance::TexasPoker(g) = &r.game {
                            g.action_player_id().and_then(|apid| {
                                r.players.iter().position(|&id| id == apid).map(|i| r.is_bot[i])
                            })
                        } else {
                            None
                        }
                    }).unwrap_or(false)
                };
                if !is_next_bot {
                    break;
                }
                // else loop again for next bot
            }
            _ => break,
        }
    }
}

fn poker_action_to_game_action(action: &bot::PokerBotAction) -> GameAction {
    match action {
        bot::PokerBotAction::Fold => GameAction::PokerFold,
        bot::PokerBotAction::Check => GameAction::PokerCheck,
        bot::PokerBotAction::Call => GameAction::PokerCall,
        bot::PokerBotAction::Raise(amount) => GameAction::PokerRaise { amount: *amount },
    }
}

fn start_game(
    s: &mut AppState,
    room_id: Uuid,
    game_type: GameType,
    players: Vec<(Uuid, bool)>, // (id, is_bot)
) {
    let player_ids: Vec<Uuid> = players.iter().map(|(id, _)| *id).collect();
    let is_bot_vec: Vec<bool> = players.iter().map(|(_, b)| *b).collect();

    // Remove human players from lobby and update status
    for (pid, is_bot) in &players {
        if !is_bot {
            s.lobby.remove(pid);
        }
        if let Some(p) = s.players.get_mut(pid) {
            p.status = PlayerStatus::InGame(room_id);
        }
    }

    match &game_type {
        GameType::Durak => {
            let g = DurakGame::new();
            let h0 = g.hands[0].clone();
            let h1 = g.hands[1].clone();
            let h0_len = h0.len() as u8;
            let h1_len = h1.len() as u8;
            let trump = Some(g.trump_card);
            let remaining = g.deck.remaining() as u8;
            let p0_attacks = g.attacker == 0;

            let p0_id = player_ids[0];
            let p1_id = player_ids[1];
            let p0_bot = is_bot_vec[0];
            let p1_bot = is_bot_vec[1];
            let p0_name = s.players.get(&p0_id).map(|p| p.username.clone()).unwrap_or_default();
            let p1_name = s.players.get(&p1_id).map(|p| p.username.clone()).unwrap_or_default();

            let room = GameRoom {
                id: room_id,
                game_type: game_type.clone(),
                players: player_ids,
                is_bot: is_bot_vec,
                game: GameInstance::Durak(g),
            };
            s.rooms.insert(room_id, room);

            if !p0_bot {
                send_to(s, p0_id, &ServerMessage::GameStarted {
                    room_id,
                    game: game_type.clone(),
                    your_hand: h0,
                    opponent_name: if p1_bot { "Bot".into() } else { p1_name.clone() },
                    opponent_id: if p1_bot { None } else { Some(p1_id) },
                    trump,
                    deck_remaining: remaining,
                    you_attack_first: p0_attacks,
                });
                // Tell p0 how many cards their opponent starts with
                send_to(s, p0_id, &ServerMessage::OpponentHandCount { count: h1_len });
            }
            if !p1_bot {
                send_to(s, p1_id, &ServerMessage::GameStarted {
                    room_id,
                    game: game_type.clone(),
                    your_hand: h1,
                    opponent_name: p0_name,
                    opponent_id: if p0_bot { None } else { Some(p0_id) },
                    trump,
                    deck_remaining: remaining,
                    you_attack_first: !p0_attacks,
                });
                // Tell p1 how many cards their opponent starts with
                send_to(s, p1_id, &ServerMessage::OpponentHandCount { count: h0_len });
            }
        }

        GameType::Blackjack => {
            let g = BlackjackGame::new();
            let h0 = g.player_hand.clone();
            // dealer_hand[0] = face-down, dealer_hand[1] = face-up
            let dealer_face_down = g.dealer_hand[0];
            let dealer_face_up   = g.dealer_hand[1];
            let remaining = g.deck.remaining() as u8;

            let p0_id = player_ids[0];
            let p0_bot = is_bot_vec[0];

            let room = GameRoom {
                id: room_id,
                game_type: game_type.clone(),
                players: player_ids,
                is_bot: is_bot_vec,
                game: GameInstance::Blackjack(g),
            };
            s.rooms.insert(room_id, room);

            if !p0_bot {
                send_to(s, p0_id, &ServerMessage::GameStarted {
                    room_id,
                    game: game_type.clone(),
                    your_hand: h0,
                    opponent_name: "Dealer".into(),
                    opponent_id: None,
                    trump: None,
                    deck_remaining: remaining,
                    you_attack_first: true,
                });
                // Send initial dealer cards: one hidden, one visible
                send_to(s, p0_id, &ServerMessage::DealerCard { card: dealer_face_down, hidden: true });
                send_to(s, p0_id, &ServerMessage::DealerCard { card: dealer_face_up,   hidden: false });
            }
        }

        GameType::TexasPoker => {
            let g = TexasPokerGame::new(player_ids.clone(), 1000);
            let dealer_seat = g.dealer_seat;
            let small_blind = g.small_blind;
            let big_blind = g.big_blind;

            // Collect seat info for all players
            let seat_infos: Vec<PokerSeatInfo> = player_ids.iter().zip(is_bot_vec.iter()).map(|(pid, &is_bot)| {
                let name = s.players.get(pid).map(|p| p.username.clone()).unwrap_or("Bot".into());
                let chips = g.players.iter().find(|p| p.id == *pid).map(|p| p.chips).unwrap_or(1000);
                PokerSeatInfo { id: *pid, name, chips, is_bot }
            }).collect();

            // Build initial player info
            let players_info: Vec<PokerPlayerInfo> = g.players.iter().map(|p| PokerPlayerInfo {
                id: p.id,
                chips: p.chips,
                bet: p.bet,
                folded: p.folded,
                active: p.active,
                all_in: p.all_in,
            }).collect();

            let community_cards = g.community_cards.clone();
            let pot = g.pot;
            let current_bet = g.current_bet;
            let action_player_id = g.action_player_id().unwrap_or(player_ids[0]);

            let room = GameRoom {
                id: room_id,
                game_type: game_type.clone(),
                players: player_ids.clone(),
                is_bot: is_bot_vec.clone(),
                game: GameInstance::TexasPoker(g),
            };
            s.rooms.insert(room_id, room);

            // Send PokerGameStarted to each human player with their hole cards
            for (seat, (pid, &is_bot)) in player_ids.iter().zip(is_bot_vec.iter()).enumerate() {
                if is_bot {
                    continue;
                }
                let hole_cards = {
                    if let Some(room) = s.rooms.get(&room_id) {
                        if let GameInstance::TexasPoker(g) = &room.game {
                            g.players.iter().find(|p| p.id == *pid).map(|p| p.hole_cards)
                        } else { None }
                    } else { None }
                };
                if let Some(hole_cards) = hole_cards {
                    let chips = players_info.iter().find(|p| p.id == *pid).map(|p| p.chips).unwrap_or(1000);
                    let bet = players_info.iter().find(|p| p.id == *pid).map(|p| p.bet).unwrap_or(0);
                    send_to(s, *pid, &ServerMessage::PokerGameStarted {
                        room_id,
                        your_hole_cards: hole_cards,
                        your_seat: seat,
                        players: seat_infos.clone(),
                        dealer_seat,
                        small_blind,
                        big_blind,
                    });
                    // Send initial state update
                    send_to(s, *pid, &ServerMessage::PokerStateUpdate {
                        room_id,
                        community_cards: community_cards.clone(),
                        pot,
                        current_bet,
                        your_chips: chips,
                        your_bet: bet,
                        action_player_id,
                        dealer_seat,
                        players_info: players_info.clone(),
                    });
                }
            }
        }
    }

    broadcast_lobby_update(s);
}

async fn handle_game_move(
    player_id: Uuid,
    room_id: Uuid,
    action: GameAction,
    state: &crate::AppStateHandle,
) {
    let is_poker_action = matches!(action,
        GameAction::PokerFold | GameAction::PokerCheck | GameAction::PokerCall | GameAction::PokerRaise { .. }
    );

    {
        let mut s = state.write().await;

        let room = match s.rooms.get(&room_id) {
            Some(r) => r,
            None => {
                send_to(&s, player_id, &ServerMessage::Error { msg: "Room not found".into() });
                return;
            }
        };

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
            GameAction::PokerFold | GameAction::PokerCheck | GameAction::PokerCall | GameAction::PokerRaise { .. } => {
                handle_poker_move(&mut s, room_id, player_id, action).await;
            }
        }
    } // write lock released

    // After a poker move, trigger any bot responses
    if is_poker_action {
        trigger_bot_move_if_needed(room_id, state).await;
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

    loop {
        let room = match s.rooms.get_mut(&room_id) {
            Some(r) => r,
            None => return,
        };
        let is_bot = room.is_bot.clone();
        let game = match &mut room.game {
            GameInstance::Durak(g) => g,
            _ => return,
        };

        // Capture hand sizes before any mutation so we can compute new cards later
        let h0_before = game.hands[0].len();
        let h1_before = game.hands[1].len();

        let result = match current_action.clone() {
            GameAction::DurakAttack { card } => game.attack(current_player_idx, card),
            GameAction::DurakDefend { attack_card, defend_card } => {
                game.defend(current_player_idx, attack_card, defend_card)
            }
            GameAction::DurakTakeCards => {
                match game.take_cards(current_player_idx) {
                    Ok(cards) => {
                        // Capture updated state while game is still borrowed
                        let p0_attacks    = game.attacker == 0;
                        let deck_rem      = game.deck.remaining() as u8;
                        // new_pX = cards added to each hand (taken-table + deck-refill for
                        // defender; deck-refill only for attacker)
                        let new_p0: Vec<_> = game.hands[0][h0_before..].to_vec();
                        let new_p1: Vec<_> = game.hands[1][h1_before..].to_vec();
                        let opp0_count    = game.hands[1].len() as u8;
                        let opp1_count    = game.hands[0].len() as u8;
                        let p0_id = room.players[0];
                        let p1_id = room.players[1];
                        // game & room borrows end here (NLL)

                        broadcast_room(s, room_id, &ServerMessage::CardsTaken { cards });

                        // Send per-player TurnEnded so both sides get their new cards
                        // and correct attack/defend button state
                        if !is_bot[0] {
                            send_to(s, p0_id, &ServerMessage::TurnEnded {
                                you_attack: p0_attacks,
                                your_new_cards: new_p0,
                                deck_remaining: deck_rem,
                            });
                            send_to(s, p0_id, &ServerMessage::OpponentHandCount { count: opp0_count });
                        }
                        if !is_bot[1] {
                            send_to(s, p1_id, &ServerMessage::TurnEnded {
                                you_attack: !p0_attacks,
                                your_new_cards: new_p1,
                                deck_remaining: deck_rem,
                            });
                            send_to(s, p1_id, &ServerMessage::OpponentHandCount { count: opp1_count });
                        }

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
                let room = s.rooms.get(&room_id).unwrap();
                let game = match &room.game {
                    GameInstance::Durak(g) => g,
                    _ => return,
                };

                // DurakEndAttack: send per-player TurnEnded with their actual new cards.
                // Broadcasting a single message would give player 1 the wrong you_attack flag.
                if matches!(&current_action, GameAction::DurakEndAttack) {
                    let p0_attacks  = game.attacker == 0;
                    let deck_rem    = game.deck.remaining() as u8;
                    let new_p0: Vec<_> = game.hands[0][h0_before..].to_vec();
                    let new_p1: Vec<_> = game.hands[1][h1_before..].to_vec();
                    let opp0_count  = game.hands[1].len() as u8;
                    let opp1_count  = game.hands[0].len() as u8;
                    let p0_id = room.players[0];
                    let p1_id = room.players[1];
                    // borrows end; s can now be taken for send_to

                    if !is_bot[0] {
                        send_to(s, p0_id, &ServerMessage::TurnEnded {
                            you_attack: p0_attacks,
                            your_new_cards: new_p0,
                            deck_remaining: deck_rem,
                        });
                        send_to(s, p0_id, &ServerMessage::OpponentHandCount { count: opp0_count });
                    }
                    if !is_bot[1] {
                        send_to(s, p1_id, &ServerMessage::TurnEnded {
                            you_attack: !p0_attacks,
                            your_new_cards: new_p1,
                            deck_remaining: deck_rem,
                        });
                        send_to(s, p1_id, &ServerMessage::OpponentHandCount { count: opp1_count });
                    }
                } else {
                    let msg = match &current_action {
                        GameAction::DurakAttack { card } => ServerMessage::CardAttacked { card: *card },
                        GameAction::DurakDefend { attack_card, defend_card } => {
                            ServerMessage::CardDefended { attack_card: *attack_card, defend_card: *defend_card }
                        }
                        _ => break,
                    };
                    // After attack/defend, update opponent hand count for both players
                    let opp0_count = game.hands[1].len() as u8;
                    let opp1_count = game.hands[0].len() as u8;
                    let p0_id = room.players[0];
                    let p1_id = room.players[1];

                    broadcast_room(s, room_id, &msg);
                    if !is_bot[0] {
                        send_to(s, p0_id, &ServerMessage::OpponentHandCount { count: opp0_count });
                    }
                    if !is_bot[1] {
                        send_to(s, p1_id, &ServerMessage::OpponentHandCount { count: opp1_count });
                    }
                }

                check_durak_victory(s, room_id);

                if !s.rooms.contains_key(&room_id) {
                    break;
                }

                let room = s.rooms.get(&room_id).unwrap();
                let bot_idx = is_bot.iter().enumerate()
                    .find(|(_, &b)| b)
                    .map(|(i, _)| i);

                if let Some(bidx) = bot_idx {
                    if let GameInstance::Durak(g) = &room.game {
                        let bot_should_move =
                            (g.attacker == bidx && matches!(g.state, shared::durak::DurakState::PlayerAttacks))
                            || (g.defender() == bidx && matches!(g.state, shared::durak::DurakState::PlayerDefends));

                        if bot_should_move {
                            if let Some(ba) = bot::durak_move(g, bidx) {
                                current_action = ba;
                                current_player_idx = bidx;
                                continue;
                            }
                        }
                    }
                }
                break;
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

async fn handle_poker_move(
    s: &mut AppState,
    room_id: Uuid,
    player_id: Uuid,
    action: GameAction,
) {
    let poker_action = match &action {
        GameAction::PokerFold => PokerAction::Fold,
        GameAction::PokerCheck => PokerAction::Check,
        GameAction::PokerCall => PokerAction::Call,
        GameAction::PokerRaise { amount } => PokerAction::Raise(*amount),
        _ => return,
    };

    let result = {
        let room = match s.rooms.get_mut(&room_id) {
            Some(r) => r,
            None => return,
        };
        match &mut room.game {
            GameInstance::TexasPoker(g) => g.apply_action(player_id, poker_action),
            _ => return,
        }
    };

    match result {
        Err(PokerError::NotYourTurn) => {
            send_to(s, player_id, &ServerMessage::Error { msg: "Not your turn".into() });
            return;
        }
        Err(e) => {
            send_to(s, player_id, &ServerMessage::Error { msg: format!("{:?}", e) });
            return;
        }
        Ok(()) => {}
    }

    // If fold, notify everyone
    if matches!(action, GameAction::PokerFold) {
        broadcast_room(s, room_id, &ServerMessage::PokerPlayerFolded { player_id });
    }

    // Check for showdown
    let showdown_data = {
        let room = s.rooms.get(&room_id);
        room.and_then(|r| {
            if let GameInstance::TexasPoker(g) = &r.game {
                if matches!(g.round, shared::poker::BettingRound::Showdown) {
                    Some(g.community_cards.clone())
                } else {
                    None
                }
            } else {
                None
            }
        })
    };

    if let Some(community) = showdown_data {
        // Build showdown hands
        let (hands, awards) = {
            let room = s.rooms.get_mut(&room_id).unwrap();
            if let GameInstance::TexasPoker(g) = &mut room.game {
                let hands: Vec<(Uuid, Vec<shared::deck::Card>, shared::poker::HandRank)> =
                    g.players.iter().filter(|p| !p.folded).map(|p| {
                        let (rank, best) = shared::poker::evaluate_hand(&p.hole_cards, &community);
                        (p.id, best, rank)
                    }).collect();
                let awards = g.award_pot();
                (hands, awards)
            } else {
                return;
            }
        };

        let winner_id = awards.first().map(|(id, _)| *id).unwrap_or(player_id);
        let pot_won = awards.first().map(|(_, chips)| *chips).unwrap_or(0);

        let showdown_msg = ServerMessage::PokerShowdown { hands, winner_id, pot_won };
        broadcast_room(s, room_id, &showdown_msg);
        end_game(s, room_id, Some(winner_id), format!("Pot of {} chips awarded", pot_won));
        return;
    }

    // Send state update to all human players
    let player_indices: Vec<usize> = {
        let room = s.rooms.get(&room_id).unwrap();
        (0..room.players.len()).collect()
    };
    for idx in player_indices {
        send_poker_state_update(s, room_id, idx);
    }
}

fn send_poker_state_update(s: &mut AppState, room_id: Uuid, _player_idx: usize) {
    let room = match s.rooms.get(&room_id) {
        Some(r) => r,
        None => return,
    };
    let g = match &room.game {
        GameInstance::TexasPoker(g) => g,
        _ => return,
    };

    let community_cards = g.community_cards.clone();
    let pot = g.pot;
    let current_bet = g.current_bet;
    let action_player_id = g.action_player_id().unwrap_or(room.players[0]);
    let dealer_seat = g.dealer_seat;

    let players_info: Vec<PokerPlayerInfo> = g.players.iter().map(|p| PokerPlayerInfo {
        id: p.id,
        chips: p.chips,
        bet: p.bet,
        folded: p.folded,
        active: p.active,
        all_in: p.all_in,
    }).collect();

    // Send personalized update to each human
    let human_players: Vec<(Uuid, u32, u32)> = g.players.iter()
        .filter_map(|p| {
            let idx = room.players.iter().position(|&id| id == p.id)?;
            if room.is_bot[idx] { return None; }
            Some((p.id, p.chips, p.bet))
        })
        .collect();

    for (pid, chips, bet) in human_players {
        send_to(s, pid, &ServerMessage::PokerStateUpdate {
            room_id,
            community_cards: community_cards.clone(),
            pot,
            current_bet,
            your_chips: chips,
            your_bet: bet,
            action_player_id,
            dealer_seat,
            players_info: players_info.clone(),
        });
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
            // Clean up bot player entry
            s.players.remove(&pid);
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
        _ => return,
    };

    drop(s);

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

    if let PlayerStatus::InGame(room_id) = player.status {
        let room = s.rooms.remove(&room_id);
        if let Some(room) = room {
            for (idx, &pid) in room.players.iter().enumerate() {
                if pid == player_id || room.is_bot[idx] {
                    if room.is_bot[idx] {
                        s.players.remove(&pid);
                    }
                    continue;
                }
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

    // Clean up any pending rooms this player was part of
    s.pending_rooms.retain(|_, pr| {
        !pr.accepted.iter().any(|(id, _)| *id == player_id)
            && !pr.pending.contains(&player_id)
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
    let players: Vec<LobbyPlayer> = s.players.values()
        .filter(|p| p.username != "Bot") // skip bot placeholders
        .map(|p| {
            let game_type = match &p.status {
                PlayerStatus::InGame(room_id) => {
                    s.rooms.get(room_id).map(|r| r.game_type.clone())
                }
                _ => None,
            };
            LobbyPlayer {
                id: p.id,
                username: p.username.clone(),
                available: p.status == PlayerStatus::Lobby,
                game_type,
            }
        })
        .collect();

    let msg = ServerMessage::LobbyUpdate { players };
    for player in s.players.values() {
        if player.status == PlayerStatus::Lobby && player.username != "Bot" {
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

impl GameInstance {
    fn as_blackjack(&self) -> Option<&BlackjackGame> {
        match self {
            GameInstance::Blackjack(g) => Some(g),
            _ => None,
        }
    }
}
