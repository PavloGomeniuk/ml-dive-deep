use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    Document, Element, HtmlCanvasElement, HtmlInputElement, HtmlElement,
    KeyboardEvent, MouseEvent, WebSocket, Window,
};
use js_sys;

use shared::messages::*;
use shared::deck::Card;
use crate::canvas::PokerSeatRender;

use crate::canvas::{self, GameRender, CardAnim, CARD_W, CARD_H};
use crate::net;

// ── App state ────────────────────────────────────────────────────────────────

struct App {
    ws: Option<WebSocket>,
    player_id: Option<uuid::Uuid>,
    username: String,
    current_room: Option<uuid::Uuid>,
    render: GameRender,
    anims: Vec<CardAnim>,
    selected_game: GameType,
    pending_invite: Option<(uuid::Uuid, uuid::Uuid, GameType)>, // (from_id, room_id, game)
    result_countdown: Option<i32>,  // seconds remaining before auto-return to lobby
    poker_raise_amount: u32,        // tracked for raise slider
}

impl App {
    fn new() -> Self {
        App {
            ws: None,
            player_id: None,
            username: String::new(),
            current_room: None,
            render: GameRender::new(),
            anims: Vec::new(),
            selected_game: GameType::Durak,
            pending_invite: None,
            result_countdown: None,
            poker_raise_amount: 40,
        }
    }
}

type AppHandle = Rc<RefCell<App>>;

// ── Init ─────────────────────────────────────────────────────────────────────

pub fn init() -> Result<(), JsValue> {
    let app: AppHandle = Rc::new(RefCell::new(App::new()));

    setup_username_modal(app.clone())?;
    setup_game_picker(app.clone())?;
    setup_chat(app.clone())?;
    setup_action_buttons(app.clone())?;
    setup_poker_buttons(app.clone())?;
    setup_forfeit_button(app.clone())?;
    setup_keyboard_shortcuts(app.clone())?;
    setup_canvas_events(app.clone())?;
    setup_result_buttons(app.clone())?;

    // Check localStorage for persisted username/session
    if let Some(storage) = window().local_storage().ok().flatten() {
        if let Ok(Some(name)) = storage.get_item("card_arena_username") {
            if !name.is_empty() {
                let input = input_el("username-input");
                input.set_value(&name);
            }
        }
    }

    start_raf_loop(app)?;

    Ok(())
}

fn setup_username_modal(app: AppHandle) -> Result<(), JsValue> {
    let submit_btn = el("username-submit");
    let input = input_el("username-input");

    // Auto-focus
    let _ = input.focus();

    // Submit on button click
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut()>::new(move || {
            do_join(app.clone());
        });
        submit_btn.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Submit on Enter key
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
            if e.key() == "Enter" {
                do_join(app.clone());
            }
        });
        input.add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    Ok(())
}

fn do_join(app: AppHandle) {
    let input = input_el("username-input");
    let username = input.value().trim().to_string();

    if username.is_empty() {
        set_text("username-error", "Name cannot be empty");
        return;
    }
    if username.len() > 32 {
        set_text("username-error", "Name too long (max 32 chars)");
        return;
    }

    set_text("username-error", "");
    let username_clone = username.clone();
    let app_clone = app.clone();

    // Connect WebSocket
    let ws = net::connect(move |json| {
        handle_server_message(app_clone.clone(), json);
    })
    .expect("WebSocket connect failed");

    // Wait for open then send Join
    let username_inner = username_clone.clone();
    let ws_inner = ws.clone();
    let onopen = Closure::<dyn FnMut()>::new(move || {
        net::send(&ws_inner, &ClientMessage::Join { username: username_inner.clone() });
    });
    ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
    onopen.forget();

    // Persist username
    if let Some(storage) = window().local_storage().ok().flatten() {
        let _ = storage.set_item("card_arena_username", &username_clone);
    }

    let mut a = app.borrow_mut();
    a.username = username_clone;
    a.ws = Some(ws);
}

fn setup_game_picker(app: AppHandle) -> Result<(), JsValue> {
    // Game type selection cards
    for game_id in &["pick-durak", "pick-blackjack", "pick-poker"] {
        if doc().get_element_by_id(game_id).is_none() { continue; }
        let el = el(game_id);
        let app = app.clone();
        let game_id_str = game_id.to_string();
        let cb = Closure::<dyn FnMut()>::new(move || {
            let game = match game_id_str.as_str() {
                "pick-durak" => GameType::Durak,
                "pick-blackjack" => GameType::Blackjack,
                _ => GameType::TexasPoker,
            };
            app.borrow_mut().selected_game = game;
            // Update visual selection
            for id in &["pick-durak", "pick-blackjack", "pick-poker"] {
                if let Some(e) = doc().get_element_by_id(id) {
                    e.class_list().remove_1("selected").unwrap();
                }
            }
            el_by_id(&game_id_str).class_list().add_1("selected").unwrap();
        });
        el.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Invite accept/decline banner
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut()>::new(move || {
            let (from_id, room_id, _game) = {
                let a = app.borrow();
                match &a.pending_invite {
                    Some(inv) => inv.clone(),
                    None => return,
                }
            };
            hide("invite-banner");
            // Send accept
            let ws = app.borrow().ws.clone();
            if let Some(ws) = ws {
                net::send(&ws, &ClientMessage::AcceptInvite { room_id });
            }
        });
        el("invite-accept").add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut()>::new(move || {
            let room_id = {
                let a = app.borrow();
                match &a.pending_invite {
                    Some(inv) => inv.1,
                    None => return,
                }
            };
            hide("invite-banner");
            app.borrow_mut().pending_invite = None;
            let ws = app.borrow().ws.clone();
            if let Some(ws) = ws {
                net::send(&ws, &ClientMessage::DeclineInvite { room_id });
            }
        });
        el("invite-decline").add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Waiting cancel
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut()>::new(move || {
            hide("waiting-overlay");
            // We can't cancel the invite cleanly without tracking the room_id here.
            // For now just hide the overlay — the invite expires on the server in ~30s.
        });
        el("waiting-cancel").add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    Ok(())
}

fn setup_chat(app: AppHandle) -> Result<(), JsValue> {
    let send_btn = el("chat-send");
    let input = input_el("chat-input");

    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut()>::new(move || {
            do_send_chat(app.clone());
        });
        send_btn.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
            if e.key() == "Enter" {
                do_send_chat(app.clone());
            }
        });
        input.add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    Ok(())
}

fn do_send_chat(app: AppHandle) {
    let input = input_el("chat-input");
    let text = input.value().trim().to_string();
    if text.is_empty() {
        return;
    }
    input.set_value("");
    let ws = app.borrow().ws.clone();
    if let Some(ws) = ws {
        net::send(&ws, &ClientMessage::ChatMessage { text });
    }
}

fn setup_action_buttons(app: AppHandle) -> Result<(), JsValue> {
    // Durak buttons
    btn_click("btn-attack", app.clone(), |app| {
        let (ws, room_id, selected) = {
            let a = app.borrow();
            (a.ws.clone(), a.current_room, a.render.selected_hand_idx)
        };
        if let (Some(ws), Some(room_id), Some(idx)) = (ws, room_id, selected) {
            let card = app.borrow().render.your_hand.get(idx).copied();
            if let Some(card) = card {
                net::send(&ws, &ClientMessage::GameMove {
                    room_id,
                    action: GameAction::DurakAttack { card },
                });
            }
        }
    })?;

    btn_click("btn-defend", app.clone(), |app| {
        let (ws, room_id, selected_hand, selected_atk) = {
            let a = app.borrow();
            (a.ws.clone(), a.current_room, a.render.selected_hand_idx, a.render.selected_table_attack)
        };
        if let (Some(ws), Some(room_id), Some(idx), Some(attack_card)) = (ws, room_id, selected_hand, selected_atk) {
            let defend_card = app.borrow().render.your_hand.get(idx).copied();
            if let Some(defend_card) = defend_card {
                net::send(&ws, &ClientMessage::GameMove {
                    room_id,
                    action: GameAction::DurakDefend { attack_card, defend_card },
                });
            }
        }
    })?;

    btn_click("btn-take", app.clone(), |app| {
        let (ws, room_id) = {
            let a = app.borrow();
            (a.ws.clone(), a.current_room)
        };
        if let (Some(ws), Some(room_id)) = (ws, room_id) {
            net::send(&ws, &ClientMessage::GameMove {
                room_id,
                action: GameAction::DurakTakeCards,
            });
        }
    })?;

    btn_click("btn-end-attack", app.clone(), |app| {
        let (ws, room_id) = {
            let a = app.borrow();
            (a.ws.clone(), a.current_room)
        };
        if let (Some(ws), Some(room_id)) = (ws, room_id) {
            net::send(&ws, &ClientMessage::GameMove {
                room_id,
                action: GameAction::DurakEndAttack,
            });
        }
    })?;

    // Blackjack buttons
    btn_click("btn-hit", app.clone(), |app| {
        let (ws, room_id) = {
            let a = app.borrow();
            (a.ws.clone(), a.current_room)
        };
        if let (Some(ws), Some(room_id)) = (ws, room_id) {
            net::send(&ws, &ClientMessage::GameMove {
                room_id,
                action: GameAction::BlackjackHit,
            });
        }
    })?;

    btn_click("btn-stand", app.clone(), |app| {
        let (ws, room_id) = {
            let a = app.borrow();
            (a.ws.clone(), a.current_room)
        };
        if let (Some(ws), Some(room_id)) = (ws, room_id) {
            net::send(&ws, &ClientMessage::GameMove {
                room_id,
                action: GameAction::BlackjackStand,
            });
        }
    })?;

    Ok(())
}

fn btn_click(id: &'static str, app: AppHandle, f: impl Fn(AppHandle) + 'static) -> Result<(), JsValue> {
    let cb = Closure::<dyn FnMut()>::new(move || f(app.clone()));
    el(id).add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())?;
    cb.forget();
    Ok(())
}

fn setup_canvas_events(app: AppHandle) -> Result<(), JsValue> {
    let canvas = canvas_el();

    // Mouse move — hover
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            let canvas = canvas_el();
            let rect = canvas.get_bounding_client_rect();
            let mx = e.client_x() as f64 - rect.left();
            let my = e.client_y() as f64 - rect.top();
            let w = canvas.width() as f64;
            let h = canvas.height() as f64;
            let idx = canvas::hit_test_hand(&app.borrow().render, w, h, mx, my);
            app.borrow_mut().render.hover_hand_idx = idx;
        });
        canvas.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Click — select card or table attack card
    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            let canvas = canvas_el();
            let rect = canvas.get_bounding_client_rect();
            let mx = e.client_x() as f64 - rect.left();
            let my = e.client_y() as f64 - rect.top();
            let w = canvas.width() as f64;
            let h = canvas.height() as f64;

            let hand_idx = canvas::hit_test_hand(&app.borrow().render, w, h, mx, my);
            let table_card = canvas::hit_test_table(&app.borrow().render, w, h, mx, my);

            let mut a = app.borrow_mut();
            if let Some(idx) = hand_idx {
                a.render.selected_hand_idx = Some(idx);
            } else if let Some(atk) = table_card {
                a.render.selected_table_attack = Some(atk);
            } else {
                a.render.selected_hand_idx = None;
                a.render.selected_table_attack = None;
            }
            update_action_buttons(&a);
        });
        canvas.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Canvas resize
    {
        let cb = Closure::<dyn FnMut()>::new(move || {
            canvas::resize_canvas(&canvas_el());
        });
        window().add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    Ok(())
}

fn setup_result_buttons(app: AppHandle) -> Result<(), JsValue> {
    btn_click("result-play-again", app.clone(), |app| {
        hide("result-overlay");
        show("picker-screen");
    })?;
    btn_click("result-lobby", app.clone(), |_app| {
        hide("result-overlay");
        show("picker-screen");
    })?;
    Ok(())
}

// ── rAF loop ─────────────────────────────────────────────────────────────────

fn start_raf_loop(app: AppHandle) -> Result<(), JsValue> {
    let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move |ts: f64| {
        let canvas = canvas_el();

        // Resize if needed
        if canvas.width() == 0 {
            canvas::resize_canvas(&canvas);
        }

        {
            let mut a = app.borrow_mut();
            // Prune finished animations
            a.anims.retain(|anim| ts < anim.start_ts + anim.duration);
        }

        let a = app.borrow();
        canvas::render(&canvas, &a.render, &a.anims, ts);

        // Re-schedule
        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut(f64)>));

    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}

fn request_animation_frame(f: &Closure<dyn FnMut(f64)>) {
    window()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("request_animation_frame failed");
}

// ── Server message handler ────────────────────────────────────────────────────

fn handle_server_message(app: AppHandle, json: String) {
    let msg: ServerMessage = match serde_json::from_str(&json) {
        Ok(m) => m,
        Err(e) => {
            web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(
                &format!("Parse error: {} -- {}", e, json)
            ));
            return;
        }
    };

    match msg {
        ServerMessage::Welcome { player_id } => {
            app.borrow_mut().player_id = Some(player_id);
            hide("username-modal");
            let username = app.borrow().username.clone();
            set_text("picker-greeting", &format!("Hello, {}", username));
            show("picker-screen");
        }

        ServerMessage::LobbyUpdate { players } => {
            update_player_list(&app, players);
        }

        ServerMessage::IncomingInvite { from, from_id, room_id, game } => {
            app.borrow_mut().pending_invite = Some((from_id, room_id, game.clone()));
            let game_name = match &game {
                GameType::Durak => "Durak",
                GameType::Blackjack => "Blackjack",
                GameType::TexasPoker => "Texas Hold'em",
            };
            set_text("invite-text", &format!("{} wants to play {}", from, game_name));
            show("invite-banner");
        }

        ServerMessage::InviteDeclined { by } => {
            hide("waiting-overlay");
            // Show brief error in picker
            show_picker_error(&format!("{} declined your invite", by));
        }

        ServerMessage::GameStarted { room_id, game, your_hand, opponent_name, trump, deck_remaining, you_attack_first } => {
            hide("picker-screen");
            hide("waiting-overlay");
            hide("invite-banner");
            show("game-screen");
            show("chat-panel");

            let mut a = app.borrow_mut();
            a.current_room = Some(room_id);
            a.render = GameRender::new();
            a.render.game_type = game.clone();
            a.render.your_hand = your_hand;
            a.render.trump_card = trump;
            a.render.deck_remaining = deck_remaining;
            a.render.is_attacker = you_attack_first;
            a.render.your_turn = you_attack_first;

            set_text("sb-opponent", &opponent_name);
            if let Some(t) = trump {
                set_text("sb-trump", &format!("{}{}", t.rank.display(), t.suit.symbol()));
            }
            set_text("sb-deck", &deck_remaining.to_string());
            set_text("sb-turn", if you_attack_first { "Your turn" } else { "Opponent's turn" });

            // Show game-appropriate buttons
            match game {
                GameType::Durak => {
                    show_durak_buttons(you_attack_first);
                    show("action-bar");
                }
                GameType::Blackjack => {
                    show_blackjack_buttons(true);
                    show("action-bar");
                }
                GameType::TexasPoker => {
                    // Poker handled via PokerGameStarted
                }
            }

            canvas::resize_canvas(&canvas_el());
            append_chat_system(&format!("Game started vs {}", opponent_name));
        }

        ServerMessage::CardAttacked { card } => {
            {
                let mut a = app.borrow_mut();
                // Remove from attacker's hand if it's ours
                // The server already validated; we just update display state
                a.render.table.push((card, None));
                // Remove from our hand if we played it
                a.render.your_hand.retain(|c| *c != card);
                a.render.selected_hand_idx = None;
                set_text("sb-turn", "Defend");
            }
            update_action_buttons(&app.borrow());
        }

        ServerMessage::CardDefended { attack_card, defend_card } => {
            {
                let mut a = app.borrow_mut();
                for (atk, def) in &mut a.render.table {
                    if *atk == attack_card && def.is_none() {
                        *def = Some(defend_card);
                        break;
                    }
                }
                a.render.your_hand.retain(|c| *c != defend_card);
                a.render.selected_hand_idx = None;
                a.render.selected_table_attack = None;
            }
            update_action_buttons(&app.borrow());
        }

        ServerMessage::CardsTaken { cards } => {
            let mut a = app.borrow_mut();
            // If these are ours (we took), add to hand
            // If opponent took, they just get added to their count
            // Simple heuristic: if our hand count stays the same or increases, it was ours
            a.render.table.clear();
            // We receive all cards so we can tell if we're the ones taking
            // The server should send us our new hand in TurnEnded; for now just clear table
            set_text("sb-turn", "Cards taken — your attack next");
        }

        ServerMessage::TurnEnded { you_attack, your_new_cards, deck_remaining } => {
            let mut a = app.borrow_mut();
            a.render.table.clear();
            a.render.your_hand.extend(your_new_cards);
            a.render.is_attacker = you_attack;
            a.render.your_turn = you_attack;
            a.render.deck_remaining = deck_remaining;
            a.render.selected_hand_idx = None;
            a.render.selected_table_attack = None;
            set_text("sb-deck", &deck_remaining.to_string());
            set_text("sb-turn", if you_attack { "Your attack" } else { "Opponent's attack" });
            drop(a);
            show_durak_buttons(app.borrow().render.is_attacker);
        }

        ServerMessage::OpponentHandCount { count } => {
            app.borrow_mut().render.opponent_card_count = count;
        }

        // Blackjack
        ServerMessage::DealerCard { card, hidden } => {
            app.borrow_mut().render.dealer_hand.push((card, !hidden));
        }

        ServerMessage::PlayerCard { card } => {
            let mut a = app.borrow_mut();
            a.render.your_hand.push(card);
            a.render.player_score = shared::blackjack::hand_value(&a.render.your_hand);
            set_text("sb-turn", &format!("Your score: {}", a.render.player_score));
        }

        ServerMessage::DealerRevealed { card } => {
            // Flip the hidden dealer card
            let mut a = app.borrow_mut();
            if let Some((c, face_up)) = a.render.dealer_hand.first_mut() {
                *c = card;
                *face_up = true;
            }
        }

        ServerMessage::HandResult { outcome, your_score, dealer_score } => {
            let mut a = app.borrow_mut();
            a.render.player_score = your_score;
            a.render.dealer_score = dealer_score;
        }

        ServerMessage::GameOver { winner, reason } => {
            let player_id = app.borrow().player_id;
            let won = winner == player_id;
            let push = winner.is_none() && reason.to_lowercase().contains("push");
            show_result(won, push, &reason, app.clone());
        }

        ServerMessage::PokerGameStarted { room_id, your_hole_cards, your_seat, players, dealer_seat, small_blind, big_blind } => {
            hide("picker-screen");
            hide("waiting-overlay");
            hide("invite-banner");
            show("game-screen");
            show("chat-panel");
            show("forfeit-btn");

            let my_id = app.borrow().player_id;
            let mut a = app.borrow_mut();
            a.current_room = Some(room_id);
            a.render = GameRender::new();
            a.render.game_type = GameType::TexasPoker;
            a.render.poker_hole_cards = your_hole_cards.to_vec();
            a.render.poker_your_seat = your_seat;
            a.render.poker_dealer_seat = dealer_seat;

            a.render.poker_seats = players.iter().enumerate().map(|(i, p)| {
                PokerSeatRender {
                    id: p.id,
                    name: p.name.clone(),
                    chips: p.chips,
                    bet: 0,
                    folded: false,
                    active: true,
                    all_in: false,
                    is_action: false,
                    is_dealer: i == dealer_seat,
                    is_you: Some(p.id) == my_id,
                }
            }).collect();

            set_text("sb-opponent", &format!("{} players", players.len()));
            set_text("sb-turn", "Waiting for action...");
            hide_action_bar_buttons();
            hide("action-bar");

            canvas::resize_canvas(&canvas_el());
            append_chat_system(&format!("Texas Hold'em started! Blinds {}/{}", small_blind, big_blind));
        }

        ServerMessage::PokerStateUpdate { room_id: _, community_cards, pot, current_bet, your_chips, your_bet, action_player_id, dealer_seat, players_info } => {
            let my_id = app.borrow().player_id;
            let mut a = app.borrow_mut();
            a.render.poker_community = community_cards;
            a.render.poker_pot = pot;
            a.render.poker_current_bet = current_bet;
            a.render.poker_your_chips = your_chips;
            a.render.poker_your_bet = your_bet;
            a.render.poker_action_id = Some(action_player_id);
            a.render.poker_dealer_seat = dealer_seat;

            // Update seat states: collect seat IDs first to avoid borrow conflict
            let seat_ids: Vec<uuid::Uuid> = a.render.poker_seats.iter().map(|s| s.id).collect();
            for (seat_idx, seat) in a.render.poker_seats.iter_mut().enumerate() {
                if let Some(info) = players_info.iter().find(|p| p.id == seat.id) {
                    seat.chips = info.chips;
                    seat.bet = info.bet;
                    seat.folded = info.folded;
                    seat.active = info.active;
                    seat.all_in = info.all_in;
                }
                seat.is_action = seat.id == action_player_id;
                seat.is_dealer = seat_idx == dealer_seat;
            }

            let is_my_turn = Some(action_player_id) == my_id;
            a.render.your_turn = is_my_turn;
            a.poker_raise_amount = (current_bet * 2).max(current_bet + 20);

            drop(a);
            show_poker_buttons(&app, is_my_turn);
            set_text("sb-turn", if is_my_turn { "Your turn" } else { "Waiting..." });
        }

        ServerMessage::PokerShowdown { hands, winner_id, pot_won } => {
            let player_id = app.borrow().player_id;
            let won = Some(winner_id) == player_id;
            let reason = format!("Pot of {} chips awarded", pot_won);
            append_chat_system(&format!("Showdown! {} chips to winner", pot_won));
            for (pid, cards, rank) in &hands {
                let is_me = Some(*pid) == player_id;
                let prefix = if is_me { "You" } else { "Opponent" };
                append_chat_system(&format!("{}: {:?}", prefix, rank));
            }
            show_result(won, false, &reason, app.clone());
        }

        ServerMessage::PokerPlayerFolded { player_id: folded_id } => {
            let my_id = app.borrow().player_id;
            let mut a = app.borrow_mut();
            for seat in &mut a.render.poker_seats {
                if seat.id == folded_id {
                    seat.folded = true;
                }
            }
            let name = a.render.poker_seats.iter()
                .find(|s| s.id == folded_id)
                .map(|s| s.name.clone())
                .unwrap_or_default();
            drop(a);
            append_chat_system(&format!("{} folded", name));
        }

        ServerMessage::VoiceSignalRelayed { from, signal_type, payload } => {
            // Relay to WebRTC JS handler via postMessage
            let js_obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&js_obj, &"type".into(), &"voiceSignal".into());
            let _ = js_sys::Reflect::set(&js_obj, &"from".into(), &from.to_string().into());
            let _ = js_sys::Reflect::set(&js_obj, &"signalType".into(), &signal_type.into());
            let _ = js_sys::Reflect::set(&js_obj, &"payload".into(), &payload.into());
            if let Some(w) = web_sys::window() {
                let _ = w.post_message(&js_obj, "*");
            }
        }

        ServerMessage::ChatReceived { from, text } => {
            append_chat_msg(&from, &text);
        }

        ServerMessage::Error { msg } => {
            // If we're on the username modal, show error there
            let username_modal_visible = !el("username-modal").class_list().contains("hidden");
            if username_modal_visible {
                set_text("username-error", &msg);
            } else {
                append_chat_system(&format!("Error: {}", msg));
                hide("waiting-overlay");
            }
        }
    }
}

// ── UI helpers ────────────────────────────────────────────────────────────────

fn update_player_list(app: &AppHandle, players: Vec<LobbyPlayer>) {
    let list = el("player-list");
    let my_id = app.borrow().player_id;

    // Clear existing (keep bot row if present)
    list.set_inner_html("");

    // Always show bot row first
    let bot_row = create_bot_row(app);
    list.append_child(&bot_row).unwrap();

    let others: Vec<_> = players.iter()
        .filter(|p| Some(p.id) != my_id)
        .collect();

    if others.is_empty() {
        let empty = doc().create_element("div").unwrap();
        empty.set_attribute("id", "picker-empty").unwrap();
        empty.set_inner_html("No other players online");
        empty.set_attribute("style", "padding:12px;text-align:center;font-size:11px;color:#555").unwrap();
        list.append_child(&empty).unwrap();
    } else {
        for player in others {
            let row = create_player_row(app, player);
            list.append_child(&row).unwrap();
        }
    }

    // Update online count in game screen
    let count = players.iter().filter(|p| p.available).count();
    set_text("players-count", &count.to_string());
    update_online_list(&players, my_id);
}

fn create_bot_row(app: &AppHandle) -> Element {
    let doc = doc();
    let row = doc.create_element("div").unwrap();
    row.set_attribute("class", "player-row bot-row").unwrap();

    let name_span = doc.create_element("span").unwrap();
    name_span.set_inner_html("[BOT] Computer");
    row.append_child(&name_span).unwrap();

    let btn = doc.create_element("button").unwrap();
    btn.set_attribute("class", "invite-btn").unwrap();
    btn.set_inner_html("Play");

    {
        let app = app.clone();
        let cb = Closure::<dyn FnMut()>::new(move || {
            let (ws, game) = {
                let a = app.borrow();
                (a.ws.clone(), a.selected_game.clone())
            };
            if let Some(ws) = ws {
                net::send(&ws, &ClientMessage::PlayBot { game });
            }
        });
        btn.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();
    }
    row.append_child(&btn).unwrap();
    row
}

fn create_player_row(app: &AppHandle, player: &LobbyPlayer) -> Element {
    let doc = doc();
    let row = doc.create_element("div").unwrap();
    row.set_attribute("class", "player-row").unwrap();

    let name_span = doc.create_element("span").unwrap();
    name_span.set_inner_html(&player.username);
    row.append_child(&name_span).unwrap();

    let right = doc.create_element("div").unwrap();
    right.set_attribute("style", "display:flex;gap:8px;align-items:center").unwrap();

    let badge = doc.create_element("span").unwrap();
    if player.available {
        badge.set_attribute("class", "player-badge").unwrap();
        badge.set_inner_html("available");
    } else {
        badge.set_attribute("class", "player-badge busy").unwrap();
        badge.set_inner_html("in game");
    }
    right.append_child(&badge).unwrap();

    if player.available {
        let btn = doc.create_element("button").unwrap();
        btn.set_attribute("class", "invite-btn").unwrap();
        btn.set_inner_html("Invite");

        let target_id = player.id;
        let app = app.clone();
        let cb = Closure::<dyn FnMut()>::new(move || {
            let (ws, game) = {
                let a = app.borrow();
                (a.ws.clone(), a.selected_game.clone())
            };
            if let Some(ws) = ws {
                net::send(&ws, &ClientMessage::InvitePlayer { target_id, game });
                show("waiting-overlay");
                set_text("waiting-text", "Waiting for response...");
            }
        });
        btn.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();
        right.append_child(&btn).unwrap();
    }

    row.append_child(&right).unwrap();
    row
}

fn update_online_list(players: &[LobbyPlayer], my_id: Option<uuid::Uuid>) {
    let list = el("players-online-list");
    list.set_inner_html("");
    for player in players {
        if Some(player.id) == my_id {
            continue;
        }
        let div = doc().create_element("div").unwrap();
        div.set_attribute("class", "player-row").unwrap();
        div.set_inner_html(&format!(
            "<span>{}</span><span class='player-badge{}'>{}</span>",
            player.username,
            if player.available { "" } else { " busy" },
            if player.available { "online" } else { "in game" }
        ));
        list.append_child(&div).unwrap();
    }
}

fn show_durak_buttons(is_attacker: bool) {
    if is_attacker {
        show("btn-attack");
        show("btn-end-attack");
        hide("btn-defend");
        hide("btn-take");
    } else {
        show("btn-defend");
        show("btn-take");
        hide("btn-attack");
        hide("btn-end-attack");
    }
    hide("btn-hit");
    hide("btn-stand");
}

fn show_blackjack_buttons(player_turn: bool) {
    hide("btn-attack");
    hide("btn-defend");
    hide("btn-take");
    hide("btn-end-attack");
    if player_turn {
        show("btn-hit");
        show("btn-stand");
    } else {
        hide("btn-hit");
        hide("btn-stand");
    }
}

fn update_action_buttons(a: &App) {
    // Enable/disable based on selection state
    let has_card = a.render.selected_hand_idx.is_some();
    let has_atk = a.render.selected_table_attack.is_some();

    set_disabled("btn-attack", !has_card);
    set_disabled("btn-defend", !(has_card && has_atk));
}

fn show_result(won: bool, push: bool, reason: &str, app: AppHandle) {
    hide("game-screen");
    show("result-overlay");

    let title = el("result-title");
    if push {
        title.set_inner_html("PUSH");
        title.set_attribute("class", "push").unwrap();
    } else if won {
        title.set_inner_html("YOU WIN");
        title.set_attribute("class", "win").unwrap();
    } else {
        title.set_inner_html("YOU LOSE");
        title.set_attribute("class", "lose").unwrap();
    }
    set_text("result-reason", reason);

    // Score display
    let a = app.borrow();
    if a.render.player_score > 0 {
        set_text("result-score", &format!("You: {} / Dealer: {}", a.render.player_score, a.render.dealer_score));
    } else {
        set_text("result-score", "");
    }
    drop(a);

    // Countdown: show buttons after 3s, auto-return after 30s
    let countdown_el = el("result-countdown");
    countdown_el.set_inner_html("Returning to lobby in 30s...");

    // Hide action buttons initially, show after 3s
    hide_el(&el("result-play-again"));
    hide_el(&el("result-lobby"));

    let app_3s = app.clone();
    let reveal_cb = Closure::once(move || {
        show("result-play-again");
        show("result-lobby");
    });
    window().set_timeout_with_callback_and_timeout_and_arguments_0(
        reveal_cb.as_ref().unchecked_ref(), 3000
    ).unwrap();
    reveal_cb.forget();

    let auto_cb = Closure::once(move || {
        hide("result-overlay");
        show("picker-screen");
        app_3s.borrow_mut().current_room = None;
    });
    window().set_timeout_with_callback_and_timeout_and_arguments_0(
        auto_cb.as_ref().unchecked_ref(), 30000
    ).unwrap();
    auto_cb.forget();
}

fn setup_poker_buttons(app: AppHandle) -> Result<(), JsValue> {
    // Fold (F)
    btn_click("btn-fold", app.clone(), |app| {
        send_poker_action(app, GameAction::PokerFold);
    })?;
    // Check (C)
    btn_click("btn-check", app.clone(), |app| {
        send_poker_action(app, GameAction::PokerCheck);
    })?;
    // Call (C — same button, toggled in DOM; we use separate IDs)
    btn_click("btn-call", app.clone(), |app| {
        send_poker_action(app, GameAction::PokerCall);
    })?;
    // Raise (R)
    btn_click("btn-raise", app.clone(), |app| {
        let amount = app.borrow().poker_raise_amount;
        send_poker_action(app, GameAction::PokerRaise { amount });
    })?;
    Ok(())
}

fn send_poker_action(app: AppHandle, action: GameAction) {
    let (ws, room_id) = {
        let a = app.borrow();
        (a.ws.clone(), a.current_room)
    };
    if let (Some(ws), Some(room_id)) = (ws, room_id) {
        net::send(&ws, &ClientMessage::GameMove { room_id, action });
    }
}

fn setup_forfeit_button(app: AppHandle) -> Result<(), JsValue> {
    btn_click("forfeit-btn", app.clone(), |app| {
        let (ws, room_id) = {
            let a = app.borrow();
            (a.ws.clone(), a.current_room)
        };
        if let (Some(ws), Some(room_id)) = (ws, room_id) {
            net::send(&ws, &ClientMessage::ForfeitGame { room_id });
        }
    })?;
    Ok(())
}

fn setup_keyboard_shortcuts(app: AppHandle) -> Result<(), JsValue> {
    let cb = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
        // Don't fire shortcuts when typing in input
        if let Some(target) = e.target() {
            if let Ok(elem) = target.dyn_into::<web_sys::HtmlElement>() {
                let tag = elem.tag_name().to_lowercase();
                if tag == "input" || tag == "textarea" {
                    return;
                }
            }
        }

        let game_type = app.borrow().render.game_type.clone();
        match (e.key().as_str(), &game_type) {
            ("f" | "F", GameType::TexasPoker) => {
                send_poker_action(app.clone(), GameAction::PokerFold);
            }
            ("c" | "C", GameType::TexasPoker) => {
                let current_bet = app.borrow().render.poker_current_bet;
                let your_bet = app.borrow().render.poker_your_bet;
                if current_bet == your_bet {
                    send_poker_action(app.clone(), GameAction::PokerCheck);
                } else {
                    send_poker_action(app.clone(), GameAction::PokerCall);
                }
            }
            ("r" | "R", GameType::TexasPoker) => {
                let amount = app.borrow().poker_raise_amount;
                send_poker_action(app.clone(), GameAction::PokerRaise { amount });
            }
            _ => {}
        }
    });
    window().add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())?;
    cb.forget();
    Ok(())
}

fn show_poker_buttons(app: &AppHandle, is_my_turn: bool) {
    hide("btn-attack");
    hide("btn-defend");
    hide("btn-take");
    hide("btn-end-attack");
    hide("btn-hit");
    hide("btn-stand");

    if is_my_turn {
        show("action-bar");
        let current_bet = app.borrow().render.poker_current_bet;
        let your_bet = app.borrow().render.poker_your_bet;
        show("btn-fold");
        if current_bet == your_bet {
            show("btn-check");
            hide("btn-call");
        } else {
            hide("btn-check");
            show("btn-call");
        }
        show("btn-raise");
    } else {
        hide("btn-fold");
        hide("btn-check");
        hide("btn-call");
        hide("btn-raise");
    }
}

fn hide_action_bar_buttons() {
    for id in &["btn-attack", "btn-defend", "btn-take", "btn-end-attack",
                "btn-hit", "btn-stand", "btn-fold", "btn-check", "btn-call", "btn-raise"] {
        if let Some(e) = doc().get_element_by_id(id) {
            e.class_list().add_1("hidden").unwrap();
        }
    }
}

fn append_chat_msg(from: &str, text: &str) {
    let doc = doc();
    let msgs = el("chat-messages");
    let div = doc.create_element("div").unwrap();
    div.set_attribute("class", "chat-msg").unwrap();
    div.set_inner_html(&format!(
        "<span class='chat-from'>{}: </span>{}",
        html_escape(from),
        html_escape(text)
    ));
    msgs.append_child(&div).unwrap();
    // Auto-scroll
    msgs.dyn_ref::<HtmlElement>().unwrap().scroll_to_with_x_and_y(0.0, 999999.0);
}

fn append_chat_system(text: &str) {
    let doc = doc();
    let msgs = el("chat-messages");
    let div = doc.create_element("div").unwrap();
    div.set_attribute("class", "chat-msg system").unwrap();
    div.set_inner_html(&html_escape(text));
    msgs.append_child(&div).unwrap();
    msgs.dyn_ref::<HtmlElement>().unwrap().scroll_to_with_x_and_y(0.0, 999999.0);
}

fn show_picker_error(msg: &str) {
    // Briefly flash a message in the picker — reuse a simple approach
    append_chat_system(msg);
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}

// ── DOM helpers ──────────────────────────────────────────────────────────────

fn window() -> Window {
    web_sys::window().unwrap()
}

fn doc() -> Document {
    window().document().unwrap()
}

fn el(id: &str) -> Element {
    doc().get_element_by_id(id).unwrap_or_else(|| panic!("element #{} not found", id))
}

fn el_by_id(id: &str) -> Element {
    el(id)
}

fn input_el(id: &str) -> HtmlInputElement {
    el(id).dyn_into::<HtmlInputElement>().unwrap()
}

fn canvas_el() -> HtmlCanvasElement {
    el("canvas").dyn_into::<HtmlCanvasElement>().unwrap()
}

fn show(id: &str) {
    let e = el(id);
    e.class_list().remove_1("hidden").unwrap();
}

fn hide(id: &str) {
    let e = el(id);
    e.class_list().add_1("hidden").unwrap();
}

fn hide_el(e: &Element) {
    e.class_list().add_1("hidden").unwrap();
}

fn set_text(id: &str, text: &str) {
    el(id).set_inner_html(text);
}

fn set_disabled(id: &str, disabled: bool) {
    let e = el(id);
    if disabled {
        e.set_attribute("disabled", "").unwrap();
    } else {
        e.remove_attribute("disabled").unwrap();
    }
}
