use shared::deck::{Card, Rank, Suit};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

// ── Design constants ──────────────────────────────────────────────────────────
pub const CARD_W: f64 = 80.0;
pub const CARD_H: f64 = 120.0;
pub const CARD_R: f64 = 6.0; // corner radius

const COLOR_TABLE:    &str = "#0a1628";
const COLOR_TABLE_FELT: &str = "#0d2040";
const COLOR_CARD_FACE: &str = "#f8f0e3";
const COLOR_CARD_BACK: &str = "#0f3460";
const COLOR_BACK_PATTERN: &str = "#e94560";
const COLOR_RED:      &str = "#c0392b";
const COLOR_BLACK:    &str = "#1a1a2e";
const COLOR_SELECTED: &str = "#e94560";
const COLOR_HOVER:    &str = "#2a4a7e";
const COLOR_ATTACK_SLOT: &str = "#1a2e4a";

// ── Render state ──────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct CardAnim {
    pub card: Card,
    pub from_x: f64,
    pub from_y: f64,
    pub to_x: f64,
    pub to_y: f64,
    pub start_ts: f64,
    pub duration: f64,
    pub face_up: bool,
}

#[derive(Clone, Default)]
pub struct PokerSeatRender {
    pub id: uuid::Uuid,
    pub name: String,
    pub chips: u32,
    pub bet: u32,
    pub folded: bool,
    pub active: bool,
    pub all_in: bool,
    pub is_action: bool,  // it's this player's turn
    pub is_dealer: bool,
    pub is_you: bool,
}

#[derive(Clone, Default)]
pub struct GameRender {
    pub your_hand: Vec<Card>,
    pub table: Vec<(Card, Option<Card>)>,  // (attack, defense)
    pub trump_card: Option<Card>,
    pub opponent_card_count: u8,
    pub deck_remaining: u8,
    pub selected_hand_idx: Option<usize>,
    pub hover_hand_idx: Option<usize>,
    pub selected_table_attack: Option<Card>,  // for defending: which attack card to counter
    pub your_turn: bool,
    pub is_attacker: bool,
    pub game_type: shared::messages::GameType,
    // Blackjack
    pub dealer_hand: Vec<(Card, bool)>,   // (card, face_up)
    pub player_score: u8,
    pub dealer_score: u8,
    // Poker
    pub poker_hole_cards: Vec<Card>,       // your 2 hole cards
    pub poker_community: Vec<Card>,        // 0-5 community cards
    pub poker_pot: u32,
    pub poker_current_bet: u32,
    pub poker_your_chips: u32,
    pub poker_your_bet: u32,
    pub poker_seats: Vec<PokerSeatRender>, // all seats including yours
    pub poker_your_seat: usize,
    pub poker_dealer_seat: usize,
    pub poker_action_id: Option<uuid::Uuid>,
}

impl GameRender {
    pub fn new() -> Self {
        GameRender {
            game_type: shared::messages::GameType::Durak,
            ..Default::default()
        }
    }
}

pub fn get_context(canvas: &HtmlCanvasElement) -> CanvasRenderingContext2d {
    canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap()
}

pub fn resize_canvas(canvas: &HtmlCanvasElement) {
    let window = web_sys::window().unwrap();
    let w = window.inner_width().unwrap().as_f64().unwrap() as u32;
    let h = window.inner_height().unwrap().as_f64().unwrap() as u32;
    // Status bar ≈ 36, action bar ≈ 44, players panel ≈ 36
    let h = h.saturating_sub(36 + 44 + 36);
    canvas.set_width(w);
    canvas.set_height(h);
}

// ── Full repaint ──────────────────────────────────────────────────────────────

pub fn render(canvas: &HtmlCanvasElement, state: &GameRender, anims: &[CardAnim], ts: f64) {
    let ctx = get_context(canvas);
    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    // Background
    ctx.set_fill_style_str(COLOR_TABLE);
    ctx.fill_rect(0.0, 0.0, w, h);

    // Felt oval in center — kept tight so cards fill the visible area
    ctx.set_fill_style_str(COLOR_TABLE_FELT);
    ctx.begin_path();
    let _ = ctx.ellipse(w / 2.0, h / 2.0, w * 0.28, h * 0.26, 0.0, 0.0, std::f64::consts::TAU);
    ctx.fill();

    match state.game_type {
        shared::messages::GameType::Durak => render_durak(&ctx, w, h, state, ts),
        shared::messages::GameType::Blackjack => render_blackjack(&ctx, w, h, state, ts),
        shared::messages::GameType::TexasPoker => render_poker(&ctx, w, h, state),
    }

    // Draw animations on top
    for anim in anims {
        let progress = ((ts - anim.start_ts) / anim.duration).min(1.0);
        let ease = ease_out_cubic(progress);
        let x = anim.from_x + (anim.to_x - anim.from_x) * ease;
        let y = anim.from_y + (anim.to_y - anim.from_y) * ease;
        draw_card(&ctx, x, y, anim.card, anim.face_up, false, false);
    }
}

fn render_durak(ctx: &CanvasRenderingContext2d, w: f64, h: f64, state: &GameRender, _ts: f64) {
    let center_x = w / 2.0;
    let center_y = h / 2.0;

    // ── Opponent cards (face down, top center) ──
    let opp_count = state.opponent_card_count as usize;
    let opp_y = 16.0;
    let opp_overlap = 28.0;
    let opp_start_x = center_x - (opp_count as f64 * opp_overlap) / 2.0;
    for i in 0..opp_count {
        let x = opp_start_x + i as f64 * opp_overlap;
        draw_card_back(ctx, x, opp_y);
    }

    // ── Deck (face down, left side) ──
    if state.deck_remaining > 0 {
        draw_card_back(ctx, 20.0, center_y - CARD_H / 2.0);
        // Deck count label
        ctx.set_fill_style_str("#888");
        ctx.set_font("11px monospace");
        let _ = ctx.fill_text(
            &state.deck_remaining.to_string(),
            20.0 + CARD_W / 2.0 - 8.0,
            center_y - CARD_H / 2.0 + CARD_H + 14.0,
        );
    }

    // ── Trump card (sideways beside deck) ──
    if let Some(trump) = state.trump_card {
        // Draw rotated trump card sticking out from deck
        ctx.save();
        ctx.translate(20.0 + CARD_W / 2.0, center_y + CARD_H / 2.0 - 20.0)
            .unwrap();
        ctx.rotate(std::f64::consts::FRAC_PI_2).unwrap();
        draw_card(ctx, -CARD_H / 2.0, -CARD_W / 2.0, trump, true, false, false);
        ctx.restore();
    }

    // ── Table (center) ──
    let table_len = state.table.len();
    if table_len > 0 {
        let slot_w = CARD_W + 16.0;
        let total_w = table_len as f64 * slot_w * 2.0 - slot_w;
        let table_x = center_x - total_w / 2.0;
        let atk_y = center_y - CARD_H / 2.0 - 10.0;
        let def_y = center_y - CARD_H / 2.0 + 10.0;

        for (i, (atk, def)) in state.table.iter().enumerate() {
            let x = table_x + i as f64 * slot_w * 2.0;

            // Slot background
            ctx.set_fill_style_str(COLOR_ATTACK_SLOT);
            rounded_rect(ctx, x - 4.0, atk_y - 4.0, CARD_W + 8.0, CARD_H + 8.0, 6.0);
            ctx.fill();

            let selected = state.selected_table_attack.map_or(false, |a| a == *atk);
            draw_card(ctx, x, atk_y, *atk, true, selected, false);

            if let Some(def_card) = def {
                draw_card(ctx, x + 6.0, def_y, *def_card, true, false, false);
            }
        }
    }

    // ── Your hand (bottom) ──
    render_hand(ctx, w, h, state);

    // ── Contextual hint ──
    let hint = if state.your_turn && state.is_attacker {
        "click a card to attack"
    } else if state.your_turn && !state.is_attacker {
        "click a card to defend"
    } else {
        ""
    };
    if !hint.is_empty() {
        ctx.set_fill_style_str("rgba(180,180,180,0.55)");
        ctx.set_font("11px monospace");
        let _ = ctx.fill_text(hint, center_x - 100.0, h - CARD_H - 24.0);
    }
}

fn render_blackjack(ctx: &CanvasRenderingContext2d, w: f64, h: f64, state: &GameRender, _ts: f64) {
    let center_x = w / 2.0;

    // ── Dealer hand (top) ──
    let dealer_count = state.dealer_hand.len();
    let dealer_start = center_x - dealer_count as f64 * (CARD_W + 8.0) / 2.0;
    for (i, (card, face_up)) in state.dealer_hand.iter().enumerate() {
        let x = dealer_start + i as f64 * (CARD_W + 8.0);
        let y = 30.0;
        if *face_up {
            draw_card(ctx, x, y, *card, true, false, false);
        } else {
            draw_card_back(ctx, x, y);
        }
    }

    // Dealer score label — sits just below the dealer cards
    if state.dealer_score > 0 {
        ctx.set_fill_style_str("#888");
        ctx.set_font("13px monospace");
        let _ = ctx.fill_text(&format!("Dealer: {}", state.dealer_score), center_x - 30.0, 30.0 + CARD_H + 16.0);
    }

    // ── Player hand (bottom) ──
    render_hand(ctx, w, h, state);

    // Player score
    if state.player_score > 0 {
        ctx.set_fill_style_str(if state.player_score > 21 { "#e94560" } else { "#4caf50" });
        ctx.set_font("13px monospace");
        let _ = ctx.fill_text(
            &format!("You: {}", state.player_score),
            center_x - 24.0,
            h - CARD_H - 30.0,
        );
    }
}

fn render_hand(ctx: &CanvasRenderingContext2d, w: f64, h: f64, state: &GameRender) {
    let hand = &state.your_hand;
    if hand.is_empty() {
        return;
    }
    let overlap = if hand.len() > 8 { 22.0 } else { 30.0 };
    let total_w = CARD_W + (hand.len() - 1) as f64 * overlap;
    let start_x = (w - total_w) / 2.0;
    let y = h - CARD_H - 12.0;

    for (i, card) in hand.iter().enumerate() {
        let x = start_x + i as f64 * overlap;
        let selected = state.selected_hand_idx == Some(i);
        let hovered = state.hover_hand_idx == Some(i);
        let card_y = if selected { y - 14.0 } else if hovered { y - 7.0 } else { y };
        draw_card(ctx, x, card_y, *card, true, selected, hovered);
    }
}

fn render_poker(ctx: &CanvasRenderingContext2d, w: f64, h: f64, state: &GameRender) {
    let cx = w / 2.0;
    let cy = h / 2.0;

    // ── Community cards (center table) ──
    let community = &state.poker_community;
    let n_comm = community.len();
    let comm_total = 5.0 * CARD_W + 4.0 * 8.0;
    let comm_x0 = cx - comm_total / 2.0;
    let comm_y = cy - CARD_H / 2.0;

    // Placeholder slots for 5 community cards
    for i in 0..5 {
        let x = comm_x0 + i as f64 * (CARD_W + 8.0);
        ctx.set_fill_style_str(COLOR_ATTACK_SLOT);
        rounded_rect(ctx, x, comm_y, CARD_W, CARD_H, CARD_R);
        ctx.fill();
    }
    for (i, card) in community.iter().enumerate() {
        let x = comm_x0 + i as f64 * (CARD_W + 8.0);
        draw_card(ctx, x, comm_y, *card, true, false, false);
    }

    // Pot label
    ctx.set_fill_style_str("#e0c080");
    ctx.set_font("bold 13px monospace");
    let _ = ctx.fill_text(&format!("POT: {}", state.poker_pot), cx - 36.0, comm_y - 12.0);

    // ── Player seats arranged around the table ──
    // Seat positions relative to center for 2, 3, 4 players
    // Seat 0 = you (bottom center), others clockwise
    let n_seats = state.poker_seats.len().max(1);
    let seat_positions = seat_positions_for(n_seats, w, h);

    for (i, seat) in state.poker_seats.iter().enumerate() {
        let (sx, sy) = seat_positions[i];

        // Seat box background
        let bg = if seat.is_action {
            "#1a3a1a"
        } else if seat.folded || !seat.active {
            "#1a1a1a"
        } else {
            "#0d1e2e"
        };
        ctx.set_fill_style_str(bg);
        rounded_rect(ctx, sx - 44.0, sy - 18.0, 88.0, 36.0, 6.0);
        ctx.fill();

        // Action highlight border
        if seat.is_action {
            ctx.set_stroke_style_str("#4caf50");
            ctx.set_line_width(2.0);
            rounded_rect(ctx, sx - 44.0, sy - 18.0, 88.0, 36.0, 6.0);
            ctx.stroke();
        }

        // Name
        let name_color = if seat.folded || !seat.active { "#555" } else { "#ccc" };
        ctx.set_fill_style_str(name_color);
        ctx.set_font("bold 11px monospace");
        let display_name = if seat.name.len() > 8 {
            format!("{}…", &seat.name[..7])
        } else {
            seat.name.clone()
        };
        let _ = ctx.fill_text(&display_name, sx - 40.0, sy - 4.0);

        // Chips
        ctx.set_fill_style_str("#e0c080");
        ctx.set_font("10px monospace");
        let _ = ctx.fill_text(&format!("{}c", seat.chips), sx - 40.0, sy + 10.0);

        // Bet
        if seat.bet > 0 {
            ctx.set_fill_style_str("#e94560");
            let _ = ctx.fill_text(&format!("bet:{}", seat.bet), sx + 2.0, sy + 10.0);
        }

        // Dealer ◆ button
        if seat.is_dealer {
            ctx.set_fill_style_str("#e0c080");
            ctx.set_font("bold 12px monospace");
            let _ = ctx.fill_text("◆", sx + 30.0, sy - 6.0);
        }

        // ALL-IN label
        if seat.all_in {
            ctx.set_fill_style_str("#e94560");
            ctx.set_font("bold 9px monospace");
            let _ = ctx.fill_text("ALL-IN", sx - 20.0, sy - 22.0);
        }

        // Folded dim overlay
        if seat.folded || !seat.active {
            ctx.set_fill_style_str("rgba(0,0,0,0.4)");
            rounded_rect(ctx, sx - 44.0, sy - 18.0, 88.0, 36.0, 6.0);
            ctx.fill();
        }

        // Your hole cards (bottom seat = you)
        if seat.is_you && !state.poker_hole_cards.is_empty() {
            let hx = sx - CARD_W - 4.0;
            let hy = sy - CARD_H / 2.0;
            for (j, card) in state.poker_hole_cards.iter().enumerate() {
                draw_card(ctx, hx + j as f64 * (CARD_W + 4.0), hy, *card, true, false, false);
            }
        }
    }

    // ── Status line (your turn indicator) ──
    if state.your_turn {
        ctx.set_fill_style_str("#4caf50");
        ctx.set_font("bold 12px monospace");
        let _ = ctx.fill_text("YOUR TURN", cx - 38.0, h - 8.0);
    }
}

/// Returns (x, y) canvas positions for each seat index.
/// Seat 0 is always bottom-center (you). Others go clockwise.
fn seat_positions_for(n: usize, w: f64, h: f64) -> Vec<(f64, f64)> {
    let cx = w / 2.0;
    let cy = h / 2.0;
    let rx = w * 0.38;
    let ry = h * 0.32;
    let mut positions = Vec::with_capacity(n);
    for i in 0..n {
        // Start from bottom (PI/2 = down), go clockwise
        // seat 0 = bottom, then right, top, left for 4 seats
        let angle = std::f64::consts::FRAC_PI_2 + (i as f64 * std::f64::consts::TAU / n as f64);
        let x = cx + rx * angle.cos();
        let y = cy + ry * angle.sin();
        positions.push((x, y));
    }
    positions
}

// ── Card drawing primitives ──────────────────────────────────────────────────

pub fn draw_card(
    ctx: &CanvasRenderingContext2d,
    x: f64,
    y: f64,
    card: Card,
    face_up: bool,
    selected: bool,
    hovered: bool,
) {
    if !face_up {
        draw_card_back(ctx, x, y);
        return;
    }

    // Shadow
    ctx.set_shadow_color("rgba(0,0,0,0.5)");
    ctx.set_shadow_blur(6.0);
    ctx.set_shadow_offset_x(2.0);
    ctx.set_shadow_offset_y(2.0);

    // Card face
    ctx.set_fill_style_str(COLOR_CARD_FACE);
    rounded_rect(ctx, x, y, CARD_W, CARD_H, CARD_R);
    ctx.fill();

    // Selection highlight
    if selected {
        ctx.set_stroke_style_str(COLOR_SELECTED);
        ctx.set_line_width(2.5);
        rounded_rect(ctx, x, y, CARD_W, CARD_H, CARD_R);
        ctx.stroke();
    } else if hovered {
        ctx.set_stroke_style_str(COLOR_HOVER);
        ctx.set_line_width(1.5);
        rounded_rect(ctx, x, y, CARD_W, CARD_H, CARD_R);
        ctx.stroke();
    }

    ctx.set_shadow_blur(0.0);
    ctx.set_shadow_offset_x(0.0);
    ctx.set_shadow_offset_y(0.0);

    let color = if card.suit.is_red() { COLOR_RED } else { COLOR_BLACK };
    ctx.set_fill_style_str(color);

    let rank = card.rank.display();
    let suit = card.suit.symbol().to_string();

    // Top-left rank+suit
    ctx.set_font("bold 14px monospace");
    let _ = ctx.fill_text(rank, x + 5.0, y + 18.0);
    ctx.set_font("12px monospace");
    let _ = ctx.fill_text(&suit, x + 5.0, y + 32.0);

    // Center suit
    ctx.set_font("36px monospace");
    let _ = ctx.fill_text(&suit, x + CARD_W / 2.0 - 12.0, y + CARD_H / 2.0 + 12.0);

    // Bottom-right rank+suit (rotated 180°)
    ctx.save();
    ctx.translate(x + CARD_W - 5.0, y + CARD_H - 5.0).unwrap();
    ctx.rotate(std::f64::consts::PI).unwrap();
    ctx.set_font("bold 14px monospace");
    ctx.set_fill_style_str(color);
    let _ = ctx.fill_text(rank, 0.0, 16.0);
    ctx.set_font("12px monospace");
    let _ = ctx.fill_text(&suit, 0.0, 30.0);
    ctx.restore();
}

fn draw_card_back(ctx: &CanvasRenderingContext2d, x: f64, y: f64) {
    ctx.set_shadow_color("rgba(0,0,0,0.4)");
    ctx.set_shadow_blur(4.0);
    ctx.set_shadow_offset_x(1.0);
    ctx.set_shadow_offset_y(1.0);

    ctx.set_fill_style_str(COLOR_CARD_BACK);
    rounded_rect(ctx, x, y, CARD_W, CARD_H, CARD_R);
    ctx.fill();

    ctx.set_shadow_blur(0.0);
    ctx.set_shadow_offset_x(0.0);
    ctx.set_shadow_offset_y(0.0);

    // Diamond pattern
    ctx.set_stroke_style_str(COLOR_BACK_PATTERN);
    ctx.set_line_width(0.5);
    let pad = 8.0;
    for i in 0..4 {
        let off = i as f64 * 10.0;
        rounded_rect(ctx, x + pad + off, y + pad + off, CARD_W - 2.0 * (pad + off), CARD_H - 2.0 * (pad + off), 2.0);
        ctx.stroke();
    }
}

fn rounded_rect(ctx: &CanvasRenderingContext2d, x: f64, y: f64, w: f64, h: f64, r: f64) {
    ctx.begin_path();
    ctx.move_to(x + r, y);
    ctx.line_to(x + w - r, y);
    let _ = ctx.quadratic_curve_to(x + w, y, x + w, y + r);
    ctx.line_to(x + w, y + h - r);
    let _ = ctx.quadratic_curve_to(x + w, y + h, x + w - r, y + h);
    ctx.line_to(x + r, y + h);
    let _ = ctx.quadratic_curve_to(x, y + h, x, y + h - r);
    ctx.line_to(x, y + r);
    let _ = ctx.quadratic_curve_to(x, y, x + r, y);
    ctx.close_path();
}

// ── Hit testing ──────────────────────────────────────────────────────────────

/// Returns the hand index under (mx, my), or None.
pub fn hit_test_hand(state: &GameRender, canvas_w: f64, canvas_h: f64, mx: f64, my: f64) -> Option<usize> {
    let hand = &state.your_hand;
    if hand.is_empty() {
        return None;
    }
    let overlap = if hand.len() > 8 { 22.0 } else { 30.0 };
    let total_w = CARD_W + (hand.len() - 1) as f64 * overlap;
    let start_x = (canvas_w - total_w) / 2.0;
    let y = canvas_h - CARD_H - 12.0;

    // Hit test in reverse so top cards take priority
    for i in (0..hand.len()).rev() {
        let cx = start_x + i as f64 * overlap;
        let card_y = if state.selected_hand_idx == Some(i) { y - 14.0 } else { y };
        if mx >= cx && mx <= cx + CARD_W && my >= card_y && my <= card_y + CARD_H {
            return Some(i);
        }
    }
    None
}

/// Returns the attack card under (mx, my) on the table, or None.
pub fn hit_test_table(state: &GameRender, canvas_w: f64, canvas_h: f64, mx: f64, my: f64) -> Option<Card> {
    let table_len = state.table.len();
    if table_len == 0 {
        return None;
    }
    let slot_w = CARD_W + 16.0;  // must match render_durak
    let total_w = table_len as f64 * slot_w * 2.0 - slot_w;
    let table_x = canvas_w / 2.0 - total_w / 2.0;
    let atk_y = canvas_h / 2.0 - CARD_H / 2.0 - 10.0;

    for (i, (atk, def)) in state.table.iter().enumerate() {
        if def.is_some() {
            continue; // already defended
        }
        let x = table_x + i as f64 * slot_w * 2.0;
        if mx >= x && mx <= x + CARD_W && my >= atk_y && my <= atk_y + CARD_H {
            return Some(*atk);
        }
    }
    None
}

fn ease_out_cubic(t: f64) -> f64 {
    1.0 - (1.0 - t).powi(3)
}
