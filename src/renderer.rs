use wasm_bindgen::JsValue;
use crate::game::{GameState, GamePhase};
use crate::entities::{PlayerClass, ItemKind};
use crate::items::Item;
use crate::world_map::{Location, desert_rect, city_rect, MERCHANT_X, MERCHANT_Y};

pub const CANVAS_W: f32 = 800.0;
pub const CANVAS_H: f32 = 500.0;
pub const HUD_Y: f32 = 400.0;
pub const HUD_H: f32 = 100.0;

// HoMM-inspired palette
const C_BG: &str = "#0d0d1a";
const C_WALL: &str = "#1a1a2e";
const C_WALL_EDGE: &str = "#2a2a4e";
const C_FLOOR: &str = "#1e1e1e";
const C_FLOOR_GRID: &str = "#252525";
const C_FLOOR_ALT: &str = "#0e1422";       // Room 1 floor (dark blue stone)
const C_FLOOR_GRID_ALT: &str = "#141a2a";  // Room 1 grid
const C_PARTICLE: &str = "#cc2222";
const C_HUD_BG: &str = "#0a0a14";
const C_HUD_BORDER: &str = "#5a3a15";
const C_HP_BAR: &str = "#cc2222";
const C_HP_BAR_BG: &str = "#330000";
const C_HP_ORB: &str = "#cc2222";
const C_MANA_ORB: &str = "#2244cc";
const C_OVERLAY_BG: &str = "rgba(0,0,0,0.75)";
const C_OVERLAY_TEXT: &str = "#aaffaa";
const C_GOLD: &str = "#d4af37";
const C_DAMAGE: &str = "#ffee44";
const C_AGGRO: &str = "#ff4444";
const C_POTION_HP: &str = "#cc2244";
const C_POTION_MP: &str = "#2244cc";

fn set_fill(ctx: &web_sys::CanvasRenderingContext2d, color: &str) {
    ctx.set_fill_style(&JsValue::from_str(color));
}

fn set_stroke(ctx: &web_sys::CanvasRenderingContext2d, color: &str) {
    ctx.set_stroke_style(&JsValue::from_str(color));
}

fn fill_rect(ctx: &web_sys::CanvasRenderingContext2d, x: f64, y: f64, w: f64, h: f64) {
    ctx.fill_rect(x, y, w, h);
}

fn stroke_rect(ctx: &web_sys::CanvasRenderingContext2d, x: f64, y: f64, w: f64, h: f64) {
    ctx.stroke_rect(x, y, w, h);
}

pub fn draw(gs: &GameState) {
    match gs.phase {
        GamePhase::Title => {
            crate::title::render(&gs.ctx);
        }
        GamePhase::CharacterSelect => {
            crate::character_select::render(&gs.ctx, gs.char_selected);
        }
        GamePhase::WorldMap => {
            draw_world_map(gs);
        }
        GamePhase::Playing => {
            draw_playing(gs);
        }
        GamePhase::Shopping => {
            draw_playing(gs);
            draw_shop_overlay(gs);
        }
        GamePhase::Dead => {
            draw_playing(gs);
            draw_game_over(&gs.ctx);
        }
    }
}

// ─── World Map ────────────────────────────────────────────────────────────────

fn draw_world_map(gs: &GameState) {
    let ctx = &gs.ctx;

    // Sky background
    set_fill(ctx, "#1a1a2e");
    fill_rect(ctx, 0.0, 0.0, CANVAS_W as f64, HUD_Y as f64);

    // Ground strip
    set_fill(ctx, "#2a2214");
    fill_rect(ctx, 0.0, 280.0, CANVAS_W as f64, 120.0);

    // Title
    set_fill(ctx, C_GOLD);
    ctx.set_font("bold 18px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("WORLD MAP — Click a location to travel", 180.0, 58.0);

    // ── Desert Zone ──────────────────────────────────────────────────────────
    let (dx, dy, dw, dh) = desert_rect();
    let (dx, dy, dw, dh) = (dx as f64, dy as f64, dw as f64, dh as f64);

    // Sandy background
    set_fill(ctx, "#3d2e10");
    fill_rect(ctx, dx, dy, dw, dh);

    // Desert dunes (layered arcs)
    set_fill(ctx, "#c8a040");
    ctx.begin_path();
    ctx.move_to(dx, dy + dh);
    ctx.quadratic_curve_to(dx + dw * 0.25, dy + dh * 0.55, dx + dw * 0.5, dy + dh * 0.7);
    ctx.quadratic_curve_to(dx + dw * 0.75, dy + dh * 0.85, dx + dw, dy + dh * 0.6);
    ctx.line_to(dx + dw, dy + dh);
    ctx.close_path();
    ctx.fill();

    // Second dune (lighter)
    set_fill(ctx, "#e8c060");
    ctx.begin_path();
    ctx.move_to(dx, dy + dh * 0.8);
    ctx.quadratic_curve_to(dx + dw * 0.3, dy + dh * 0.55, dx + dw * 0.55, dy + dh * 0.65);
    ctx.quadratic_curve_to(dx + dw * 0.8, dy + dh * 0.75, dx + dw, dy + dh * 0.5);
    ctx.line_to(dx + dw, dy + dh);
    ctx.line_to(dx, dy + dh);
    ctx.close_path();
    ctx.fill();

    // Pyramid (small)
    set_fill(ctx, "#b8902a");
    ctx.begin_path();
    ctx.move_to(dx + dw * 0.65, dy + dh * 0.48);
    ctx.line_to(dx + dw * 0.45, dy + dh * 0.72);
    ctx.line_to(dx + dw * 0.85, dy + dh * 0.72);
    ctx.close_path();
    ctx.fill();
    set_stroke(ctx, "#d4a030");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(dx + dw * 0.65, dy + dh * 0.48);
    ctx.line_to(dx + dw * 0.65, dy + dh * 0.72);
    ctx.stroke();

    // Dungeon entrance (dark hole)
    set_fill(ctx, "#0a0a0a");
    ctx.begin_path();
    ctx.ellipse(dx + dw * 0.25, dy + dh * 0.82, 22.0, 12.0, 0.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    set_stroke(ctx, "#5a3a15");
    ctx.set_line_width(2.0);
    ctx.begin_path();
    ctx.ellipse(dx + dw * 0.25, dy + dh * 0.82, 22.0, 12.0, 0.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.stroke();

    // Zone border
    set_stroke(ctx, "#c8a040");
    ctx.set_line_width(3.0);
    stroke_rect(ctx, dx, dy, dw, dh);

    // Label
    set_fill(ctx, "#ffe080");
    ctx.set_font("bold 15px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("DESERT", dx + 10.0, dy + 20.0);
    set_fill(ctx, "#aa8828");
    ctx.set_font("10px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("[ Dungeon ]", dx + 10.0, dy + 34.0);

    // ── City Zone ────────────────────────────────────────────────────────────
    let (cx2, cy2, cw, ch) = city_rect();
    let (cx2, cy2, cw, ch) = (cx2 as f64, cy2 as f64, cw as f64, ch as f64);

    // Stone backdrop
    set_fill(ctx, "#2a2a3e");
    fill_rect(ctx, cx2, cy2, cw, ch);

    // City walls — large block pattern
    set_fill(ctx, "#3a3a50");
    fill_rect(ctx, cx2, cy2 + ch * 0.55, cw, ch * 0.45);

    // Wall crenellations (top of wall)
    set_fill(ctx, "#3a3a50");
    let crenel_w = 18.0_f64;
    let crenel_h = 16.0_f64;
    let crenel_y = cy2 + ch * 0.55 - crenel_h;
    let mut cx_crenel = cx2;
    while cx_crenel < cx2 + cw {
        fill_rect(ctx, cx_crenel, crenel_y, crenel_w, crenel_h);
        cx_crenel += crenel_w * 2.0;
    }

    // Gate arch
    set_fill(ctx, "#1a1a28");
    let gate_cx = cx2 + cw * 0.5;
    let gate_top = cy2 + ch * 0.55;
    let gate_w = 44.0_f64;
    let gate_h = 56.0_f64;
    fill_rect(ctx, gate_cx - gate_w / 2.0, gate_top, gate_w, gate_h);
    ctx.begin_path();
    ctx.arc(gate_cx, gate_top, gate_w / 2.0, std::f64::consts::PI, 0.0).unwrap();
    ctx.fill();

    // Gate border
    set_stroke(ctx, "#6a6a8a");
    ctx.set_line_width(2.0);
    ctx.begin_path();
    ctx.move_to(gate_cx - gate_w / 2.0, gate_top + gate_h);
    ctx.line_to(gate_cx - gate_w / 2.0, gate_top);
    ctx.arc(gate_cx, gate_top, gate_w / 2.0, std::f64::consts::PI, 0.0).unwrap();
    ctx.line_to(gate_cx + gate_w / 2.0, gate_top + gate_h);
    ctx.stroke();

    // Tower left
    set_fill(ctx, "#333348");
    fill_rect(ctx, cx2 + 8.0, cy2 + ch * 0.35, 34.0, ch * 0.65);
    fill_rect(ctx, cx2 + 4.0, cy2 + ch * 0.35 - 12.0, 42.0, 14.0); // battlements
    set_fill(ctx, "#222234");
    fill_rect(ctx, cx2 + 19.0, cy2 + ch * 0.55, 10.0, 22.0); // window

    // Tower right
    set_fill(ctx, "#333348");
    fill_rect(ctx, cx2 + cw - 42.0, cy2 + ch * 0.35, 34.0, ch * 0.65);
    fill_rect(ctx, cx2 + cw - 46.0, cy2 + ch * 0.35 - 12.0, 42.0, 14.0);
    set_fill(ctx, "#222234");
    fill_rect(ctx, cx2 + cw - 29.0, cy2 + ch * 0.55, 10.0, 22.0);

    // Zone border
    set_stroke(ctx, "#8888aa");
    ctx.set_line_width(3.0);
    stroke_rect(ctx, cx2, cy2, cw, ch);

    // Label
    set_fill(ctx, "#ccccee");
    ctx.set_font("bold 15px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("CITY", cx2 + 10.0, cy2 + 20.0);
    set_fill(ctx, "#7777aa");
    ctx.set_font("10px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("[ Shop + Rest ]", cx2 + 10.0, cy2 + 34.0);

    // ── Minimal HUD (gold only) ───────────────────────────────────────────────
    set_fill(ctx, C_HUD_BG);
    fill_rect(ctx, 0.0, HUD_Y as f64, CANVAS_W as f64, HUD_H as f64);
    set_stroke(ctx, C_HUD_BORDER);
    ctx.set_line_width(3.0);
    ctx.begin_path();
    ctx.move_to(0.0, HUD_Y as f64);
    ctx.line_to(CANVAS_W as f64, HUD_Y as f64);
    ctx.stroke();

    set_fill(ctx, C_GOLD);
    ctx.set_font("bold 18px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text(&format!("\u{269C} {} Gold", gs.gold), 350.0, 440.0);

    set_fill(ctx, "#4a4a6a");
    ctx.set_font("11px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("Click a location to enter", 320.0, 465.0);
}

// ─── Playing scene ────────────────────────────────────────────────────────────

fn draw_playing(gs: &GameState) {
    let ctx = &gs.ctx;

    ctx.save();
    ctx.translate(gs.screen_shake.offset_x as f64, gs.screen_shake.offset_y as f64).unwrap();

    // Background
    set_fill(ctx, C_BG);
    fill_rect(ctx, 0.0, 0.0, CANVAS_W as f64, HUD_Y as f64);

    match gs.location {
        Location::Desert => {
            draw_dungeon(ctx, gs);

            // Chests (behind characters)
            for chest in &gs.chests {
                crate::sprites::draw_chest(ctx, chest.x as f64, chest.y as f64, chest.opened);
            }
        }
        Location::City => {
            draw_city_scene(ctx, gs);
        }
    }

    // Items on floor
    for item in &gs.items_on_floor {
        draw_item(ctx, item);
    }

    // Enemies (desert only — city is safe)
    for enemy in &gs.enemies {
        if enemy.alive {
            draw_enemy(ctx, enemy);
        }
    }

    // Player
    draw_player(ctx, &gs.player);

    // Particles
    draw_particles(ctx, &gs.particles);

    // Damage numbers
    draw_damage_numbers(ctx, &gs.damage_numbers);

    ctx.restore();

    // HUD (not affected by screen shake)
    draw_hud(ctx, gs);

    // WASM overlay
    draw_overlay(ctx, gs);

    // Room transition fade overlay
    if gs.transition_fade > 0 {
        let alpha = if gs.transition_fade > 8 {
            (gs.transition_fade - 8) as f64 / 8.0
        } else {
            gs.transition_fade as f64 / 8.0
        };
        set_fill(ctx, &format!("rgba(0,0,0,{:.3})", alpha));
        fill_rect(ctx, 0.0, 0.0, CANVAS_W as f64, CANVAS_H as f64);
    }
}

fn draw_city_scene(ctx: &web_sys::CanvasRenderingContext2d, _gs: &GameState) {
    // Sky
    set_fill(ctx, "#1a1e30");
    fill_rect(ctx, 0.0, 0.0, CANVAS_W as f64, HUD_Y as f64);

    // Stone floor
    set_fill(ctx, "#2a2a38");
    fill_rect(ctx, 0.0, 260.0, CANVAS_W as f64, 140.0);

    // Floor tiles
    set_stroke(ctx, "#333344");
    ctx.set_line_width(1.0);
    let tile = 40.0_f64;
    let mut tx = 0.0_f64;
    while tx <= CANVAS_W as f64 {
        ctx.begin_path();
        ctx.move_to(tx, 260.0);
        ctx.line_to(tx, HUD_Y as f64);
        ctx.stroke();
        tx += tile;
    }
    let mut ty = 260.0_f64;
    while ty <= HUD_Y as f64 {
        ctx.begin_path();
        ctx.move_to(0.0, ty);
        ctx.line_to(CANVAS_W as f64, ty);
        ctx.stroke();
        ty += tile;
    }

    // Back wall
    set_fill(ctx, "#222232");
    fill_rect(ctx, 0.0, 30.0, CANVAS_W as f64, 230.0);

    // Wall stone blocks
    set_stroke(ctx, "#2c2c40");
    ctx.set_line_width(1.0);
    let block_w = 80.0_f64;
    let block_h = 30.0_f64;
    let mut row = 0;
    let mut wall_y = 30.0_f64;
    while wall_y < 260.0 {
        let offset = if row % 2 == 0 { 0.0 } else { block_w / 2.0 };
        let mut bx = offset - block_w;
        while bx < CANVAS_W as f64 + block_w {
            ctx.begin_path();
            ctx.rect(bx, wall_y, block_w, block_h);
            ctx.stroke();
            bx += block_w;
        }
        row += 1;
        wall_y += block_h;
    }

    // Torches on the wall
    draw_torch(ctx, 150.0, 120.0);
    draw_torch(ctx, 650.0, 120.0);

    // Barrels (left side props)
    draw_barrel(ctx, 100.0, 300.0);
    draw_barrel(ctx, 125.0, 290.0);

    // Merchant sprite
    crate::sprites::draw_merchant(ctx, MERCHANT_X as f64, MERCHANT_Y as f64);

    // Merchant label / prompt
    set_fill(ctx, C_GOLD);
    ctx.set_font("11px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("MERCHANT", MERCHANT_X as f64 - 30.0, MERCHANT_Y as f64 - 50.0);
    set_fill(ctx, "#888888");
    ctx.set_font("9px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("[click to shop]", MERCHANT_X as f64 - 30.0, MERCHANT_Y as f64 - 38.0);

    // City label
    set_fill(ctx, "#4a4a6a");
    ctx.set_font("10px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("CITY — Safe Zone", 20.0, 50.0);
}

fn draw_torch(ctx: &web_sys::CanvasRenderingContext2d, x: f64, y: f64) {
    // Handle
    set_fill(ctx, "#5a3a15");
    fill_rect(ctx, x - 3.0, y, 6.0, 14.0);
    // Flame
    set_fill(ctx, "#ff8800");
    ctx.begin_path();
    ctx.move_to(x, y - 12.0);
    ctx.quadratic_curve_to(x + 6.0, y - 4.0, x + 2.0, y);
    ctx.quadratic_curve_to(x, y - 2.0, x - 2.0, y);
    ctx.quadratic_curve_to(x - 6.0, y - 4.0, x, y - 12.0);
    ctx.fill();
    set_fill(ctx, "#ffdd00");
    ctx.begin_path();
    ctx.move_to(x, y - 8.0);
    ctx.quadratic_curve_to(x + 3.0, y - 3.0, x, y);
    ctx.quadratic_curve_to(x - 3.0, y - 3.0, x, y - 8.0);
    ctx.fill();
}

fn draw_barrel(ctx: &web_sys::CanvasRenderingContext2d, x: f64, y: f64) {
    set_fill(ctx, "#3a2a10");
    ctx.begin_path();
    ctx.ellipse(x, y, 12.0, 18.0, 0.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    // Hoops
    set_stroke(ctx, "#5a3a10");
    ctx.set_line_width(2.0);
    for offset in &[-7.0_f64, 0.0, 7.0] {
        ctx.begin_path();
        ctx.ellipse(x, y + offset, 12.0, 4.0, 0.0, 0.0, std::f64::consts::TAU).unwrap();
        ctx.stroke();
    }
}

fn draw_dungeon(ctx: &web_sys::CanvasRenderingContext2d, gs: &GameState) {
    let room = gs.dungeon.current_room();
    let room_idx = room.index;

    // Outer wall
    set_fill(ctx, C_WALL);
    fill_rect(ctx, room.x as f64, room.y as f64, room.w as f64, room.h as f64);

    // Wall edge highlight
    set_stroke(ctx, C_WALL_EDGE);
    ctx.set_line_width(2.0);
    stroke_rect(ctx, room.x as f64, room.y as f64, room.w as f64, room.h as f64);

    let fx = room.floor_x() as f64;
    let fy = room.floor_y() as f64;
    let fw = room.floor_w() as f64;
    let fh = room.floor_h() as f64;

    // Floor (desert-tinted for room 0, dark blue-stone for room 1)
    let (floor_col, grid_col) = if room_idx == 0 {
        ("#1e1810", "#252018")
    } else {
        (C_FLOOR_ALT, C_FLOOR_GRID_ALT)
    };
    set_fill(ctx, floor_col);
    fill_rect(ctx, fx, fy, fw, fh);

    // Floor grid
    set_stroke(ctx, grid_col);
    ctx.set_line_width(0.5);
    let grid = 40.0_f64;
    let cols = (fw / grid).ceil() as i32;
    let rows = (fh / grid).ceil() as i32;
    for c in 0..=cols {
        let x = fx + c as f64 * grid;
        ctx.begin_path();
        ctx.move_to(x, fy);
        ctx.line_to(x, fy + fh);
        ctx.stroke();
    }
    for r in 0..=rows {
        let y = fy + r as f64 * grid;
        ctx.begin_path();
        ctx.move_to(fx, y);
        ctx.line_to(fx + fw, y);
        ctx.stroke();
    }

    // Room label
    if room_idx == 1 {
        set_fill(ctx, "#3a4a6a");
        ctx.set_font("10px 'VT323', 'Courier New', monospace");
        let _ = ctx.fill_text("CATACOMBS", fx + 4.0, fy + 14.0);
    } else {
        set_fill(ctx, "#3a2a14");
        ctx.set_font("10px 'VT323', 'Courier New', monospace");
        let _ = ctx.fill_text("DESERT RUINS", fx + 4.0, fy + 14.0);
    }

    // Door arch
    draw_door(ctx, room);
}

fn draw_door(ctx: &web_sys::CanvasRenderingContext2d, room: &crate::dungeon::Room) {
    let (dcx, dcy) = room.door_center();
    let (dcx, dcy) = (dcx as f64, dcy as f64);

    let door_w = 50.0_f64;
    let door_h = 40.0_f64;
    let pillar_w = 8.0_f64;

    let door_x = dcx - door_w / 2.0;
    let door_y = dcy - door_h;

    set_fill(ctx, "#3a2a0a");
    ctx.fill_rect(door_x, door_y, pillar_w, door_h);
    ctx.fill_rect(door_x + door_w - pillar_w, door_y, pillar_w, door_h);
    set_fill(ctx, "#1a1000");
    ctx.fill_rect(door_x + pillar_w, door_y + 10.0, door_w - pillar_w * 2.0, door_h - 10.0);

    set_fill(ctx, "#1a1000");
    ctx.begin_path();
    let _ = ctx.arc(dcx, door_y + 10.0, (door_w - pillar_w * 2.0) / 2.0, std::f64::consts::PI, 0.0);
    ctx.fill();

    set_stroke(ctx, "#5a3a15");
    ctx.set_line_width(2.0);
    ctx.stroke_rect(door_x, door_y, door_w, door_h);

    set_stroke(ctx, "#d4af37");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(door_x + pillar_w, door_y + door_h);
    ctx.line_to(door_x + pillar_w, door_y + 10.0);
    let _ = ctx.arc(dcx, door_y + 10.0, (door_w - pillar_w * 2.0) / 2.0, std::f64::consts::PI, 0.0);
    ctx.line_to(door_x + door_w - pillar_w, door_y + door_h);
    ctx.stroke();

    set_fill(ctx, "#d4af37");
    ctx.set_font("9px 'VT323', 'Courier New', monospace");
    let label = if room.index == 0 { "LOC 2 \u{25BC}" } else { "LOC 1 \u{25B2}" };
    let _ = ctx.fill_text(label, dcx - 20.0, door_y - 4.0);
}

fn draw_item(ctx: &web_sys::CanvasRenderingContext2d, item: &Item) {
    let x = item.x as f64;
    let y = item.y as f64;

    match item.kind {
        ItemKind::HpPotion | ItemKind::MpPotion => {
            let is_hp = item.kind == ItemKind::HpPotion;
            crate::sprites::draw_potion(ctx, x, y, is_hp);
        }
        _ => {
            // Equipment: draw rotating diamond
            let color = match item.kind {
                ItemKind::Sword => "#aaaacc",
                ItemKind::Staff => "#8866aa",
                ItemKind::Tome => "#aa6622",
                _ => "#888888",
            };
            ctx.save();
            ctx.translate(x, y).unwrap();
            let _ = ctx.rotate(std::f64::consts::PI / 4.0);
            set_fill(ctx, color);
            ctx.fill_rect(-5.0, -5.0, 10.0, 10.0);
            ctx.restore();

            if item.label_life > 0 {
                let alpha = (item.label_life as f64 / 180.0).min(1.0);
                let label = format!("[E] {}", item.kind.name());
                set_fill(ctx, &format!("rgba(200,184,154,{:.2})", alpha));
                ctx.set_font("9px 'VT323', 'Courier New', monospace");
                let _ = ctx.fill_text(&label, x - 20.0, y - 12.0);
            }
        }
    }
}

fn draw_enemy(ctx: &web_sys::CanvasRenderingContext2d, enemy: &crate::entities::Enemy) {
    let x = enemy.x as f64;
    let y = enemy.y as f64;

    if enemy.frozen_timer > 0.0 {
        ctx.save();
        ctx.set_global_alpha(0.7);
    }

    crate::sprites::draw_skeleton(ctx, x, y);

    if enemy.frozen_timer > 0.0 {
        ctx.restore();
        set_fill(ctx, "rgba(100,150,255,0.3)");
        fill_rect(ctx, x - 21.0, y - 32.0, 42.0, 52.0);
    }

    if enemy.state != crate::entities::EnemyState::Patrolling {
        set_fill(ctx, C_AGGRO);
        ctx.begin_path();
        ctx.arc(x, y - 40.0, 4.0, 0.0, std::f64::consts::TAU).unwrap();
        ctx.fill();
    }

    // HP bar
    let bar_w = 36.0_f64;
    let bar_h = 4.0_f64;
    let bar_x = x - 18.0;
    let bar_y = y + 22.0;
    set_fill(ctx, C_HP_BAR_BG);
    fill_rect(ctx, bar_x, bar_y, bar_w, bar_h);
    let hp_frac = (enemy.hp / enemy.max_hp) as f64;
    set_fill(ctx, C_HP_BAR);
    fill_rect(ctx, bar_x, bar_y, bar_w * hp_frac, bar_h);
}

fn draw_player(ctx: &web_sys::CanvasRenderingContext2d, player: &crate::entities::Player) {
    let x = player.x as f64;
    let y = player.y as f64;

    match player.class {
        PlayerClass::Warrior => crate::sprites::draw_warrior(ctx, x, y),
        PlayerClass::Magician => crate::sprites::draw_magician(ctx, x, y),
    }

    // HP bar
    let bar_w = 36.0_f64;
    let bar_h = 4.0_f64;
    let bar_x = x - 18.0;
    let bar_y = y + 30.0;
    set_fill(ctx, C_HP_BAR_BG);
    fill_rect(ctx, bar_x, bar_y, bar_w, bar_h);
    let hp_frac = (player.hp / player.max_hp) as f64;
    set_fill(ctx, C_HP_BAR);
    fill_rect(ctx, bar_x, bar_y, bar_w * hp_frac, bar_h);
}

fn draw_particles(ctx: &web_sys::CanvasRenderingContext2d, pool: &crate::particles::ParticlePool) {
    for p in pool.particles.iter() {
        if !p.active {
            continue;
        }
        let alpha = (p.life / p.max_life).min(1.0) as f64;
        let color = format!("rgba(204,34,34,{:.2})", alpha);
        set_fill(ctx, &color);
        ctx.begin_path();
        ctx.arc(p.x as f64, p.y as f64, 3.0, 0.0, std::f64::consts::TAU).unwrap();
        ctx.fill();
    }
}

fn draw_damage_numbers(ctx: &web_sys::CanvasRenderingContext2d, nums: &[crate::game::DamageNumber]) {
    for dn in nums.iter() {
        let alpha = (dn.life as f64 / 30.0).min(1.0);
        let color = format!("rgba(255,238,68,{:.2})", alpha);
        set_fill(ctx, &color);
        ctx.set_font("bold 16px 'VT323', 'Courier New', monospace");
        let _ = ctx.fill_text(&format!("-{}", dn.value), dn.x as f64 - 10.0, dn.y as f64);
    }
}

fn draw_hud(ctx: &web_sys::CanvasRenderingContext2d, gs: &GameState) {
    // HUD background
    set_fill(ctx, C_HUD_BG);
    fill_rect(ctx, 0.0, HUD_Y as f64, CANVAS_W as f64, HUD_H as f64);

    // HUD top border
    set_stroke(ctx, C_HUD_BORDER);
    ctx.set_line_width(3.0);
    ctx.begin_path();
    ctx.move_to(0.0, HUD_Y as f64);
    ctx.line_to(CANVAS_W as f64, HUD_Y as f64);
    ctx.stroke();

    // HP Orb
    draw_orb(ctx, 50.0, 450.0, 38.0, C_HP_ORB, gs.player.hp / gs.player.max_hp);
    set_fill(ctx, "#cc4444");
    ctx.set_font("11px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("HP", 38.0, 497.0);

    // Mana Orb
    let mana_frac = gs.player.mana / gs.player.max_mana;
    draw_orb(ctx, 750.0, 450.0, 38.0, C_MANA_ORB, mana_frac);
    set_fill(ctx, "#4466cc");
    ctx.set_font("11px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("MP", 738.0, 497.0);

    // Ability slot + cooldown (center-left)
    let ability_x = 106.0_f64;
    let ability_y = 412.0_f64;
    let ability_size = 44.0_f64;

    set_stroke(ctx, C_HUD_BORDER);
    ctx.set_line_width(2.0);
    set_fill(ctx, "#111122");
    fill_rect(ctx, ability_x, ability_y, ability_size, ability_size);
    stroke_rect(ctx, ability_x, ability_y, ability_size, ability_size);

    set_fill(ctx, C_GOLD);
    ctx.set_font("22px 'VT323', 'Courier New', monospace");
    let icon = match gs.player.class {
        PlayerClass::Warrior => "\u{2694}",
        PlayerClass::Magician => "\u{2744}",
    };
    let _ = ctx.fill_text(icon, ability_x + 11.0, ability_y + 30.0);

    if gs.player.ability_cooldown > 0.0 {
        let cd_frac = (gs.player.ability_cooldown / gs.player.class.ability_cooldown_max()) as f64;
        set_fill(ctx, &format!("rgba(0,0,0,{:.2})", cd_frac * 0.8));
        fill_rect(ctx, ability_x, ability_y, ability_size, ability_size);
        set_fill(ctx, "#ffffff");
        ctx.set_font("bold 12px 'VT323', 'Courier New', monospace");
        let cd_text = format!("{:.1}", gs.player.ability_cooldown);
        let _ = ctx.fill_text(&cd_text, ability_x + 8.0, ability_y + 26.0);
    }

    // Equipment slots
    let slot_start_x = 162.0_f64;
    let slot_y = 416.0_f64;
    let slot_size = 40.0_f64;
    let slot_gap = 6.0_f64;

    for i in 0..3usize {
        let sx = slot_start_x + i as f64 * (slot_size + slot_gap);
        let is_equipped = gs.player.equipment[i].is_some();

        set_fill(ctx, "#111122");
        fill_rect(ctx, sx, slot_y, slot_size, slot_size);
        if is_equipped {
            set_stroke(ctx, C_GOLD);
        } else {
            set_stroke(ctx, C_HUD_BORDER);
        }
        ctx.set_line_width(2.0);
        stroke_rect(ctx, sx, slot_y, slot_size, slot_size);

        if let Some(kind) = gs.player.equipment[i] {
            let item_color = match kind {
                ItemKind::Sword => "#aaaacc",
                ItemKind::Staff => "#8866aa",
                ItemKind::Tome => "#aa6622",
                _ => "#888888",
            };
            ctx.save();
            ctx.translate(sx + slot_size / 2.0, slot_y + slot_size / 2.0).unwrap();
            let _ = ctx.rotate(std::f64::consts::PI / 4.0);
            set_fill(ctx, item_color);
            fill_rect(ctx, -7.0, -7.0, 14.0, 14.0);
            ctx.restore();

            set_fill(ctx, C_GOLD);
            ctx.set_font("bold 8px 'VT323', 'Courier New', monospace");
            let _ = ctx.fill_text("E", sx + slot_size - 10.0, slot_y + 10.0);
        }
    }

    // Potion slots ([1] HP, [2] MP)
    let potion_start_x = slot_start_x + 3.0 * (slot_size + slot_gap) + 10.0;
    let potion_size = 32.0_f64;
    let potion_y = slot_y + (slot_size - potion_size) / 2.0;

    // HP potion slot
    {
        let px = potion_start_x;
        set_fill(ctx, "#110808");
        fill_rect(ctx, px, potion_y, potion_size, potion_size);
        set_stroke(ctx, if gs.player.hp_potions > 0 { C_POTION_HP } else { C_HUD_BORDER });
        ctx.set_line_width(2.0);
        stroke_rect(ctx, px, potion_y, potion_size, potion_size);
        if gs.player.hp_potions > 0 {
            crate::sprites::draw_potion(ctx, px + potion_size / 2.0, potion_y + potion_size / 2.0 - 2.0, true);
        }
        set_fill(ctx, "#888888");
        ctx.set_font("8px 'VT323', 'Courier New', monospace");
        let _ = ctx.fill_text("[1]", px + 2.0, potion_y + potion_size - 2.0);
        if gs.player.hp_potions > 0 {
            set_fill(ctx, C_POTION_HP);
            ctx.set_font("bold 9px 'VT323', 'Courier New', monospace");
            let _ = ctx.fill_text(&format!("x{}", gs.player.hp_potions), px + potion_size - 14.0, potion_y + 10.0);
        }
    }

    // MP potion slot
    {
        let px = potion_start_x + potion_size + 4.0;
        set_fill(ctx, "#080811");
        fill_rect(ctx, px, potion_y, potion_size, potion_size);
        set_stroke(ctx, if gs.player.mp_potions > 0 { C_POTION_MP } else { C_HUD_BORDER });
        ctx.set_line_width(2.0);
        stroke_rect(ctx, px, potion_y, potion_size, potion_size);
        if gs.player.mp_potions > 0 {
            crate::sprites::draw_potion(ctx, px + potion_size / 2.0, potion_y + potion_size / 2.0 - 2.0, false);
        }
        set_fill(ctx, "#888888");
        ctx.set_font("8px 'VT323', 'Courier New', monospace");
        let _ = ctx.fill_text("[2]", px + 2.0, potion_y + potion_size - 2.0);
        if gs.player.mp_potions > 0 {
            set_fill(ctx, C_POTION_MP);
            ctx.set_font("bold 9px 'VT323', 'Courier New', monospace");
            let _ = ctx.fill_text(&format!("x{}", gs.player.mp_potions), px + potion_size - 14.0, potion_y + 10.0);
        }
    }

    // Key hints
    set_fill(ctx, "#3a3a5a");
    ctx.set_font("9px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("[ E ] equip  [ Space ] ability  [ ESC ] map", slot_start_x, slot_y + slot_size + 14.0);

    // Gold
    set_fill(ctx, C_GOLD);
    ctx.set_font("bold 14px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text(&format!("\u{269C} {}", gs.gold), 390.0, 430.0);

    // Location indicator
    set_fill(ctx, "#5a5a7a");
    ctx.set_font("10px 'VT323', 'Courier New', monospace");
    let loc_label = match gs.location {
        Location::Desert => {
            if gs.dungeon.current_room == 0 { "Desert Ruins" } else { "Catacombs" }
        }
        Location::City => "City",
    };
    let _ = ctx.fill_text(loc_label, 370.0, 445.0);

    // HUD flash text
    if let Some(ref flash) = gs.hud_flash {
        let alpha = flash.life as f64 / 60.0;
        set_fill(ctx, &format!("rgba(255,238,68,{:.2})", alpha));
        ctx.set_font("bold 12px 'VT323', 'Courier New', monospace");
        let _ = ctx.fill_text(flash.text, 310.0, 470.0);
    }
}

fn draw_orb(
    ctx: &web_sys::CanvasRenderingContext2d,
    cx: f64,
    cy: f64,
    r: f64,
    color: &str,
    fraction: f32,
) {
    ctx.save();
    ctx.begin_path();
    ctx.arc(cx, cy, r, 0.0, std::f64::consts::TAU).unwrap();
    ctx.clip();

    set_fill(ctx, "#111122");
    fill_rect(ctx, cx - r, cy - r, r * 2.0, r * 2.0);

    let fill_h = r * 2.0 * fraction as f64;
    let fill_y = cy + r - fill_h;
    set_fill(ctx, color);
    fill_rect(ctx, cx - r, fill_y, r * 2.0, fill_h);

    ctx.restore();

    set_stroke(ctx, C_HUD_BORDER);
    ctx.set_line_width(3.0);
    ctx.begin_path();
    ctx.arc(cx, cy, r, 0.0, std::f64::consts::TAU).unwrap();
    ctx.stroke();
}

// ─── Shop Overlay ─────────────────────────────────────────────────────────────

// Layout constants (exported so game.rs can use shop_hit_test)
const SHOP_PANEL_X: f64 = 120.0;
const SHOP_PANEL_Y: f64 = 60.0;
const SHOP_PANEL_W: f64 = 560.0;
const SHOP_PANEL_H: f64 = 280.0;
const SHOP_SLOT_SIZE: f64 = 64.0;
const SHOP_SLOT_GAP: f64 = 16.0;
const SHOP_SLOTS_X: f64 = SHOP_PANEL_X + 24.0;
const SHOP_SLOTS_Y: f64 = SHOP_PANEL_Y + 80.0;

/// Hit-test a click against shop item slots. Returns Some(index) if a slot was clicked.
pub fn shop_hit_test(x: f32, y: f32, count: usize) -> Option<usize> {
    let (mx, my) = (x as f64, y as f64);
    for i in 0..count {
        let sx = SHOP_SLOTS_X + i as f64 * (SHOP_SLOT_SIZE + SHOP_SLOT_GAP);
        let sy = SHOP_SLOTS_Y;
        if mx >= sx && mx <= sx + SHOP_SLOT_SIZE && my >= sy && my <= sy + SHOP_SLOT_SIZE {
            return Some(i);
        }
    }
    None
}

fn draw_shop_overlay(gs: &GameState) {
    let ctx = &gs.ctx;

    // Dim background
    set_fill(ctx, "rgba(0,0,0,0.72)");
    fill_rect(ctx, 0.0, 0.0, CANVAS_W as f64, HUD_Y as f64);

    // Panel
    set_fill(ctx, "#0d0d1e");
    fill_rect(ctx, SHOP_PANEL_X, SHOP_PANEL_Y, SHOP_PANEL_W, SHOP_PANEL_H);
    set_stroke(ctx, C_GOLD);
    ctx.set_line_width(3.0);
    stroke_rect(ctx, SHOP_PANEL_X, SHOP_PANEL_Y, SHOP_PANEL_W, SHOP_PANEL_H);

    // Inner border (decorative)
    set_stroke(ctx, "#5a3a15");
    ctx.set_line_width(1.0);
    stroke_rect(ctx, SHOP_PANEL_X + 6.0, SHOP_PANEL_Y + 6.0, SHOP_PANEL_W - 12.0, SHOP_PANEL_H - 12.0);

    // Header
    set_fill(ctx, C_GOLD);
    ctx.set_font("bold 20px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("MERCHANT", SHOP_PANEL_X + SHOP_PANEL_W / 2.0 - 60.0, SHOP_PANEL_Y + 36.0);

    set_fill(ctx, "#888888");
    ctx.set_font("11px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text(
        &format!("Gold: {}", gs.gold),
        SHOP_PANEL_X + SHOP_PANEL_W - 100.0,
        SHOP_PANEL_Y + 36.0,
    );

    // Separator
    set_stroke(ctx, "#5a3a15");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(SHOP_PANEL_X + 16.0, SHOP_PANEL_Y + 50.0);
    ctx.line_to(SHOP_PANEL_X + SHOP_PANEL_W - 16.0, SHOP_PANEL_Y + 50.0);
    ctx.stroke();

    // Item slots
    for (i, shop_item) in gs.shop_items.iter().enumerate() {
        let sx = SHOP_SLOTS_X + i as f64 * (SHOP_SLOT_SIZE + SHOP_SLOT_GAP);
        let sy = SHOP_SLOTS_Y;

        let is_selected = gs.shop_selected == Some(i);

        // Slot background
        set_fill(ctx, if is_selected { "#1a1a38" } else { "#111122" });
        fill_rect(ctx, sx, sy, SHOP_SLOT_SIZE, SHOP_SLOT_SIZE);

        // Slot border (gold when selected)
        if is_selected {
            set_stroke(ctx, C_GOLD);
            ctx.set_line_width(3.0);
        } else {
            set_stroke(ctx, C_HUD_BORDER);
            ctx.set_line_width(1.0);
        }
        stroke_rect(ctx, sx, sy, SHOP_SLOT_SIZE, SHOP_SLOT_SIZE);

        // Item sprite in slot
        let cx_slot = sx + SHOP_SLOT_SIZE / 2.0;
        let cy_slot = sy + SHOP_SLOT_SIZE / 2.0 - 4.0;
        match shop_item.kind {
            ItemKind::HpPotion => crate::sprites::draw_potion(ctx, cx_slot, cy_slot, true),
            ItemKind::MpPotion => crate::sprites::draw_potion(ctx, cx_slot, cy_slot, false),
            _ => crate::sprites::draw_floor_item(ctx, cx_slot, cy_slot, shop_item.kind),
        }

        // Item name
        let can_afford = gs.gold >= shop_item.price;
        set_fill(ctx, if can_afford { "#cccccc" } else { "#666666" });
        ctx.set_font("9px 'VT323', 'Courier New', monospace");
        let name = shop_item.kind.name();
        let name_x = sx + SHOP_SLOT_SIZE / 2.0 - (name.len() as f64 * 4.5);
        let _ = ctx.fill_text(name, name_x, sy + SHOP_SLOT_SIZE + 14.0);

        // Price
        set_fill(ctx, if can_afford { C_GOLD } else { "#664422" });
        ctx.set_font("bold 10px 'VT323', 'Courier New', monospace");
        let price_str = format!("{}g", shop_item.price);
        let price_x = sx + SHOP_SLOT_SIZE / 2.0 - (price_str.len() as f64 * 4.0);
        let _ = ctx.fill_text(&price_str, price_x, sy + SHOP_SLOT_SIZE + 26.0);
    }

    // Selected item detail text
    if let Some(sel) = gs.shop_selected {
        if let Some(item) = gs.shop_items.get(sel) {
            let detail_y = SHOP_PANEL_Y + SHOP_PANEL_H - 38.0;
            set_fill(ctx, "#aaaaaa");
            ctx.set_font("11px 'VT323', 'Courier New', monospace");
            let detail = match item.kind {
                ItemKind::Sword => "Increases melee damage",
                ItemKind::Staff => "Enables frost nova ability",
                ItemKind::Tome => "Increases magic power",
                ItemKind::HpPotion => "Restores 30 HP  [key: 1]",
                ItemKind::MpPotion => "Restores 30 MP  [key: 2]",
            };
            let _ = ctx.fill_text(detail, SHOP_PANEL_X + 20.0, detail_y);
        }
    }

    // Key hints
    set_fill(ctx, "#4a4a6a");
    ctx.set_font("10px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text(
        "[ \u{2190} \u{2192} ] select  [ Enter ] buy  [ ESC ] close",
        SHOP_PANEL_X + 130.0,
        SHOP_PANEL_Y + SHOP_PANEL_H - 14.0,
    );
}

fn draw_overlay(ctx: &web_sys::CanvasRenderingContext2d, gs: &GameState) {
    let ox = 622.0_f64;
    let oy = 8.0_f64;
    let ow = 170.0_f64;
    let oh = 98.0_f64;

    set_fill(ctx, C_OVERLAY_BG);
    fill_rect(ctx, ox, oy, ow, oh);

    set_stroke(ctx, "#334433");
    ctx.set_line_width(1.0);
    stroke_rect(ctx, ox, oy, ow, oh);

    set_fill(ctx, C_OVERLAY_TEXT);
    ctx.set_font("bold 11px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("\u{25C6} WASM RUNTIME", ox + 8.0, oy + 18.0);

    ctx.set_font("11px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text(&format!("{:.1}ms/frame", gs.last_frame_ms), ox + 8.0, oy + 34.0);
    let _ = ctx.fill_text(
        &format!("{} entities", gs.enemies.iter().filter(|e| e.alive).count() + 1),
        ox + 8.0,
        oy + 50.0,
    );
    let _ = ctx.fill_text(
        &format!("{}KB WASM mem", crate::overlay::get_memory_kb()),
        ox + 8.0,
        oy + 66.0,
    );

    set_fill(ctx, "#88cc88");
    ctx.set_font("10px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("All state in Rust/WASM", ox + 8.0, oy + 88.0);
}

fn draw_game_over(ctx: &web_sys::CanvasRenderingContext2d) {
    set_fill(ctx, "rgba(0,0,0,0.65)");
    fill_rect(ctx, 0.0, 0.0, CANVAS_W as f64, CANVAS_H as f64);

    set_fill(ctx, "#cc2222");
    ctx.set_font("bold 48px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("YOU DIED", 260.0, 220.0);

    set_fill(ctx, "#888888");
    ctx.set_font("18px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("Click to return to title", 258.0, 265.0);
}
