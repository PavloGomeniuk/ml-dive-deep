use wasm_bindgen::JsValue;
use web_sys::CanvasRenderingContext2d;
use crate::renderer::{CANVAS_W, CANVAS_H};
use crate::entities::PlayerClass;

// Card rects: x, y, w, h
pub const WARRIOR_CARD: (f32, f32, f32, f32) = (80.0, 60.0, 280.0, 310.0);
pub const MAGICIAN_CARD: (f32, f32, f32, f32) = (440.0, 60.0, 280.0, 310.0);

fn sf(ctx: &CanvasRenderingContext2d, color: &str) {
    ctx.set_fill_style(&JsValue::from_str(color));
}

fn ss(ctx: &CanvasRenderingContext2d, color: &str) {
    ctx.set_stroke_style(&JsValue::from_str(color));
}

fn fill_rect(ctx: &CanvasRenderingContext2d, x: f64, y: f64, w: f64, h: f64) {
    ctx.fill_rect(x, y, w, h);
}

fn stroke_rect(ctx: &CanvasRenderingContext2d, x: f64, y: f64, w: f64, h: f64) {
    ctx.stroke_rect(x, y, w, h);
}

pub fn render(ctx: &CanvasRenderingContext2d, selected: Option<usize>) {
    let w = CANVAS_W as f64;
    let h = CANVAS_H as f64;

    // Background
    sf(ctx, "#0d0d1a");
    ctx.fill_rect(0.0, 0.0, w, h);

    // Header
    sf(ctx, "#d4af37");
    ctx.set_font("bold 14px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("CHOOSE YOUR FATE", w / 2.0 - 108.0, 45.0);

    // Divider
    ss(ctx, "#5a3a15");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(40.0, 55.0);
    ctx.line_to(w - 40.0, 55.0);
    ctx.stroke();

    // Cards
    draw_card(ctx, WARRIOR_CARD, PlayerClass::Warrior, selected == Some(0));
    draw_card(ctx, MAGICIAN_CARD, PlayerClass::Magician, selected == Some(1));

    // Hint bar
    sf(ctx, "#3a3a5a");
    ctx.set_font("10px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text(
        "click to select  \u{00B7}  ENTER to begin  \u{00B7}  ESC for title",
        w / 2.0 - 162.0,
        h - 20.0,
    );
}

fn draw_card(
    ctx: &CanvasRenderingContext2d,
    rect: (f32, f32, f32, f32),
    class: PlayerClass,
    is_selected: bool,
) {
    let (rx, ry, rw, rh) = rect;
    let (rx, ry, rw, rh) = (rx as f64, ry as f64, rw as f64, rh as f64);

    // Card background
    sf(ctx, "#0f0f1e");
    fill_rect(ctx, rx, ry, rw, rh);

    // Border
    if is_selected {
        ss(ctx, "#d4af37");
        ctx.set_line_width(2.0);
        ctx.set_shadow_blur(15.0);
        ctx.set_shadow_color("rgba(212,175,55,0.4)");
    } else {
        ss(ctx, "#5a3a15");
        ctx.set_line_width(2.0);
        ctx.set_shadow_blur(0.0);
    }
    stroke_rect(ctx, rx, ry, rw, rh);
    ctx.set_shadow_blur(0.0);

    // Sprite area (120px high)
    let sprite_cx = rx + rw / 2.0;
    let sprite_cy = ry + 60.0;
    draw_class_sprite(ctx, sprite_cx, sprite_cy, class, 1.5);

    // Class name
    sf(ctx, "#d4af37");
    ctx.set_font("bold 16px 'VT323', 'Courier New', monospace");
    let name = match class {
        PlayerClass::Warrior => "WARRIOR",
        PlayerClass::Magician => "MAGICIAN",
    };
    let name_x = rx + rw / 2.0 - (name.len() as f64 * 9.6) / 2.0;
    let _ = ctx.fill_text(name, name_x, ry + 140.0);

    // Stats
    sf(ctx, "#8888aa");
    ctx.set_font("11px 'VT323', 'Courier New', monospace");
    let (hp, mana) = match class {
        PlayerClass::Warrior => ("HP: 150  ATK: 15-20", "MANA: 30"),
        PlayerClass::Magician => ("HP: 90   ATK: 12-18", "MANA: 100"),
    };
    let _ = ctx.fill_text(hp, rx + 16.0, ry + 162.0);
    let _ = ctx.fill_text(mana, rx + 16.0, ry + 178.0);

    // Ability box
    let ab_y = ry + 196.0;
    sf(ctx, "#1a1a2e");
    fill_rect(ctx, rx + 10.0, ab_y, rw - 20.0, 60.0);
    ss(ctx, "#3a3a5a");
    ctx.set_line_width(1.0);
    stroke_rect(ctx, rx + 10.0, ab_y, rw - 20.0, 60.0);

    let (ability_icon, ability_name, ability_desc) = match class {
        PlayerClass::Warrior => ("\u{2694}", "CLEAVE  [Space]", "AoE melee, 80px radius, 4s CD"),
        PlayerClass::Magician => ("\u{2744}", "FROST NOVA  [Space]", "AoE freeze, 70px radius, 3s CD"),
    };
    sf(ctx, "#d4af37");
    ctx.set_font("bold 12px 'VT323', 'Courier New', monospace");
    let ability_text = format!("{} {}", ability_icon, ability_name);
    let _ = ctx.fill_text(&ability_text, rx + 18.0, ab_y + 20.0);
    sf(ctx, "#5a5a7a");
    ctx.set_font("10px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text(ability_desc, rx + 18.0, ab_y + 38.0);

    // Selected indicator
    if is_selected {
        sf(ctx, "rgba(212,175,55,0.08)");
        fill_rect(ctx, rx, ry, rw, rh);
        sf(ctx, "#d4af37");
        ctx.set_font("10px 'VT323', 'Courier New', monospace");
        let _ = ctx.fill_text("SELECTED - ENTER to confirm", rx + 20.0, ry + rh - 14.0);
    }
}

/// Draw a scaled class sprite for the character select screen.
pub fn draw_class_sprite(
    ctx: &CanvasRenderingContext2d,
    cx: f64,
    cy: f64,
    class: PlayerClass,
    scale: f64,
) {
    match class {
        PlayerClass::Warrior => draw_warrior(ctx, cx, cy, scale),
        PlayerClass::Magician => draw_magician(ctx, cx, cy, scale),
    }
}

fn sf_js(ctx: &CanvasRenderingContext2d, color: &str) {
    ctx.set_fill_style(&JsValue::from_str(color));
}

fn ss_js(ctx: &CanvasRenderingContext2d, color: &str) {
    ctx.set_stroke_style(&JsValue::from_str(color));
}

fn draw_warrior(ctx: &CanvasRenderingContext2d, cx: f64, cy: f64, s: f64) {
    // Body: 22x28
    sf_js(ctx, "#c8b89a");
    ctx.fill_rect(cx - 11.0 * s, cy - 10.0 * s, 22.0 * s, 28.0 * s);

    // Head
    ctx.begin_path();
    let _ = ctx.arc(cx, cy - 18.0 * s, 9.0 * s, 0.0, std::f64::consts::TAU);
    ctx.fill();

    // Helmet
    sf_js(ctx, "#888888");
    ctx.fill_rect(cx - 11.0 * s, cy - 27.0 * s, 22.0 * s, 6.0 * s);

    // Shield (left)
    ctx.fill_rect(cx - 19.0 * s, cy - 8.0 * s, 8.0 * s, 10.0 * s);

    // Sword
    ss_js(ctx, "#aaaacc");
    ctx.set_line_width(3.0 * s);
    ctx.begin_path();
    ctx.move_to(cx + 11.0 * s, cy - 5.0 * s);
    ctx.line_to(cx + 28.0 * s, cy - 22.0 * s);
    ctx.stroke();
    // Crossguard
    ctx.set_line_width(2.0 * s);
    ctx.begin_path();
    ctx.move_to(cx + 14.0 * s, cy - 1.0 * s);
    ctx.line_to(cx + 22.0 * s, cy - 9.0 * s);
    ctx.stroke();
}

fn draw_magician(ctx: &CanvasRenderingContext2d, cx: f64, cy: f64, s: f64) {
    // Body: 16x26 (narrower, purple robes)
    sf_js(ctx, "#9988cc");
    ctx.fill_rect(cx - 8.0 * s, cy - 10.0 * s, 16.0 * s, 26.0 * s);

    // Head
    sf_js(ctx, "#c8b89a");
    ctx.begin_path();
    let _ = ctx.arc(cx, cy - 18.0 * s, 8.0 * s, 0.0, std::f64::consts::TAU);
    ctx.fill();

    // Pointed hat (trapezoid)
    sf_js(ctx, "#4444aa");
    ctx.begin_path();
    ctx.move_to(cx, cy - 38.0 * s);          // tip
    ctx.line_to(cx + 12.0 * s, cy - 24.0 * s); // base right
    ctx.line_to(cx - 12.0 * s, cy - 24.0 * s); // base left
    ctx.close_path();
    ctx.fill();

    // Staff (left side)
    ss_js(ctx, "#8888aa");
    ctx.set_line_width(2.0 * s);
    ctx.begin_path();
    ctx.move_to(cx - 14.0 * s, cy - 28.0 * s);
    ctx.line_to(cx - 14.0 * s, cy + 20.0 * s);
    ctx.stroke();

    // Orb at staff top
    ctx.set_shadow_color("rgba(136,136,255,0.8)");
    ctx.set_shadow_blur(6.0 * s);
    sf_js(ctx, "#aaaaff");
    ctx.begin_path();
    let _ = ctx.arc(cx - 14.0 * s, cy - 28.0 * s, 4.0 * s, 0.0, std::f64::consts::TAU);
    ctx.fill();
    ctx.set_shadow_blur(0.0);
}

/// Hit test for a card rect.
pub fn hit_test_card(x: f32, y: f32, rect: (f32, f32, f32, f32)) -> bool {
    let (rx, ry, rw, rh) = rect;
    x >= rx && x <= rx + rw && y >= ry && y <= ry + rh
}
