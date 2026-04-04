use wasm_bindgen::JsValue;
use web_sys::CanvasRenderingContext2d;
use crate::renderer::{CANVAS_W, CANVAS_H};

// Button rect: x, y, w, h
pub const NEW_GAME_BTN: (f32, f32, f32, f32) = (300.0, 210.0, 200.0, 50.0);

fn sf(ctx: &CanvasRenderingContext2d, color: &str) {
    ctx.set_fill_style(&JsValue::from_str(color));
}

fn ss(ctx: &CanvasRenderingContext2d, color: &str) {
    ctx.set_stroke_style(&JsValue::from_str(color));
}

pub fn render(ctx: &CanvasRenderingContext2d) {
    let w = CANVAS_W as f64;
    let h = CANVAS_H as f64;

    // Background
    sf(ctx, "#0d0d1a");
    ctx.fill_rect(0.0, 0.0, w, h);

    // Center highlight
    sf(ctx, "#1a1420");
    ctx.fill_rect(w / 4.0, h / 6.0, w / 2.0, h * 2.0 / 3.0);

    // Top divider
    ss(ctx, "#5a3a15");
    ctx.set_line_width(1.0);
    let div_x = (w - 400.0) / 2.0;
    ctx.begin_path();
    ctx.move_to(div_x, 110.0);
    ctx.line_to(div_x + 400.0, 110.0);
    ctx.stroke();

    // Title
    sf(ctx, "#d4af37");
    ctx.set_font("bold 44px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("MEDIEVAL RPG", div_x + 14.0, 160.0);

    // Subtitle
    sf(ctx, "#5a3a15");
    ctx.set_font("12px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("built with Rust + WebAssembly  \u{00B7}  zero npm", div_x + 28.0, 185.0);

    // Bottom divider
    ss(ctx, "#5a3a15");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(div_x, 198.0);
    ctx.line_to(div_x + 400.0, 198.0);
    ctx.stroke();

    // NEW GAME button
    let (bx, by, bw, bh) = NEW_GAME_BTN;
    let (bx, by, bw, bh) = (bx as f64, by as f64, bw as f64, bh as f64);
    sf(ctx, "rgba(0,0,0,0)");
    ctx.fill_rect(bx, by, bw, bh);
    ss(ctx, "#d4af37");
    ctx.set_line_width(2.0);
    ctx.stroke_rect(bx, by, bw, bh);
    sf(ctx, "#d4af37");
    ctx.set_font("bold 16px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("\u{2694} NEW GAME", bx + 36.0, by + 31.0);

    // Flavor text
    sf(ctx, "#3a3a5a");
    ctx.set_font("10px 'VT323', 'Courier New', monospace");
    let _ = ctx.fill_text("\u{2620} ENTER IF YOU DARE \u{2620}", w / 2.0 - 82.0, h - 30.0);
}

pub fn hit_test_new_game(x: f32, y: f32) -> bool {
    let (bx, by, bw, bh) = NEW_GAME_BTN;
    x >= bx && x <= bx + bw && y >= by && y <= by + bh
}
