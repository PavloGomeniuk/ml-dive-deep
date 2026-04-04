//! HoMM-scale sprite draw functions. All sprites use Canvas 2D path API.
//! Bounding boxes: Warrior 48×56, Magician 38×64, Skeleton 42×52, Chest 24×20.
//! Physics constants in game.rs were updated to match the larger footprints.

use wasm_bindgen::JsValue;
use web_sys::CanvasRenderingContext2d;

fn set_fill(ctx: &CanvasRenderingContext2d, color: &str) {
    ctx.set_fill_style(&JsValue::from_str(color));
}

fn set_stroke(ctx: &CanvasRenderingContext2d, color: &str) {
    ctx.set_stroke_style(&JsValue::from_str(color));
}

// ─── Warrior ─────────────────────────────────────────────────────────────────
// Bounding box: 48×56px, origin at sprite center (x, y).
pub fn draw_warrior(ctx: &CanvasRenderingContext2d, x: f64, y: f64) {
    // Body — chainmail tabard
    set_fill(ctx, "#c8b89a");
    ctx.fill_rect(x - 16.0, y - 14.0, 32.0, 40.0);

    // Chainmail lines across body
    set_stroke(ctx, "#8a7a6a");
    ctx.set_line_width(1.0);
    let mut line_y = y - 10.0;
    while line_y < y + 26.0 {
        ctx.begin_path();
        ctx.move_to(x - 16.0, line_y);
        ctx.line_to(x + 16.0, line_y);
        ctx.stroke();
        line_y += 5.0;
    }

    // Pauldrons (shoulder plates)
    set_fill(ctx, "#888888");
    ctx.fill_rect(x - 24.0, y - 14.0, 10.0, 10.0); // left
    ctx.fill_rect(x + 14.0, y - 14.0, 10.0, 10.0); // right

    // Head
    set_fill(ctx, "#c8b89a");
    ctx.begin_path();
    ctx.arc(x, y - 26.0, 13.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();

    // Helmet — full cap with nose guard
    set_fill(ctx, "#888888");
    ctx.fill_rect(x - 14.0, y - 39.0, 28.0, 14.0);
    // Nose guard
    ctx.fill_rect(x - 2.0, y - 32.0, 4.0, 10.0);
    // Visor line
    set_stroke(ctx, "#666666");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(x - 14.0, y - 32.0);
    ctx.line_to(x + 14.0, y - 32.0);
    ctx.stroke();

    // Shield (left side)
    set_fill(ctx, "#888888");
    ctx.fill_rect(x - 32.0, y - 14.0, 12.0, 22.0);
    // Shield boss
    set_fill(ctx, "#aaaaaa");
    ctx.begin_path();
    ctx.arc(x - 26.0, y - 3.0, 4.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    // Shield border
    set_stroke(ctx, "#666666");
    ctx.set_line_width(1.0);
    ctx.stroke_rect(x - 32.0, y - 14.0, 12.0, 22.0);

    // Sword (right side, diagonal)
    set_stroke(ctx, "#aaaacc");
    ctx.set_line_width(4.0);
    ctx.begin_path();
    ctx.move_to(x + 16.0, y - 4.0);
    ctx.line_to(x + 40.0, y - 28.0);
    ctx.stroke();
    // Crossguard
    ctx.set_line_width(3.0);
    ctx.begin_path();
    ctx.move_to(x + 19.0, y + 2.0);
    ctx.line_to(x + 30.0, y - 9.0);
    ctx.stroke();
    // Pommel
    set_fill(ctx, "#aaaacc");
    ctx.begin_path();
    ctx.arc(x + 14.0, y - 2.0, 3.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    // Sword tip
    set_fill(ctx, "#ccccee");
    ctx.begin_path();
    ctx.move_to(x + 40.0, y - 28.0);
    ctx.line_to(x + 36.0, y - 24.0);
    ctx.line_to(x + 43.0, y - 22.0);
    ctx.close_path();
    ctx.fill();

    // Legs
    set_fill(ctx, "#8a8a8a");
    ctx.fill_rect(x - 14.0, y + 26.0, 10.0, 14.0); // left leg
    ctx.fill_rect(x + 4.0, y + 26.0, 10.0, 14.0);  // right leg

    // Boots
    set_fill(ctx, "#443322");
    ctx.fill_rect(x - 16.0, y + 36.0, 12.0, 8.0);
    ctx.fill_rect(x + 4.0, y + 36.0, 12.0, 8.0);
}

// ─── Magician ────────────────────────────────────────────────────────────────
// Bounding box: 38×64px, origin at sprite center (x, y).
pub fn draw_magician(ctx: &CanvasRenderingContext2d, x: f64, y: f64) {
    // Robe — wider at bottom (trapezoid approximation with two rects)
    set_fill(ctx, "#9988cc");
    ctx.fill_rect(x - 11.0, y - 14.0, 22.0, 30.0); // upper robe
    // Robe hem (wider)
    ctx.begin_path();
    ctx.move_to(x - 14.0, y + 16.0);
    ctx.line_to(x + 14.0, y + 16.0);
    ctx.line_to(x + 16.0, y + 36.0);
    ctx.line_to(x - 16.0, y + 36.0);
    ctx.close_path();
    ctx.fill();

    // Robe trim — gold border at hem
    set_stroke(ctx, "#d4af37");
    ctx.set_line_width(1.5);
    ctx.begin_path();
    ctx.move_to(x - 16.0, y + 36.0);
    ctx.line_to(x + 16.0, y + 36.0);
    ctx.stroke();
    // Robe vertical center line (spell focus)
    ctx.begin_path();
    ctx.move_to(x, y - 14.0);
    ctx.line_to(x, y + 36.0);
    ctx.stroke();

    // Head
    set_fill(ctx, "#c8b89a");
    ctx.begin_path();
    ctx.arc(x, y - 26.0, 11.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();

    // Pointed hat
    set_fill(ctx, "#4444aa");
    ctx.begin_path();
    ctx.move_to(x, y - 56.0);         // tip
    ctx.line_to(x + 16.0, y - 32.0); // right brim
    ctx.line_to(x - 16.0, y - 32.0); // left brim
    ctx.close_path();
    ctx.fill();
    // Hat brim band
    set_fill(ctx, "#333388");
    ctx.fill_rect(x - 17.0, y - 35.0, 34.0, 6.0);
    // Star decoration on hat
    set_fill(ctx, "#d4af37");
    ctx.begin_path();
    ctx.arc(x - 4.0, y - 46.0, 3.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();

    // Staff (left side)
    set_stroke(ctx, "#8888aa");
    ctx.set_line_width(3.0);
    ctx.begin_path();
    ctx.move_to(x - 20.0, y - 44.0);
    ctx.line_to(x - 20.0, y + 36.0);
    ctx.stroke();

    // Staff orb (glowing)
    ctx.set_shadow_color("rgba(136,136,255,0.8)");
    ctx.set_shadow_blur(8.0);
    set_fill(ctx, "#aaaaff");
    ctx.begin_path();
    ctx.arc(x - 20.0, y - 44.0, 6.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    ctx.set_shadow_blur(0.0);
    // Orb ring
    set_stroke(ctx, "#ccccff");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.arc(x - 20.0, y - 44.0, 8.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.stroke();

    // Sleeve cuffs
    set_fill(ctx, "#7766aa");
    ctx.fill_rect(x - 11.0, y + 6.0, 5.0, 8.0);  // left cuff
    ctx.fill_rect(x + 6.0, y + 6.0, 5.0, 8.0);   // right cuff
}

// ─── Skeleton ────────────────────────────────────────────────────────────────
// Bounding box: 42×52px, origin at sprite center (x, y).
pub fn draw_skeleton(ctx: &CanvasRenderingContext2d, x: f64, y: f64) {
    // Body / spine
    set_fill(ctx, "#8888aa");
    ctx.fill_rect(x - 10.0, y - 12.0, 20.0, 28.0);

    // Ribcage lines
    set_stroke(ctx, "#6666888");
    ctx.set_line_width(1.0);
    let mut rib_y = y - 8.0;
    while rib_y < y + 10.0 {
        ctx.begin_path();
        ctx.move_to(x - 10.0, rib_y);
        ctx.line_to(x + 10.0, rib_y);
        ctx.stroke();
        rib_y += 5.0;
    }

    // Skull
    set_fill(ctx, "#ccccee");
    ctx.begin_path();
    ctx.arc(x, y - 26.0, 14.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    // Skull jaw
    set_fill(ctx, "#aaaacc");
    ctx.fill_rect(x - 8.0, y - 18.0, 16.0, 6.0);
    // Jaw gap (teeth)
    set_fill(ctx, "#0d0d1a");
    for i in 0..4i32 {
        ctx.fill_rect(x - 7.0 + i as f64 * 4.0, y - 17.0, 3.0, 4.0);
    }
    // Eye sockets
    set_fill(ctx, "#000000");
    ctx.begin_path();
    ctx.arc(x - 5.0, y - 28.0, 4.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    ctx.begin_path();
    ctx.arc(x + 5.0, y - 28.0, 4.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    // Red glowing pupils
    ctx.set_shadow_color("rgba(255,0,0,0.9)");
    ctx.set_shadow_blur(4.0);
    set_fill(ctx, "#ff2222");
    ctx.begin_path();
    ctx.arc(x - 5.0, y - 28.0, 2.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    ctx.begin_path();
    ctx.arc(x + 5.0, y - 28.0, 2.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    ctx.set_shadow_blur(0.0);

    // Arm bones
    set_stroke(ctx, "#8888aa");
    ctx.set_line_width(3.0);
    // Left arm (holds nothing)
    ctx.begin_path();
    ctx.move_to(x - 10.0, y - 8.0);
    ctx.line_to(x - 20.0, y + 4.0);
    ctx.line_to(x - 18.0, y + 16.0);
    ctx.stroke();
    // Right arm (weapon arm)
    ctx.begin_path();
    ctx.move_to(x + 10.0, y - 8.0);
    ctx.line_to(x + 20.0, y - 2.0);
    ctx.stroke();

    // Leg bones
    ctx.set_line_width(3.0);
    ctx.begin_path();
    ctx.move_to(x - 6.0, y + 16.0);
    ctx.line_to(x - 8.0, y + 36.0);
    ctx.stroke();
    ctx.begin_path();
    ctx.move_to(x + 6.0, y + 16.0);
    ctx.line_to(x + 8.0, y + 36.0);
    ctx.stroke();

    // Rusty scythe (right side, diagonal)
    set_stroke(ctx, "#8a6a4a");
    ctx.set_line_width(2.0);
    ctx.begin_path();
    ctx.move_to(x + 20.0, y - 2.0);
    ctx.line_to(x + 36.0, y - 24.0);
    ctx.stroke();
    // Scythe blade
    set_fill(ctx, "#6a5a3a");
    ctx.begin_path();
    ctx.move_to(x + 36.0, y - 24.0);
    ctx.line_to(x + 22.0, y - 14.0);
    ctx.line_to(x + 28.0, y - 8.0);
    ctx.close_path();
    ctx.fill();
}

// ─── Chest ───────────────────────────────────────────────────────────────────
// Bounding box: 28×22px, origin at center (x, y).
pub fn draw_chest(ctx: &CanvasRenderingContext2d, x: f64, y: f64, opened: bool) {
    // Base
    set_fill(ctx, "#5a3a10");
    ctx.fill_rect(x - 14.0, y - 2.0, 28.0, 14.0);
    // Base border
    set_stroke(ctx, "#8a7a5a");
    ctx.set_line_width(1.0);
    ctx.stroke_rect(x - 14.0, y - 2.0, 28.0, 14.0);
    // Base metal band
    set_stroke(ctx, "#8a7a5a");
    ctx.set_line_width(2.0);
    ctx.begin_path();
    ctx.move_to(x - 14.0, y + 4.0);
    ctx.line_to(x + 14.0, y + 4.0);
    ctx.stroke();

    if opened {
        // Lid swung open (rotated ~45° back)
        set_fill(ctx, "#3a2208");
        ctx.begin_path();
        ctx.move_to(x - 14.0, y - 2.0);
        ctx.line_to(x + 14.0, y - 2.0);
        ctx.line_to(x + 12.0, y - 14.0);
        ctx.line_to(x - 12.0, y - 14.0);
        ctx.close_path();
        ctx.fill();
        // Dark interior visible
        set_fill(ctx, "#0a0a0a");
        ctx.fill_rect(x - 10.0, y - 2.0, 20.0, 6.0);
    } else {
        // Closed lid
        set_fill(ctx, "#6a4418");
        ctx.fill_rect(x - 14.0, y - 10.0, 28.0, 10.0);
        set_stroke(ctx, "#8a7a5a");
        ctx.set_line_width(1.0);
        ctx.stroke_rect(x - 14.0, y - 10.0, 28.0, 10.0);
        // Lock
        set_fill(ctx, "#d4af37");
        ctx.fill_rect(x - 3.0, y - 9.0, 6.0, 7.0);
        ctx.begin_path();
        ctx.arc(x, y - 9.0, 3.0, std::f64::consts::PI, 0.0).unwrap();
        ctx.fill();
    }

    // Hinges
    set_fill(ctx, "#8a7a5a");
    ctx.fill_rect(x - 14.0, y - 4.0, 4.0, 5.0);
    ctx.fill_rect(x + 10.0, y - 4.0, 4.0, 5.0);
}

// ─── Merchant ────────────────────────────────────────────────────────────────
// Robed trader NPC, bounding box ~36×60px.
pub fn draw_merchant(ctx: &CanvasRenderingContext2d, x: f64, y: f64) {
    // Robe — brown trading robes
    set_fill(ctx, "#664422");
    ctx.fill_rect(x - 12.0, y - 12.0, 24.0, 30.0);
    // Robe hem
    ctx.begin_path();
    ctx.move_to(x - 14.0, y + 18.0);
    ctx.line_to(x + 14.0, y + 18.0);
    ctx.line_to(x + 16.0, y + 34.0);
    ctx.line_to(x - 16.0, y + 34.0);
    ctx.close_path();
    ctx.fill();
    // Gold trim
    set_stroke(ctx, "#d4af37");
    ctx.set_line_width(2.0);
    ctx.begin_path();
    ctx.move_to(x, y - 12.0);
    ctx.line_to(x, y + 34.0);
    ctx.stroke();

    // Head
    set_fill(ctx, "#c8a880");
    ctx.begin_path();
    ctx.arc(x, y - 24.0, 11.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    // Hood
    set_fill(ctx, "#553311");
    ctx.fill_rect(x - 12.0, y - 35.0, 24.0, 14.0);
    ctx.begin_path();
    ctx.arc(x, y - 35.0, 12.0, std::f64::consts::PI, 0.0).unwrap();
    ctx.fill();

    // Merchant bag/pack (right side)
    set_fill(ctx, "#8a6633");
    ctx.fill_rect(x + 12.0, y - 8.0, 10.0, 16.0);
    set_stroke(ctx, "#664422");
    ctx.set_line_width(1.0);
    ctx.stroke_rect(x + 12.0, y - 8.0, 10.0, 16.0);

    // Gold coin in left hand
    set_fill(ctx, "#d4af37");
    ctx.begin_path();
    ctx.arc(x - 18.0, y + 2.0, 5.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    set_stroke(ctx, "#aa8822");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.arc(x - 18.0, y + 2.0, 5.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.stroke();
}

// ─── Potion (floor item) ─────────────────────────────────────────────────────
pub fn draw_potion(ctx: &CanvasRenderingContext2d, x: f64, y: f64, is_hp: bool) {
    let color = if is_hp { "#cc2244" } else { "#2244cc" };
    let glow = if is_hp { "rgba(204,34,68,0.6)" } else { "rgba(34,68,204,0.6)" };

    // Glow
    ctx.set_shadow_color(glow);
    ctx.set_shadow_blur(6.0);

    // Flask body (rounded rect approximation)
    set_fill(ctx, color);
    ctx.fill_rect(x - 5.0, y - 6.0, 10.0, 12.0);
    // Rounded bottom
    ctx.begin_path();
    ctx.arc(x, y + 6.0, 5.0, 0.0, std::f64::consts::PI).unwrap();
    ctx.fill();
    // Rounded top shoulders
    ctx.begin_path();
    ctx.arc(x - 5.0, y - 6.0, 2.0, std::f64::consts::PI, std::f64::consts::PI * 1.5).unwrap();
    ctx.fill();
    ctx.begin_path();
    ctx.arc(x + 5.0, y - 6.0, 2.0, std::f64::consts::PI * 1.5, 0.0).unwrap();
    ctx.fill();

    ctx.set_shadow_blur(0.0);

    // Flask neck
    set_fill(ctx, "#8a7a5a");
    ctx.fill_rect(x - 3.0, y - 12.0, 6.0, 7.0);
    // Cork
    set_fill(ctx, "#8b4513");
    ctx.fill_rect(x - 3.0, y - 15.0, 6.0, 4.0);

    // Highlight on flask
    set_fill(ctx, "rgba(255,255,255,0.3)");
    ctx.fill_rect(x - 3.0, y - 4.0, 3.0, 6.0);
}

// ─── Floor Item (equipment diamond or potion) ─────────────────────────────────
pub fn draw_floor_item(ctx: &CanvasRenderingContext2d, x: f64, y: f64, kind: crate::entities::ItemKind) {
    use crate::entities::ItemKind;
    match kind {
        ItemKind::HpPotion => draw_potion(ctx, x, y, true),
        ItemKind::MpPotion => draw_potion(ctx, x, y, false),
        _ => {
            let color = match kind {
                ItemKind::Sword => "#aaaacc",
                ItemKind::Staff => "#8866aa",
                ItemKind::Tome => "#aa6622",
                _ => "#888888",
            };
            // Diamond shape (larger than before — 14×14)
            ctx.save();
            ctx.translate(x, y).unwrap();
            let _ = ctx.rotate(std::f64::consts::PI / 4.0);
            set_fill(ctx, color);
            ctx.fill_rect(-7.0, -7.0, 14.0, 14.0);
            // Highlight edge
            set_stroke(ctx, "rgba(255,255,255,0.3)");
            ctx.set_line_width(1.0);
            ctx.stroke_rect(-7.0, -7.0, 14.0, 14.0);
            ctx.restore();
        }
    }
}
