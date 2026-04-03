use wasm_bindgen::JsValue;
use crate::game::GameState;

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
const C_PLAYER: &str = "#c8b89a";
const C_PLAYER_SWORD: &str = "#aaaacc";
const C_ENEMY: &str = "#8888aa";
const C_ENEMY_SKULL: &str = "#ccccee";
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
    let ctx = &gs.ctx;

    ctx.save();

    // Apply screen shake offset
    ctx.translate(gs.screen_shake.offset_x as f64, gs.screen_shake.offset_y as f64).unwrap();

    // Background
    set_fill(ctx, C_BG);
    fill_rect(ctx, 0.0, 0.0, CANVAS_W as f64, HUD_Y as f64);

    // Draw dungeon room
    draw_dungeon(ctx, gs);

    // Draw enemies (behind player)
    for enemy in &gs.enemies {
        if enemy.alive {
            draw_enemy(ctx, enemy);
        }
    }

    // Draw player
    draw_player(ctx, &gs.player);

    // Draw particles
    draw_particles(ctx, &gs.particles);

    // Draw damage numbers
    draw_damage_numbers(ctx, &gs.damage_numbers);

    ctx.restore();

    // HUD (not affected by screen shake)
    draw_hud(ctx, gs);

    // WASM overlay
    draw_overlay(ctx, gs);

    // Game over screen
    if gs.game_over {
        draw_game_over(ctx);
    }
}

fn draw_dungeon(ctx: &web_sys::CanvasRenderingContext2d, gs: &GameState) {
    let room = &gs.dungeon.room;

    // Outer wall
    set_fill(ctx, C_WALL);
    fill_rect(ctx, room.x as f64, room.y as f64, room.w as f64, room.h as f64);

    // Wall edge highlight
    set_stroke(ctx, C_WALL_EDGE);
    ctx.set_line_width(2.0);
    stroke_rect(ctx, room.x as f64, room.y as f64, room.w as f64, room.h as f64);

    // Floor
    let fx = room.floor_x() as f64;
    let fy = room.floor_y() as f64;
    let fw = room.floor_w() as f64;
    let fh = room.floor_h() as f64;

    set_fill(ctx, C_FLOOR);
    fill_rect(ctx, fx, fy, fw, fh);

    // Floor grid lines
    set_stroke(ctx, C_FLOOR_GRID);
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
}

fn draw_enemy(ctx: &web_sys::CanvasRenderingContext2d, enemy: &crate::entities::Enemy) {
    let x = enemy.x as f64;
    let y = enemy.y as f64;

    // Body
    set_fill(ctx, C_ENEMY);
    fill_rect(ctx, x - 10.0, y - 10.0, 20.0, 24.0);

    // Skull head (circle)
    set_fill(ctx, C_ENEMY_SKULL);
    ctx.begin_path();
    ctx.arc(x, y - 18.0, 10.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();

    // Eye sockets
    set_fill(ctx, C_ENEMY);
    ctx.begin_path();
    ctx.arc(x - 3.5, y - 20.0, 2.5, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    ctx.begin_path();
    ctx.arc(x + 3.5, y - 20.0, 2.5, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();

    // Aggro indicator
    if enemy.state != crate::entities::EnemyState::Patrolling {
        set_fill(ctx, C_AGGRO);
        ctx.begin_path();
        ctx.arc(x, y - 34.0, 4.0, 0.0, std::f64::consts::TAU).unwrap();
        ctx.fill();
    }

    // HP bar
    let bar_w = 30.0_f64;
    let bar_h = 4.0_f64;
    let bar_x = x - 15.0;
    let bar_y = y + 18.0;
    set_fill(ctx, C_HP_BAR_BG);
    fill_rect(ctx, bar_x, bar_y, bar_w, bar_h);
    let hp_frac = (enemy.hp / enemy.max_hp) as f64;
    set_fill(ctx, C_HP_BAR);
    fill_rect(ctx, bar_x, bar_y, bar_w * hp_frac, bar_h);
}

fn draw_player(ctx: &web_sys::CanvasRenderingContext2d, player: &crate::entities::Player) {
    let x = player.x as f64;
    let y = player.y as f64;

    // Body
    set_fill(ctx, C_PLAYER);
    fill_rect(ctx, x - 10.0, y - 10.0, 20.0, 24.0);

    // Head
    ctx.begin_path();
    ctx.arc(x, y - 18.0, 9.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();

    // Sword (diagonal line to right)
    set_stroke(ctx, C_PLAYER_SWORD);
    ctx.set_line_width(2.5);
    ctx.begin_path();
    ctx.move_to(x + 10.0, y - 5.0);
    ctx.line_to(x + 26.0, y - 21.0);
    ctx.stroke();
    // Crossguard
    ctx.begin_path();
    ctx.move_to(x + 14.0, y - 1.0);
    ctx.line_to(x + 22.0, y - 9.0);
    ctx.stroke();

    // HP bar
    let bar_w = 30.0_f64;
    let bar_h = 4.0_f64;
    let bar_x = x - 15.0;
    let bar_y = y + 18.0;
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
        ctx.set_font("bold 16px monospace");
        ctx.fill_text(&format!("-{}", dn.value), dn.x as f64 - 10.0, dn.y as f64).unwrap();
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

    // HP Orb (left)
    draw_orb(ctx, 80.0, 450.0, 38.0, C_HP_ORB, gs.player.hp / gs.player.max_hp);

    // HP label
    set_fill(ctx, "#cc4444");
    ctx.set_font("11px monospace");
    ctx.fill_text("HP", 68.0, 497.0).unwrap();

    // Mana Orb (right) - static 80%
    draw_orb(ctx, 720.0, 450.0, 38.0, C_MANA_ORB, 0.8);

    // Mana label
    set_fill(ctx, "#4466cc");
    ctx.set_font("11px monospace");
    ctx.fill_text("MP", 708.0, 497.0).unwrap();

    // Ability slots (center)
    let slot_y = 440.0_f64;
    let slot_w = 40.0_f64;
    let slot_gap = 8.0_f64;
    let total = 4.0 * slot_w + 3.0 * slot_gap;
    let start_x = (CANVAS_W as f64 - total) / 2.0;
    for i in 0..4 {
        let sx = start_x + i as f64 * (slot_w + slot_gap);
        set_stroke(ctx, C_HUD_BORDER);
        ctx.set_line_width(2.0);
        set_fill(ctx, "#111122");
        fill_rect(ctx, sx, slot_y, slot_w, 40.0);
        stroke_rect(ctx, sx, slot_y, slot_w, 40.0);
        if i == 0 {
            // Attack slot indicator
            set_fill(ctx, "#cc8844");
            ctx.set_font("18px monospace");
            ctx.fill_text("⚔", sx + 11.0, slot_y + 26.0).unwrap();
        }
    }

    // Gold
    set_fill(ctx, C_GOLD);
    ctx.set_font("bold 14px monospace");
    ctx.fill_text(&format!("⚜ {}", gs.gold), 370.0, 430.0).unwrap();
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

    // Clip to circle
    ctx.begin_path();
    ctx.arc(cx, cy, r, 0.0, std::f64::consts::TAU).unwrap();
    ctx.clip();

    // Background
    set_fill(ctx, "#111122");
    fill_rect(ctx, cx - r, cy - r, r * 2.0, r * 2.0);

    // Fill level
    let fill_h = r * 2.0 * fraction as f64;
    let fill_y = cy + r - fill_h;
    set_fill(ctx, color);
    fill_rect(ctx, cx - r, fill_y, r * 2.0, fill_h);

    ctx.restore();

    // Border ring
    set_stroke(ctx, C_HUD_BORDER);
    ctx.set_line_width(3.0);
    ctx.begin_path();
    ctx.arc(cx, cy, r, 0.0, std::f64::consts::TAU).unwrap();
    ctx.stroke();
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
    ctx.set_font("bold 11px monospace");
    ctx.fill_text("◆ WASM RUNTIME", ox + 8.0, oy + 18.0).unwrap();

    ctx.set_font("11px monospace");
    ctx.fill_text(&format!("{:.1}ms/frame", gs.last_frame_ms), ox + 8.0, oy + 34.0).unwrap();
    ctx.fill_text(
        &format!("{} entities", gs.enemies.iter().filter(|e| e.alive).count() + 1),
        ox + 8.0,
        oy + 50.0,
    ).unwrap();
    ctx.fill_text(
        &format!("{}KB WASM mem", crate::overlay::get_memory_kb()),
        ox + 8.0,
        oy + 66.0,
    ).unwrap();

    set_fill(ctx, "#88cc88");
    ctx.set_font("10px monospace");
    ctx.fill_text("All state in Rust/WASM", ox + 8.0, oy + 88.0).unwrap();
}

fn draw_game_over(ctx: &web_sys::CanvasRenderingContext2d) {
    set_fill(ctx, "rgba(0,0,0,0.65)");
    fill_rect(ctx, 0.0, 0.0, CANVAS_W as f64, CANVAS_H as f64);

    set_fill(ctx, "#cc2222");
    ctx.set_font("bold 48px monospace");
    ctx.fill_text("YOU DIED", 260.0, 220.0).unwrap();

    set_fill(ctx, "#888888");
    ctx.set_font("18px monospace");
    ctx.fill_text("Refresh to play again", 280.0, 265.0).unwrap();
}
