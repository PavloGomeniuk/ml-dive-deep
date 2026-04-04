use wasm_bindgen::JsValue;
use crate::game::{GameState, GamePhase};
use crate::entities::PlayerClass;
use crate::items::Item;
use crate::entities::ItemKind;

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
    match gs.phase {
        GamePhase::Title => {
            crate::title::render(&gs.ctx);
        }
        GamePhase::CharacterSelect => {
            crate::character_select::render(&gs.ctx, gs.char_selected);
        }
        GamePhase::Playing => {
            draw_playing(gs);
        }
        GamePhase::Dead => {
            draw_playing(gs);
            draw_game_over(&gs.ctx);
        }
    }
}

fn draw_playing(gs: &GameState) {
    let ctx = &gs.ctx;

    ctx.save();
    ctx.translate(gs.screen_shake.offset_x as f64, gs.screen_shake.offset_y as f64).unwrap();

    // Background
    set_fill(ctx, C_BG);
    fill_rect(ctx, 0.0, 0.0, CANVAS_W as f64, HUD_Y as f64);

    draw_dungeon(ctx, gs);

    // Items on floor (behind characters)
    for item in &gs.items_on_floor {
        draw_item(ctx, item);
    }

    // Enemies
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

    // Floor (different color per room)
    let (floor_col, grid_col) = if room_idx == 0 {
        (C_FLOOR, C_FLOOR_GRID)
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

    // Room label (top-left of floor for room 1)
    if room_idx == 1 {
        set_fill(ctx, "#3a4a6a");
        ctx.set_font("10px 'Courier New', monospace");
        let _ = ctx.fill_text("CATACOMBS", fx + 4.0, fy + 14.0);
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

    // Fill door frame area
    set_fill(ctx, "#3a2a0a");
    // Left pillar
    ctx.fill_rect(door_x, door_y, pillar_w, door_h);
    // Right pillar
    ctx.fill_rect(door_x + door_w - pillar_w, door_y, pillar_w, door_h);
    // Inner arch area (slightly lighter)
    set_fill(ctx, "#1a1000");
    ctx.fill_rect(door_x + pillar_w, door_y + 10.0, door_w - pillar_w * 2.0, door_h - 10.0);

    // Arch top (semicircle)
    set_fill(ctx, "#1a1000");
    ctx.begin_path();
    let _ = ctx.arc(dcx, door_y + 10.0, (door_w - pillar_w * 2.0) / 2.0, std::f64::consts::PI, 0.0);
    ctx.fill();

    // Door border
    set_stroke(ctx, "#5a3a15");
    ctx.set_line_width(2.0);
    ctx.stroke_rect(door_x, door_y, door_w, door_h);

    // Gold frame line
    set_stroke(ctx, "#d4af37");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(door_x + pillar_w, door_y + door_h);
    ctx.line_to(door_x + pillar_w, door_y + 10.0);
    let _ = ctx.arc(dcx, door_y + 10.0, (door_w - pillar_w * 2.0) / 2.0, std::f64::consts::PI, 0.0);
    ctx.line_to(door_x + door_w - pillar_w, door_y + door_h);
    ctx.stroke();

    // Label
    set_fill(ctx, "#d4af37");
    ctx.set_font("9px 'Courier New', monospace");
    let label = if room.index == 0 { "LOC 2 \u{25BC}" } else { "LOC 1 \u{25B2}" };
    let _ = ctx.fill_text(label, dcx - 20.0, door_y - 4.0);
}

fn draw_item(ctx: &web_sys::CanvasRenderingContext2d, item: &Item) {
    let x = item.x as f64;
    let y = item.y as f64;

    let color = match item.kind {
        ItemKind::Sword => "#aaaacc",
        ItemKind::Staff => "#8866aa",
        ItemKind::Tome => "#aa6622",
    };

    // Diamond shape (rotated 45° square)
    ctx.save();
    ctx.translate(x, y).unwrap();
    let _ = ctx.rotate(std::f64::consts::PI / 4.0);
    set_fill(ctx, color);
    ctx.fill_rect(-5.0, -5.0, 10.0, 10.0);
    ctx.restore();

    // Label (fades after label_life > 0)
    if item.label_life > 0 {
        let alpha = (item.label_life as f64 / 180.0).min(1.0);
        let label = format!("[E] {}", item.kind.name());
        set_fill(ctx, &format!("rgba(200,184,154,{:.2})", alpha));
        ctx.set_font("9px 'Courier New', monospace");
        let _ = ctx.fill_text(&label, x - 20.0, y - 12.0);
    }
}

fn draw_enemy(ctx: &web_sys::CanvasRenderingContext2d, enemy: &crate::entities::Enemy) {
    let x = enemy.x as f64;
    let y = enemy.y as f64;

    // Frozen tint
    if enemy.frozen_timer > 0.0 {
        ctx.save();
        ctx.set_global_alpha(0.7);
    }

    set_fill(ctx, C_ENEMY);
    fill_rect(ctx, x - 10.0, y - 10.0, 20.0, 24.0);

    set_fill(ctx, C_ENEMY_SKULL);
    ctx.begin_path();
    ctx.arc(x, y - 18.0, 10.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();

    set_fill(ctx, C_ENEMY);
    ctx.begin_path();
    ctx.arc(x - 3.5, y - 20.0, 2.5, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    ctx.begin_path();
    ctx.arc(x + 3.5, y - 20.0, 2.5, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();

    if enemy.frozen_timer > 0.0 {
        // Ice overlay
        ctx.restore();
        set_fill(ctx, "rgba(100,150,255,0.3)");
        fill_rect(ctx, x - 10.0, y - 28.0, 20.0, 38.0);
    }

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

    match player.class {
        PlayerClass::Warrior => draw_warrior(ctx, x, y),
        PlayerClass::Magician => draw_magician(ctx, x, y),
    }

    // HP bar (same for both)
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

fn draw_warrior(ctx: &web_sys::CanvasRenderingContext2d, x: f64, y: f64) {
    // Body: 22x28
    set_fill(ctx, C_PLAYER);
    fill_rect(ctx, x - 11.0, y - 10.0, 22.0, 28.0);

    // Head
    ctx.begin_path();
    ctx.arc(x, y - 18.0, 9.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();

    // Helmet
    set_fill(ctx, "#888888");
    fill_rect(ctx, x - 11.0, y - 27.0, 22.0, 6.0);

    // Shield (left)
    fill_rect(ctx, x - 19.0, y - 8.0, 8.0, 10.0);

    // Sword
    set_stroke(ctx, C_PLAYER_SWORD);
    ctx.set_line_width(3.0);
    ctx.begin_path();
    ctx.move_to(x + 11.0, y - 5.0);
    ctx.line_to(x + 28.0, y - 22.0);
    ctx.stroke();
    // Crossguard
    ctx.set_line_width(2.0);
    ctx.begin_path();
    ctx.move_to(x + 14.0, y - 1.0);
    ctx.line_to(x + 22.0, y - 9.0);
    ctx.stroke();
}

fn draw_magician(ctx: &web_sys::CanvasRenderingContext2d, x: f64, y: f64) {
    // Body: 16x26 (narrower, purple robes)
    set_fill(ctx, "#9988cc");
    fill_rect(ctx, x - 8.0, y - 10.0, 16.0, 26.0);

    // Head
    set_fill(ctx, C_PLAYER);
    ctx.begin_path();
    ctx.arc(x, y - 18.0, 8.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();

    // Pointed hat
    set_fill(ctx, "#4444aa");
    ctx.begin_path();
    ctx.move_to(x, y - 38.0);
    ctx.line_to(x + 12.0, y - 24.0);
    ctx.line_to(x - 12.0, y - 24.0);
    ctx.close_path();
    ctx.fill();

    // Staff
    set_stroke(ctx, "#8888aa");
    ctx.set_line_width(2.0);
    ctx.begin_path();
    ctx.move_to(x - 14.0, y - 28.0);
    ctx.line_to(x - 14.0, y + 20.0);
    ctx.stroke();

    // Orb with glow
    ctx.set_shadow_color("rgba(136,136,255,0.8)");
    ctx.set_shadow_blur(6.0);
    set_fill(ctx, "#aaaaff");
    ctx.begin_path();
    ctx.arc(x - 14.0, y - 28.0, 4.0, 0.0, std::f64::consts::TAU).unwrap();
    ctx.fill();
    ctx.set_shadow_blur(0.0);
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
    ctx.set_font("11px monospace");
    let _ = ctx.fill_text("HP", 38.0, 497.0);

    // Mana Orb
    let mana_frac = gs.player.mana / gs.player.max_mana;
    draw_orb(ctx, 750.0, 450.0, 38.0, C_MANA_ORB, mana_frac);
    set_fill(ctx, "#4466cc");
    ctx.set_font("11px monospace");
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

    // Ability icon
    set_fill(ctx, "#d4af37");
    ctx.set_font("22px monospace");
    let icon = match gs.player.class {
        PlayerClass::Warrior => "\u{2694}",
        PlayerClass::Magician => "\u{2744}",
    };
    let _ = ctx.fill_text(icon, ability_x + 11.0, ability_y + 30.0);

    // Cooldown overlay
    if gs.player.ability_cooldown > 0.0 {
        let cd_frac = (gs.player.ability_cooldown / gs.player.class.ability_cooldown_max()) as f64;
        set_fill(ctx, &format!("rgba(0,0,0,{:.2})", cd_frac * 0.8));
        fill_rect(ctx, ability_x, ability_y, ability_size, ability_size);
        set_fill(ctx, "#ffffff");
        ctx.set_font("bold 12px monospace");
        let cd_text = format!("{:.1}", gs.player.ability_cooldown);
        let _ = ctx.fill_text(&cd_text, ability_x + 8.0, ability_y + 26.0);
    }

    // Equipment slots (right side of ability slot)
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
            set_stroke(ctx, "#d4af37");
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
            };
            // Draw mini diamond
            ctx.save();
            ctx.translate(sx + slot_size / 2.0, slot_y + slot_size / 2.0).unwrap();
            let _ = ctx.rotate(std::f64::consts::PI / 4.0);
            set_fill(ctx, item_color);
            fill_rect(ctx, -7.0, -7.0, 14.0, 14.0);
            ctx.restore();

            // "E" label top-right
            set_fill(ctx, "#d4af37");
            ctx.set_font("bold 8px monospace");
            let _ = ctx.fill_text("E", sx + slot_size - 10.0, slot_y + 10.0);
        }
    }

    // Key hints
    set_fill(ctx, "#3a3a5a");
    ctx.set_font("9px monospace");
    let _ = ctx.fill_text("[ E ] equip  [ Space ] ability", slot_start_x, slot_y + slot_size + 14.0);

    // Gold (center)
    set_fill(ctx, C_GOLD);
    ctx.set_font("bold 14px monospace");
    let _ = ctx.fill_text(&format!("\u{269C} {}", gs.gold), 390.0, 430.0);

    // Room indicator
    set_fill(ctx, "#5a5a7a");
    ctx.set_font("10px monospace");
    let room_label = if gs.dungeon.current_room == 0 {
        "Location 1"
    } else {
        "Catacombs"
    };
    let _ = ctx.fill_text(room_label, 370.0, 445.0);

    // HUD flash text
    if let Some(ref flash) = gs.hud_flash {
        let alpha = flash.life as f64 / 60.0;
        set_fill(ctx, &format!("rgba(255,238,68,{:.2})", alpha));
        ctx.set_font("bold 12px monospace");
        let _ = ctx.fill_text(flash.text, 340.0, 470.0);
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
    let _ = ctx.fill_text("\u{25C6} WASM RUNTIME", ox + 8.0, oy + 18.0);

    ctx.set_font("11px monospace");
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
    ctx.set_font("10px monospace");
    let _ = ctx.fill_text("All state in Rust/WASM", ox + 8.0, oy + 88.0);
}

fn draw_game_over(ctx: &web_sys::CanvasRenderingContext2d) {
    set_fill(ctx, "rgba(0,0,0,0.65)");
    fill_rect(ctx, 0.0, 0.0, CANVAS_W as f64, CANVAS_H as f64);

    set_fill(ctx, "#cc2222");
    ctx.set_font("bold 48px monospace");
    let _ = ctx.fill_text("YOU DIED", 260.0, 220.0);

    set_fill(ctx, "#888888");
    ctx.set_font("18px monospace");
    let _ = ctx.fill_text("Click to return to title", 258.0, 265.0);
}
