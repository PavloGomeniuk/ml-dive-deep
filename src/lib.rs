use bevy::prelude::*;
use bevy::window::WindowResolution;
use wasm_bindgen::prelude::*;
#[allow(unused_imports)]
use js_sys;

mod audio;
mod char_select;
mod combat;
mod components;
mod dead;
mod dungeon;
mod entities;
mod items;
mod playing;
mod resources;
mod state;
mod title;

use state::GameState;
use resources::*;
use playing::{PlayingPlugin, DungeonRes, update_hud};

/// Entry point for WASM. Called automatically by `wasm-pack` / `init()` in JS.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn run() {
    console_error_panic_hook::set_once();

    // Seed RNG from Math.random() on WASM.
    let seed = (js_sys::Math::random() * u32::MAX as f64) as u32;
    let seed = seed.max(1);

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Medieval RPG".to_string(),
                        canvas: Some("#canvas".into()),
                        resolution: WindowResolution::new(800.0, 500.0),
                        resizable: false,
                        fit_canvas_to_parent: false,
                        prevent_default_event_handling: true,
                        ..default()
                    }),
                    ..default()
                }),
        )
        // ── States ───────────────────────────────────────────────────────────
        .init_state::<GameState>()
        .enable_state_scoped_entities::<GameState>()
        // ── Resources ────────────────────────────────────────────────────────
        .insert_resource(Rng::new(seed))
        .insert_resource(Gold::default())
        .insert_resource(ScreenShakeRes::default())
        .insert_resource(HudFlash::default())
        .insert_resource(SelectedClass::default())
        .insert_resource(RunStats::default())
        .insert_resource(CharSelectHighlight::default())
        .insert_resource(PendingTransition::default())
        // ── Plugins ──────────────────────────────────────────────────────────
        .add_plugins(title::TitlePlugin)
        .add_plugins(char_select::CharSelectPlugin)
        .add_plugins(PlayingPlugin)
        .add_plugins(dead::DeadPlugin)
        // ── Camera ───────────────────────────────────────────────────────────
        .add_systems(Startup, spawn_camera)
        // ── HUD live updates ─────────────────────────────────────────────────
        .add_systems(Update, update_hud.run_if(in_state(GameState::Playing)))
        // ── Background colour ────────────────────────────────────────────────
        .insert_resource(ClearColor(Color::srgb(0.039, 0.039, 0.063)))
        .run();
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
