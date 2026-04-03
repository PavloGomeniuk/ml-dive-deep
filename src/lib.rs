use wasm_bindgen::prelude::*;

mod combat;
mod dungeon;
mod entities;
mod game;
mod overlay;
mod particles;
mod renderer;

use game::GameState;

#[wasm_bindgen]
pub struct Game {
    state: GameState,
}

#[wasm_bindgen]
impl Game {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Game, JsValue> {
        console_error_panic_hook::set_once();
        let state = GameState::new()?;
        Ok(Game { state })
    }

    pub fn tick(&mut self, dt: f32) {
        self.state.tick(dt);
    }

    pub fn on_click(&mut self, x: f32, y: f32) {
        self.state.on_click(x, y);
    }

    pub fn get_frame_ms(&self) -> f32 {
        self.state.last_frame_ms
    }

    pub fn get_entity_count(&self) -> u32 {
        self.state.enemies.iter().filter(|e| e.alive).count() as u32 + 1
    }

    pub fn get_wasm_memory_kb(&self) -> u32 {
        overlay::get_memory_kb()
    }
}
