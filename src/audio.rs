//! JS Web Audio bridge. All sounds are synthesized in JS — no audio files needed.
//! Called from Bevy systems via wasm-bindgen extern.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = window, js_name = __playHitSound)]
    pub fn play_hit();

    #[wasm_bindgen(js_namespace = window, js_name = __playDeathSound)]
    pub fn play_death();

    #[wasm_bindgen(js_namespace = window, js_name = __playEquipSound)]
    pub fn play_equip();

    #[wasm_bindgen(js_namespace = window, js_name = __playTransitionSound)]
    pub fn play_transition();
}
