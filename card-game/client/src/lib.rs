use wasm_bindgen::prelude::*;

mod canvas;
mod ui;
mod net;

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    ui::init()
}
