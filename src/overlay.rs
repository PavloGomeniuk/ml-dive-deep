use wasm_bindgen::JsValue;
use wasm_bindgen::JsCast;

pub fn get_memory_kb() -> u32 {
    let mem = wasm_bindgen::memory();
    let mem: js_sys::WebAssembly::Memory = match mem.dyn_into() {
        Ok(m) => m,
        Err(_) => return 0,
    };
    let buffer = mem.buffer();
    let byte_len = js_sys::Reflect::get(&buffer, &JsValue::from_str("byteLength"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as u32;
    byte_len / 1024
}
