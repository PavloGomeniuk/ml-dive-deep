use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CloseEvent, ErrorEvent, MessageEvent, WebSocket};

/// Open a WebSocket connection and register callbacks.
/// `on_message` is called with each raw JSON string received from the server.
pub fn connect(on_message: impl Fn(String) + 'static) -> Result<WebSocket, JsValue> {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let host = location.host().unwrap();
    let protocol = if location.protocol().unwrap() == "https:" { "wss" } else { "ws" };
    let url = format!("{}://{}/ws", protocol, host);

    let ws = WebSocket::new(&url)?;

    // onmessage
    let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
        if let Ok(text) = e.data().dyn_into::<js_sys::JsString>() {
            on_message(String::from(text));
        }
    });
    ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();

    // onerror
    let onerror = Closure::<dyn FnMut(ErrorEvent)>::new(|_e: ErrorEvent| {
        web_sys::console::error_1(&wasm_bindgen::JsValue::from_str("WebSocket error"));
    });
    ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
    onerror.forget();

    // onclose
    let onclose = Closure::<dyn FnMut(CloseEvent)>::new(|_e: CloseEvent| {
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str("WebSocket closed"));
    });
    ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
    onclose.forget();

    Ok(ws)
}

pub fn send(ws: &WebSocket, msg: &shared::messages::ClientMessage) {
    if let Ok(json) = serde_json::to_string(msg) {
        let _ = ws.send_with_str(&json);
    }
}
