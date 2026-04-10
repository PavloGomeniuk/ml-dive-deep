mod hub;
mod bot;

use axum::{
    extract::{State, WebSocketUpgrade, Path},
    response::IntoResponse,
    routing::get,
    Router,
};
use rust_embed::RustEmbed;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing_subscriber;

pub type AppStateHandle = Arc<RwLock<hub::AppState>>;

#[derive(RustEmbed)]
#[folder = "static/"]
struct StaticFiles;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state: AppStateHandle = Arc::new(RwLock::new(hub::AppState::new()));

    let app = Router::new()
        .route("/", get(serve_index))
        .route("/pkg/*path", get(serve_static))
        .route("/ws", get(ws_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("bind failed");

    tracing::info!("CARD ARENA running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn serve_index() -> impl IntoResponse {
    serve_file("index.html")
}

async fn serve_static(Path(path): Path<String>) -> impl IntoResponse {
    // rust-embed keys are relative to static/, so pkg/* files have key "pkg/..."
    serve_file(&format!("pkg/{}", path))
}

fn serve_file(path: &str) -> impl IntoResponse {
    use axum::http::{header, StatusCode};
    match StaticFiles::get(path) {
        Some(content) => {
            let mime = mime_for(path);
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, mime)],
                content.data.into_owned(),
            )
                .into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn mime_for(path: &str) -> &'static str {
    if path.ends_with(".wasm") {
        "application/wasm"
    } else if path.ends_with(".js") {
        "application/javascript"
    } else if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else {
        "application/octet-stream"
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppStateHandle>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| hub::handle_connection(socket, state))
}
