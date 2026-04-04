# ── Stage 1: Build ────────────────────────────────────────────────────────────
# Compile Rust → WASM, then run wasm-bindgen to generate JS glue.
# wasm-bindgen-cli must match the `wasm-bindgen` crate version in Cargo.lock (0.2.117).
FROM rust:1.94-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN rustup target add wasm32-unknown-unknown

# Pin CLI to match Cargo.lock — version mismatch panics at bindgen time
RUN cargo install wasm-bindgen-cli --version 0.2.117 --locked

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --target wasm32-unknown-unknown --release --locked

COPY static ./static

RUN wasm-bindgen \
    target/wasm32-unknown-unknown/release/game.wasm \
    --out-dir static/pkg \
    --target web


# ── Stage 2: Serve ────────────────────────────────────────────────────────────
FROM nginx:alpine

COPY --from=builder /app/static /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
