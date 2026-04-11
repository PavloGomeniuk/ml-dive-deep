# ─────────────────────────────────────────────
# Stage 1: Build WASM client
# ─────────────────────────────────────────────
FROM rust:1.94-slim AS wasm-builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Add wasm target
RUN rustup target add wasm32-unknown-unknown

# Install wasm-pack (binary release — faster than cargo install)
RUN curl -sSL https://github.com/rustwasm/wasm-pack/releases/download/v0.13.1/wasm-pack-v0.13.1-x86_64-unknown-linux-musl.tar.gz \
    | tar -xz --strip-components=1 -C /usr/local/bin wasm-pack-v0.13.1-x86_64-unknown-linux-musl/wasm-pack

WORKDIR /app

# Copy workspace files needed for the client build
COPY Cargo.toml Cargo.lock ./
COPY shared/ shared/
COPY client/ client/
# Server Cargo.toml needed for workspace resolution
COPY server/Cargo.toml server/Cargo.toml
RUN mkdir -p server/src && echo 'fn main() {}' > server/src/main.rs
# Build WASM — output lands in server/static/pkg/
RUN wasm-pack build client --target web --out-dir ../server/static/pkg

# ─────────────────────────────────────────────
# Stage 2: Build server binary
# ─────────────────────────────────────────────
FROM rust:1.94-slim AS server-builder

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY shared/ shared/
COPY server/ server/
COPY client/Cargo.toml client/Cargo.toml
# Stub client src so workspace resolves without wasm target
RUN mkdir -p client/src && echo '#[allow(dead_code)] fn main() {}' > client/src/lib.rs

# Copy built WASM from stage 1
COPY --from=wasm-builder /app/server/static/ server/static/

# Build release server binary
RUN cargo build --release -p server

# ─────────────────────────────────────────────
# Stage 3: Minimal runtime image
# ─────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=server-builder /app/target/release/server ./server

# Static files are embedded in the binary via rust-embed — nothing else needed
EXPOSE 3000

ENV RUST_LOG=info

CMD ["./server"]
