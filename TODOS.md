# TODOS

## P1

### [ ] Startup SECRET_KEY entropy check
**What:** At startup, check `SECRET_KEY` length. If under 32 characters, log a `WARNING` (not a hard error — don't block startup).
**Why:** `SHA-256` doesn't add entropy. A key like `password123` produces a deterministic AES key with near-zero security. Operators don't read READMEs.
**How:** `if len(os.environ.get("SECRET_KEY", "")) < 32: logger.warning("SECRET_KEY is under 32 characters — use a high-entropy value: openssl rand -hex 32")`
**Effort:** XS (CC: ~5 min)
**Context:** Added during CEO review of SecretDrop design (2026-03-29).

## P2

*(empty — Slack bot Phase 2 was evaluated and skipped)*
