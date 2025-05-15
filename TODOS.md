# TODOS — FamilyRoom

## End-to-End Message Encryption (passphrase-based)

**What:** Add client-side encryption to the chat so the server stores ciphertext only.
Family members share a secret passphrase (separate from the invite code). Messages are
encrypted in the browser before sending and decrypted in the browser on receive. The
Python server never sees plaintext message content.

**Why:** The app was built specifically to replace WhatsApp for privacy. Currently,
messages are stored in plain SQLite on the server — whoever has server access can read
everything. E2EE closes this gap: even if the server is compromised, message content
is unreadable without the passphrase.

**How (no npm, browser native):**
- Use the Web Crypto API (`window.crypto.subtle`) — built into all modern browsers,
  zero dependencies
- Algorithm: AES-GCM (256-bit key). Each message gets a random 12-byte IV (initialization
  vector — a random value that ensures the same plaintext encrypts differently each time)
- Key derivation: `PBKDF2(passphrase, salt, 100000 iterations) → AES key`
  - Salt can be a fixed constant per family (stored in config.py, shared with clients)
  - OR derived from the invite code (reuses existing shared secret)
- Flow:
  1. User enters passphrase once per session (stored in `sessionStorage`, not localStorage)
  2. Browser derives AES key from passphrase
  3. On send: `encrypt(message) → {iv: base64, ciphertext: base64}` → send to server
  4. Server stores ciphertext blob (treats it as an opaque string in messages table)
  5. On receive: `decrypt(iv, ciphertext) → plaintext` → render in chat
  6. File upload filenames can also be encrypted; file content itself is not (too heavy)

**Scope of change:**
- `app.js`: add `CryptoKey` derivation on join, encrypt on send, decrypt on receive
- `db.py`: no change — message content column stores ciphertext string instead of plaintext
- `server.py`: no change — server is already content-agnostic
- `index.html`: add passphrase input field on join form (below name + invite code)
- Old messages (pre-encryption): show "[encrypted]" placeholder for messages that can't
  be decrypted (different passphrase or pre-encryption messages)

**Tradeoffs:**
- Pros: Server-side breach doesn't expose message content. Stays true to the privacy goal.
- Cons: If user forgets the passphrase, all history is unreadable. No passphrase recovery.
  File content not encrypted (too large for client-side AES). Message search becomes impossible.

**Depends on:** Core FamilyRoom MVP working first (Steps 1-6 in Next Steps)

**Effort:** Human: ~2 days / CC+gstack: ~45 min
