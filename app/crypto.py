import base64
import hashlib
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def derive_key(secret_key: str) -> bytes:
    """SHA-256 hash of SECRET_KEY to produce a 32-byte AES key."""
    return hashlib.sha256(secret_key.encode()).digest()


def encrypt(plaintext: str, key: bytes) -> tuple[str, str]:
    """Encrypt plaintext with AES-256-GCM. Returns (ciphertext_b64, nonce_b64)."""
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.b64encode(ciphertext).decode(), base64.b64encode(nonce).decode()


def decrypt(ciphertext_b64: str, nonce_b64: str, key: bytes) -> str:
    """Decrypt AES-256-GCM ciphertext. Raises cryptography.exceptions.InvalidTag on failure."""
    ciphertext = base64.b64decode(ciphertext_b64)
    nonce = base64.b64decode(nonce_b64)
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode()
