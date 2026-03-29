import pytest
from cryptography.exceptions import InvalidTag

from app.crypto import derive_key, encrypt, decrypt


KEY = derive_key("test-secret-key-for-testing-only-32chars!!")


def test_encrypt_decrypt_roundtrip():
    plaintext = "super secret API key: sk-abc123"
    ciphertext, nonce = encrypt(plaintext, KEY)
    result = decrypt(ciphertext, nonce, KEY)
    assert result == plaintext


def test_different_nonces_each_time():
    plaintext = "same secret"
    _, nonce1 = encrypt(plaintext, KEY)
    _, nonce2 = encrypt(plaintext, KEY)
    assert nonce1 != nonce2


def test_wrong_key_raises_invalid_tag():
    plaintext = "secret"
    ciphertext, nonce = encrypt(plaintext, KEY)
    wrong_key = derive_key("completely-different-key-32chars!!!")
    with pytest.raises(InvalidTag):
        decrypt(ciphertext, nonce, wrong_key)


def test_tampered_ciphertext_raises_invalid_tag():
    plaintext = "secret"
    ciphertext, nonce = encrypt(plaintext, KEY)
    import base64
    raw = bytearray(base64.b64decode(ciphertext))
    raw[0] ^= 0xFF
    bad_ciphertext = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(InvalidTag):
        decrypt(bad_ciphertext, nonce, KEY)
