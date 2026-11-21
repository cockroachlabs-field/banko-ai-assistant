"""Signed assertions for tests, the way Google's Identity-Aware Proxy signs
them: ES256, a key id in the header, and the claims IAP sends. The keys are
made here, so no test talks to Google."""

from __future__ import annotations

import time
import zlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from google.auth import crypt, jwt

AUDIENCE = "/projects/123456789/locations/us-east1/services/banko-demo"
ISSUER = "https://cloud.google.com/iap"


class Key:
    """One signing key with its id, as IAP publishes it: the public half as PEM."""

    def __init__(self, kid: str):
        self.kid = kid
        private = ec.generate_private_key(ec.SECP256R1())
        self._pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()).decode()
        self.public = private.public_key().public_bytes(serialization.Encoding.PEM,
                                                        serialization.PublicFormat.SubjectPublicKeyInfo).decode()

    def sign(self, claims: dict, *, kid: str | None = None) -> str:
        signer = crypt.ES256Signer.from_string(self._pem, kid or self.kid)
        return jwt.encode(signer, claims).decode()


def subject(email: str) -> str:
    """The Google account behind an address in these tests: one per address,
    in the form IAP sends."""
    return f"accounts.google.com:{zlib.crc32(email.strip().lower().encode())}"


def claims(email: str = "ana.b@example.com", **changed) -> dict:
    now = int(time.time())
    found = {"iss": ISSUER, "aud": AUDIENCE, "email": email, "sub": subject(email),
             "hd": email.split("@", 1)[-1], "iat": now - 5, "exp": now + 600}
    found.update(changed)
    return {k: v for k, v in found.items() if v is not None}
