"""Who is asking, from the assertion Google's Identity-Aware Proxy signs.

A hosted Banko sits behind IAP, which has already signed the caller in with
their Google account and says so in the x-goog-iap-jwt-assertion header: a
JWT signed with ES256 by a key Google publishes. That header is the only
source of identity here. x-goog-authenticated-user-email says the same in
plain text and is never read: anything that reaches the app without passing
through IAP can put any address in it.

An assertion is believed when its signature checks against Google's keys and
it was issued by IAP, for this service (the audience), and has not expired.
Anything else, a missing header and Google being out of reach included,
means nobody is signed in. It is never an error page, and never a guess.

It names the person twice: "sub", which Google calls the unique, stable
identifier for the user, and "email". The account is keyed on the first. An
address can be handed to someone else; the identifier cannot, so nobody
inherits an account with an address.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable
from typing import Any, NamedTuple

from google.auth import jwt

log = logging.getLogger("banko.web")

ASSERTION_HEADER = "x-goog-iap-jwt-assertion"
ISSUER = "https://cloud.google.com/iap"
KEYS_URL = "https://www.gstatic.com/iap/verify/public_key"
KEYS_TTL_SECONDS = 6 * 3600          # Google rotates these rarely; an unknown key id asks again sooner
UNKNOWN_KEY_REFRESH_SECONDS = 60     # anyone can put any key id in a header: one more look a minute at most
RETRY_SECONDS = 15                   # after a read failed with no keys in hand; with keys in hand, a minute
CLOCK_SKEW_SECONDS = 30              # what Google's own guidance allows between its clock and ours
MAX_LIFETIME_SECONDS = 3600          # IAP signs for ten minutes; nothing it signs is good for longer than this
REMEMBERED = 1024                    # verified assertions kept until they expire, so a page's requests verify once
_ALGORITHM = "ES256"
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# The library checks expiry against the system clock. The checks are made
# here instead, against a clock a test can move, so its own are given all
# the room there is. The signature is still its work.
_LIBRARY_CLOCK_SKEW = 10 * 365 * 24 * 3600


class Identity(NamedTuple):
    subject: str    # Google's stable identifier for the account, as sent ("accounts.google.com:1234...")
    email: str      # lowered


class _Refused(Exception):
    """An assertion turned down here, for a reason worded here. What the
    library says about one it turns down can quote the header back, and
    the header's payload is the person's email, so only these are logged."""


def fetch_keys() -> dict[str, str]:
    """Google's current IAP signing keys: key id to public key (PEM)."""
    import requests
    response = requests.get(KEYS_URL, timeout=(3, 5))   # a request waits on this: seconds to connect, seconds to read
    response.raise_for_status()
    keys: dict[str, str] = response.json()
    return keys


class Verifier:
    def __init__(self, audience: str, *, fetch_keys: Callable[[], dict[str, str]] = fetch_keys,
                 now: Callable[[], float] = time.time, elapsed: Callable[[], float] = time.monotonic) -> None:
        """now is the time of day, which an assertion's expiry is checked
        against. elapsed only runs forward, and times everything about the
        keys and the log: set back, the time of day would stop the keys
        being read for as long as it was set back."""
        self.audience = (audience or "").strip()
        self._fetch, self._now, self._elapsed = fetch_keys, now, elapsed
        self._asking = threading.Lock()         # held by the one request that is reading Google's keys
        self._keys: dict[str, str] = {}         # replaced whole, never changed in place
        self._keys_at = float("-inf")           # when the keys were last read
        self._unknown_at = float("-inf")        # when an unknown key id last asked for them again
        self._failed_at = float("-inf")         # when a read last failed
        self._seen: dict[str, tuple[Identity, float]] = {}   # assertion -> (who, expiry)
        self._said: dict[str, float] = {}       # a reason for turning one down -> when it was last logged

    @property
    def on(self) -> bool:
        return bool(self.audience)

    def identity(self, assertion: str | None) -> Identity | None:
        """Who a valid assertion names, or None."""
        if not self.on or not assertion or not isinstance(assertion, str):
            return None
        now = self._now()
        seen = self._seen.get(assertion)
        if seen is not None:
            if now <= seen[1] + CLOCK_SKEW_SECONDS:
                return seen[0]
            self._seen.pop(assertion, None)
        try:
            claims = self._verified(assertion)
            who, expiry = self._accepted(claims, now)
        except _Refused as exc:
            self._say(str(exc))
            return None
        except Exception as exc:
            self._say(f"it could not be read, or its signature does not check ({exc.__class__.__name__})")
            return None
        if len(self._seen) >= REMEMBERED:
            self._seen.clear()
        self._seen[assertion] = (who, expiry)
        return who

    def _verified(self, assertion: str) -> dict[str, Any]:
        """The claims of an assertion whose signature checks against one of Google's keys."""
        header = jwt.decode_header(assertion)
        if header.get("alg") != _ALGORITHM:
            raise _Refused(f"it is not signed with {_ALGORITHM}")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise _Refused("it names no key id")
        keys = self._keys_for(kid)
        if not keys:
            raise _Refused("Google's signing keys could not be read")
        if kid not in keys:
            raise _Refused("it is signed with a key Google does not publish")
        # The library checks the signature here and nothing else: the audience,
        # the issuer and the times are checked in _accepted, so that the log
        # says which of them was wrong ("made for another service" is the one
        # a new deployment gets, from an IAP_AUDIENCE that does not match).
        claims: dict[str, Any] = jwt.decode(assertion, certs={kid: keys[kid]},
                                            clock_skew_in_seconds=_LIBRARY_CLOCK_SKEW)
        return claims

    def _accepted(self, claims: dict[str, Any], now: float) -> tuple[Identity, float]:
        if claims.get("iss") != ISSUER:
            raise _Refused("it was not issued by IAP")
        if claims.get("aud") != self.audience:
            raise _Refused("it was made for another service (check IAP_AUDIENCE)")
        expiry, issued = claims.get("exp"), claims.get("iat")
        if (not isinstance(expiry, (int, float)) or isinstance(expiry, bool) or not math.isfinite(expiry)
                or not isinstance(issued, (int, float)) or isinstance(issued, bool) or not math.isfinite(issued)):
            raise _Refused("it has no expiry or issue time")
        if expiry - issued > MAX_LIFETIME_SECONDS:
            raise _Refused("it is good for longer than anything IAP signs")
        if now > expiry + CLOCK_SKEW_SECONDS:
            raise _Refused("it has expired")
        if issued > now + CLOCK_SKEW_SECONDS:
            raise _Refused("it was issued in the future (check this machine's clock)")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 255:
            raise _Refused("it names no account")
        email = str(claims.get("email") or "").strip().lower()
        if len(email) > 254 or not _EMAIL.match(email):
            raise _Refused("it carries no email address")
        return Identity(subject.strip(), email), float(expiry)

    def _keys_for(self, kid: str) -> dict[str, str]:
        """Google's keys, read again when they are hours old, or, at most
        once a minute, when the assertion names a key id not among them
        (Google has rotated one in). One request at a time asks Google. The
        others go on with the keys in hand when those can check their
        assertion, and wait for the answer when they cannot. A read that
        failed is not tried again for a while, so the requests behind it are
        answered at once, and keys already held keep working meanwhile."""
        keys = self._keys
        if not self._due(kid, keys):
            return keys
        if not self._asking.acquire(blocking=kid not in keys):
            return keys
        try:
            keys = self._keys
            if not self._due(kid, keys):
                return keys                 # the request ahead of this one asked
            if self._elapsed() - self._keys_at <= KEYS_TTL_SECONDS:
                self._unknown_at = self._elapsed()      # this look is for the unknown key id
            try:
                found = {str(k): str(v) for k, v in dict(self._fetch()).items()}
                if not found:
                    raise ValueError("the answer held no keys")
            except Exception as exc:
                self._failed_at = self._elapsed()
                if not keys:
                    raise _Refused(f"Google's signing keys could not be read ({exc.__class__.__name__})") from None
                log.warning("Google's IAP keys could not be read (%s); the ones already held are still used",
                            exc.__class__.__name__)
                return keys
            self._keys, self._keys_at = found, self._elapsed()
            return found
        finally:
            self._asking.release()

    def _due(self, kid: str, keys: dict[str, str]) -> bool:
        """Whether Google should be asked for its keys now."""
        now = self._elapsed()
        if now - self._failed_at <= (UNKNOWN_KEY_REFRESH_SECONDS if keys else RETRY_SECONDS):
            return False
        if now - self._keys_at > KEYS_TTL_SECONDS:      # also before the first read
            return True
        return kid not in keys and now - self._unknown_at > UNKNOWN_KEY_REFRESH_SECONDS

    def _say(self, reason: str) -> None:
        # Invalid assertions are routine traffic and anyone can send them:
        # each reason is said once a minute, so the header is seen to arrive
        # and why it fails, and junk does not take the line a real problem
        # needs. The reasons are a handful of fixed sentences.
        now = self._elapsed()
        if now - self._said.get(reason, float("-inf")) > 60:
            self._said[reason] = now
            log.info("an IAP assertion was not accepted (%s); that request is not signed in", reason)
