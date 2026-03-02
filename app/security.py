import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import Dict, Optional

from fastapi import HTTPException, Request, status


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_user_tokens() -> Dict[str, str]:
    raw_json = os.getenv("USER_TOKENS_JSON", "").strip()
    raw_csv = os.getenv("USER_TOKENS", "").strip()

    pairs: Dict[str, str] = {}
    if raw_json:
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            raise RuntimeError("USER_TOKENS_JSON must be a JSON object of {user_id: token}.")
        for user_id, token in data.items():
            u = str(user_id).strip()
            t = str(token).strip()
            if u and t:
                pairs[u] = t
    elif raw_csv:
        for chunk in raw_csv.split(","):
            piece = chunk.strip()
            if not piece:
                continue
            if ":" not in piece:
                raise RuntimeError("USER_TOKENS must use user_id:token pairs separated by commas.")
            user_id, token = piece.split(":", 1)
            u = user_id.strip()
            t = token.strip()
            if u and t:
                pairs[u] = t

    return pairs


AUTH_REQUIRED = _env_bool("AUTH_REQUIRED", False)
RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

TOKENS_BY_USER = _parse_user_tokens()
USERS_BY_TOKEN = {token: user_id for user_id, token in TOKENS_BY_USER.items()}

_RATE_EVENTS = defaultdict(deque)
_RATE_LOCK = threading.Lock()


def validate_security_configuration() -> None:
    if AUTH_REQUIRED and not USERS_BY_TOKEN:
        raise RuntimeError(
            "AUTH_REQUIRED=true but no API tokens are configured. "
            "Set USER_TOKENS or USER_TOKENS_JSON."
        )
    if RATE_LIMIT_PER_MIN < 1:
        raise RuntimeError("RATE_LIMIT_PER_MIN must be >= 1.")


def reset_rate_limiter() -> None:
    with _RATE_LOCK:
        _RATE_EVENTS.clear()


def resolve_user_id(request: Request, claimed_user_id: Optional[str]) -> str:
    if not AUTH_REQUIRED:
        if not claimed_user_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="user_id is required when authentication is disabled.",
            )
        return claimed_user_id

    auth_header = request.headers.get("Authorization", "")
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header.",
        )

    token = parts[1].strip()
    user_id = USERS_BY_TOKEN.get(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token.",
        )

    if claimed_user_id and claimed_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated token cannot access another user_id.",
        )

    return user_id


def enforce_rate_limit(user_id: str, scope: str) -> None:
    if not RATE_LIMIT_ENABLED:
        return

    now = time.monotonic()
    window_sec = 60.0
    key = f"{user_id}:{scope}"

    with _RATE_LOCK:
        events = _RATE_EVENTS[key]
        while events and (now - events[0]) > window_sec:
            events.popleft()

        if len(events) >= RATE_LIMIT_PER_MIN:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {scope}. Try again soon.",
            )

        events.append(now)
