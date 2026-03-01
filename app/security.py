import os
import time
import threading
from collections import deque
from typing import Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def _load_token_map() -> Dict[str, str]:
    raw = os.environ.get("AUTH_TOKENS", "").strip()
    if not raw:
        # Local/dev fallback only.
        return {"dev-token": "dev_user"}

    token_map: Dict[str, str] = {}
    for pair in raw.split(","):
        if not pair.strip() or ":" not in pair:
            continue
        token, user_id = pair.split(":", 1)
        token = token.strip()
        user_id = user_id.strip()
        if token and user_id:
            token_map[token] = user_id
    return token_map


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )

    token_map = _load_token_map()
    user_id = token_map.get(credentials.credentials)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )
    return user_id


_RATE_LOCK = threading.Lock()
_RATE_BUCKETS: Dict[str, deque] = {}


def enforce_rate_limit(user_id: str, route_key: str, limit: int = 60, window_s: int = 60):
    now = time.time()
    bucket_key = f"{user_id}:{route_key}"
    window_start = now - window_s

    with _RATE_LOCK:
        q = _RATE_BUCKETS.get(bucket_key)
        if q is None:
            q = deque()
            _RATE_BUCKETS[bucket_key] = q

        while q and q[0] < window_start:
            q.popleft()

        if len(q) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {route_key}. Try again later.",
            )

        q.append(now)
