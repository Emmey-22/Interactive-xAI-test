import sqlite3
from typing import Optional, Dict, Any
from datetime import datetime

DB_PATH = "feedback.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_preferences (
        user_id TEXT PRIMARY KEY,
        top_k INTEGER NOT NULL DEFAULT 8,
        style TEXT NOT NULL DEFAULT 'simple',
        updated_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        case_id TEXT,
        feature_name TEXT,
        feedback_type TEXT NOT NULL,
        message TEXT,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_activity_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        case_id TEXT,
        event_type TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # Query paths hit user+feedback_type+feature_name and chronological activity.
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_feedback_user_type_feature
    ON feedback_events(user_id, feedback_type, feature_name)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_feedback_user_case_type_feature
    ON feedback_events(user_id, case_id, feedback_type, feature_name)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_feedback_user_created_at
    ON feedback_events(user_id, created_at DESC)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_activity_user_event_created
    ON user_activity_events(user_id, event_type, created_at DESC)
    """)

    conn.commit()
    conn.close()

def ensure_user(user_id: str):
    # Ensure user has a profile row as soon as they start using the app.
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute("""
    INSERT INTO user_preferences(user_id, top_k, style, updated_at)
    VALUES(?, 8, 'simple', ?)
    ON CONFLICT(user_id) DO NOTHING
    """, (user_id, now))
    conn.commit()
    conn.close()

def log_user_activity(user_id: str, event_type: str, case_id: Optional[str] = None):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute("""
    INSERT INTO user_activity_events(user_id, case_id, event_type, created_at)
    VALUES(?, ?, ?, ?)
    """, (user_id, case_id, event_type, now))
    conn.commit()
    conn.close()

def upsert_preferences(user_id: str, top_k: int, style: str):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()

    cur.execute("""
    INSERT INTO user_preferences(user_id, top_k, style, updated_at)
    VALUES(?, ?, ?, ?)
    ON CONFLICT(user_id) DO UPDATE SET
        top_k=excluded.top_k,
        style=excluded.style,
        updated_at=excluded.updated_at
    """, (user_id, int(top_k), str(style), now))

    conn.commit()
    conn.close()

def get_preferences(user_id: str) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM user_preferences WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return {"top_k": 8, "style": "simple"}
    return {"top_k": int(row["top_k"]), "style": row["style"]}

def insert_feedback(user_id: str, feedback_type: str,
                    feature_name: Optional[str], case_id: Optional[str],
                    message: Optional[str]):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()

    cur.execute("""
    INSERT INTO feedback_events(user_id, case_id, feature_name, feedback_type, message, created_at)
    VALUES(?, ?, ?, ?, ?, ?)
    """, (user_id, case_id, feature_name, feedback_type, message, now))

    conn.commit()
    conn.close()

def get_disputed_features(user_id: str, case_id: Optional[str] = None):
    # treat "irrelevant" as disputed for rendering rules
    conn = get_conn()
    cur = conn.cursor()
    if case_id is not None:
        cur.execute("""
        SELECT DISTINCT feature_name
        FROM feedback_events
        WHERE user_id=? AND case_id=? AND feedback_type='irrelevant' AND feature_name IS NOT NULL
        """, (user_id, case_id))
    else:
        cur.execute("""
        SELECT DISTINCT feature_name
        FROM feedback_events
        WHERE user_id=? AND feedback_type='irrelevant' AND feature_name IS NOT NULL
        """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [r["feature_name"] for r in rows]

def get_confusing_features(user_id: str, case_id: Optional[str] = None):
    conn = get_conn()
    cur = conn.cursor()
    if case_id is not None:
        cur.execute("""
        SELECT DISTINCT feature_name
        FROM feedback_events
        WHERE user_id=? AND case_id=? AND feedback_type='confusing' AND feature_name IS NOT NULL
        """, (user_id, case_id))
    else:
        cur.execute("""
        SELECT DISTINCT feature_name
        FROM feedback_events
        WHERE user_id=? AND feedback_type='confusing' AND feature_name IS NOT NULL
        """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [r["feature_name"] for r in rows]

def apply_preference_from_feedback(user_id: str, feedback_type: str):
    prefs = get_preferences(user_id)
    top_k = int(prefs.get("top_k", 8))
    style = prefs.get("style", "simple")

    if feedback_type == "prefer_short":
        top_k = 3
        style = "simple"
    elif feedback_type == "prefer_long":
        top_k = 10
        style = "detailed"

    upsert_preferences(user_id, top_k, style)

def feedback_summary(user_id: str = None):
    conn = get_conn()
    cur = conn.cursor()

    if user_id is not None:
        cur.execute("""
        SELECT feedback_type, COUNT(*) AS n
        FROM feedback_events
        WHERE user_id=?
        GROUP BY feedback_type
        ORDER BY n DESC
        """, (user_id,))
    else:
        cur.execute("""
        SELECT feedback_type, COUNT(*) AS n
        FROM feedback_events
        GROUP BY feedback_type
        ORDER BY n DESC
        """)

    rows = cur.fetchall()
    conn.close()
    return [{"feedback_type": r["feedback_type"], "count": int(r["n"])} for r in rows]

def top_features_by_feedback(feedback_type: str, limit: int = 10, user_id: str = None):
    conn = get_conn()
    cur = conn.cursor()
    safe_limit = max(1, min(100, int(limit)))

    if user_id is not None:
        cur.execute("""
        SELECT feature_name, COUNT(*) AS n
        FROM feedback_events
        WHERE feedback_type=? AND user_id=? AND feature_name IS NOT NULL
        GROUP BY feature_name
        ORDER BY n DESC
        LIMIT ?
        """, (feedback_type, user_id, safe_limit))
    else:
        cur.execute("""
        SELECT feature_name, COUNT(*) AS n
        FROM feedback_events
        WHERE feedback_type=? AND feature_name IS NOT NULL
        GROUP BY feature_name
        ORDER BY n DESC
        LIMIT ?
        """, (feedback_type, safe_limit))

    rows = cur.fetchall()
    conn.close()
    return [{"feature": r["feature_name"], "count": int(r["n"])} for r in rows]
