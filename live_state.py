"""
SQLite-backed persistent state for the live trading bot.

Tracks:
  - intents:    decisions made by the model that turned into order attempts
  - positions:  filled positions (one row per round-trip; closed_at set on exit)
  - events:     append-only audit log

All timestamps are UTC ISO-8601.  Schema is forward-compatible: new columns
should be added via ALTER TABLE in `init_db`.
"""

from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DB_PATH_DEFAULT = Path(__file__).parent / "live_bot_state.db"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db(db_path: Path = DB_PATH_DEFAULT) -> None:
    with sqlite3.connect(db_path) as cx:
        cx.executescript(
            """
            CREATE TABLE IF NOT EXISTS intents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                bar_close_time  TEXT NOT NULL,
                proba_raw       REAL NOT NULL,
                proba_cal       REAL,
                threshold       REAL NOT NULL,
                action          INTEGER NOT NULL,            -- 1 = enter, 0 = skip
                features_json   TEXT NOT NULL,
                outcome         TEXT,                        -- filled / unfilled / skipped / error
                client_order_id TEXT,
                exchange_order_id TEXT,
                note            TEXT,
                UNIQUE(symbol, bar_close_time)
            );

            CREATE TABLE IF NOT EXISTS positions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_id           INTEGER NOT NULL,
                symbol              TEXT NOT NULL,
                side                TEXT NOT NULL,           -- LONG / SHORT
                opened_at           TEXT NOT NULL,
                entry_order_id      TEXT NOT NULL,
                entry_price         REAL NOT NULL,
                quantity            REAL NOT NULL,
                tp_price            REAL NOT NULL,
                sl_price            REAL NOT NULL,
                tp_order_id         TEXT,
                sl_order_id         TEXT,
                max_hold_until      TEXT NOT NULL,
                closed_at           TEXT,
                close_price         REAL,
                close_reason        TEXT,                    -- TP / SL / MAX_HOLD / MANUAL / ERROR
                gross_pnl_pct       REAL,
                FOREIGN KEY(intent_id) REFERENCES intents(id)
            );

            CREATE INDEX IF NOT EXISTS ix_positions_open
                ON positions(symbol, closed_at);

            CREATE TABLE IF NOT EXISTS risk_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                paper_equity REAL NOT NULL,
                peak_equity REAL NOT NULL,
                daily_start_equity REAL NOT NULL,
                last_trade_date TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                level       TEXT NOT NULL,
                kind        TEXT NOT NULL,
                symbol      TEXT,
                payload     TEXT
            );
            """
        )


@contextmanager
def get_conn(db_path: Path = DB_PATH_DEFAULT):
    cx = sqlite3.connect(db_path, isolation_level=None)  # autocommit
    cx.row_factory = sqlite3.Row
    try:
        yield cx
    finally:
        cx.close()


# ─────────────────────────── Events ────────────────────────────────────────


def log_event(
    kind: str,
    level: str = "INFO",
    symbol: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    db_path: Path = DB_PATH_DEFAULT,
) -> None:
    with get_conn(db_path) as cx:
        cx.execute(
            "INSERT INTO events (created_at, level, kind, symbol, payload) VALUES (?,?,?,?,?)",
            (utcnow(), level, kind, symbol, json.dumps(payload) if payload else None),
        )


# ─────────────────────────── Intents ───────────────────────────────────────


def insert_intent(
    symbol: str,
    bar_close_time: str,
    proba_raw: float,
    proba_cal: Optional[float],
    threshold: float,
    action: int,
    features: dict[str, Any],
    db_path: Path = DB_PATH_DEFAULT,
) -> int:
    """Record a model decision.  Returns intent id.

    Idempotent on (symbol, bar_close_time): if the row exists, returns its id
    without overwriting.  Caller should treat duplicate as 'already handled'.
    """
    with get_conn(db_path) as cx:
        cur = cx.execute(
            "SELECT id, action FROM intents WHERE symbol=? AND bar_close_time=?",
            (symbol, bar_close_time),
        )
        row = cur.fetchone()
        if row is not None:
            return int(row["id"])
        cur = cx.execute(
            "INSERT INTO intents (created_at, symbol, bar_close_time, proba_raw, proba_cal, "
            "threshold, action, features_json) VALUES (?,?,?,?,?,?,?,?)",
            (
                utcnow(),
                symbol,
                bar_close_time,
                proba_raw,
                proba_cal,
                threshold,
                action,
                json.dumps(features),
            ),
        )
        return int(cur.lastrowid)


def update_intent_outcome(
    intent_id: int,
    outcome: str,
    client_order_id: Optional[str] = None,
    exchange_order_id: Optional[str] = None,
    note: Optional[str] = None,
    db_path: Path = DB_PATH_DEFAULT,
) -> None:
    with get_conn(db_path) as cx:
        cx.execute(
            "UPDATE intents SET outcome=?, client_order_id=?, exchange_order_id=?, note=? WHERE id=?",
            (outcome, client_order_id, exchange_order_id, note, intent_id),
        )


# ─────────────────────────── Positions ─────────────────────────────────────


def insert_position(
    intent_id: int,
    symbol: str,
    side: str,
    entry_order_id: str,
    entry_price: float,
    quantity: float,
    tp_price: float,
    sl_price: float,
    max_hold_until: str,
    db_path: Path = DB_PATH_DEFAULT,
) -> int:
    with get_conn(db_path) as cx:
        cur = cx.execute(
            "INSERT INTO positions (intent_id, symbol, side, opened_at, entry_order_id, "
            "entry_price, quantity, tp_price, sl_price, max_hold_until) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                intent_id,
                symbol,
                side,
                utcnow(),
                entry_order_id,
                entry_price,
                quantity,
                tp_price,
                sl_price,
                max_hold_until,
            ),
        )
        return int(cur.lastrowid)


def attach_exit_orders(
    position_id: int,
    tp_order_id: Optional[str],
    sl_order_id: Optional[str],
    db_path: Path = DB_PATH_DEFAULT,
) -> None:
    with get_conn(db_path) as cx:
        cx.execute(
            "UPDATE positions SET tp_order_id=?, sl_order_id=? WHERE id=?",
            (tp_order_id, sl_order_id, position_id),
        )


def close_position(
    position_id: int,
    close_price: float,
    close_reason: str,
    gross_pnl_pct: float,
    db_path: Path = DB_PATH_DEFAULT,
) -> None:
    with get_conn(db_path) as cx:
        cx.execute(
            "UPDATE positions SET closed_at=?, close_price=?, close_reason=?, gross_pnl_pct=? "
            "WHERE id=?",
            (utcnow(), close_price, close_reason, gross_pnl_pct, position_id),
        )


def list_open_positions(
    db_path: Path = DB_PATH_DEFAULT, symbol: Optional[str] = None
) -> list[dict[str, Any]]:
    with get_conn(db_path) as cx:
        if symbol:
            rows = cx.execute(
                "SELECT * FROM positions WHERE closed_at IS NULL AND symbol=?",
                (symbol,),
            ).fetchall()
        else:
            rows = cx.execute(
                "SELECT * FROM positions WHERE closed_at IS NULL"
            ).fetchall()
        return [dict(r) for r in rows]


def has_open_position(symbol: str, db_path: Path = DB_PATH_DEFAULT) -> bool:
    return len(list_open_positions(db_path=db_path, symbol=symbol)) > 0


def count_open_positions(db_path: Path = DB_PATH_DEFAULT) -> int:
    return len(list_open_positions(db_path=db_path))

# ─────────────────────────── Risk State ────────────────────────────────────

def get_risk_state(db_path: Path = DB_PATH_DEFAULT) -> Optional[dict[str, Any]]:
    with get_conn(db_path) as cx:
        row = cx.execute("SELECT * FROM risk_state WHERE id=1").fetchone()
        return dict(row) if row else None

def insert_risk_state(state_dict: dict[str, Any], db_path: Path = DB_PATH_DEFAULT) -> None:
    with get_conn(db_path) as cx:
        cx.execute(
            "INSERT INTO risk_state (id, paper_equity, peak_equity, daily_start_equity, last_trade_date) "
            "VALUES (1, :paper_equity, :peak_equity, :daily_start_equity, :last_trade_date)",
            state_dict,
        )

def update_risk_state(updates: dict[str, Any], db_path: Path = DB_PATH_DEFAULT) -> None:
    if not updates:
        return
    set_clause = ", ".join(f"{k}=:{k}" for k in updates.keys())
    updates["id"] = 1
    with get_conn(db_path) as cx:
        cx.execute(f"UPDATE risk_state SET {set_clause} WHERE id=:id", updates)

if __name__ == "__main__":
    # Smoke test.
    init_db()
    log_event("bot_start", payload={"version": "0.1"})
    print("DB initialised OK at", DB_PATH_DEFAULT)
