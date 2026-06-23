"""
telemetry_pipeline.py
Async IDE telemetry ingestion pipeline.

Your IDE plugin (VS Code extension, JetBrains plugin, etc.) POSTs events here.
Events are queued in memory and flushed to SQLite in batches — so the IDE
never blocks waiting for a DB write.

Event types captured:
  keystroke_burst   → deep coding activity (flow state signal)
  file_save         → task completion proxy
  ai_query          → AI tool usage (efficiency signal)
  context_switch    → flow interruption (negative efficiency signal)
  debug_session     → problem-solving activity
  terminal_command  → build/test activity
"""

import asyncio
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TelemetryEvent:
    developer_id: str
    event_type: str
    metadata: dict
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    clearance_level: int = 1  # mirrors RBAC clearance — used for row-level security


# ── Pipeline ──────────────────────────────────────────────────────────────────

class TelemetryPipeline:
    """
    Non-blocking telemetry ingestion with batched SQLite writes.

    Usage:
        pipeline = TelemetryPipeline()
        asyncio.create_task(pipeline.run())          # start background worker
        await pipeline.ingest(TelemetryEvent(...))   # fire-and-forget from IDE
    """

    def __init__(self, db_path: str = "telemetry.db", batch_size: int = 50):
        self.queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue()
        self.db_path = db_path
        self.batch_size = batch_size
        self._batch: list[TelemetryEvent] = []
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS telemetry_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                developer_id    TEXT    NOT NULL,
                event_type      TEXT    NOT NULL,
                session_id      TEXT    NOT NULL,
                metadata        TEXT,
                timestamp       TEXT    NOT NULL,
                clearance_level INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_dev_id
                ON telemetry_events(developer_id);
            CREATE INDEX IF NOT EXISTS idx_event_type
                ON telemetry_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_timestamp
                ON telemetry_events(timestamp);
        """)
        conn.commit()
        conn.close()
        print(f"📦 Telemetry DB ready: {self.db_path}")

    async def ingest(self, event: TelemetryEvent):
        """Non-blocking — caller never waits on the DB write."""
        await self.queue.put(event)

    async def run(self):
        """Background worker — drains the queue continuously."""
        print("📡 Telemetry pipeline running...")
        while True:
            event = await self.queue.get()
            self._batch.append(event)
            if len(self._batch) >= self.batch_size or self.queue.empty():
                self._flush()
            self.queue.task_done()

    def _flush(self):
        if not self._batch:
            return
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            """INSERT INTO telemetry_events
               (developer_id, event_type, session_id, metadata, timestamp, clearance_level)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    e.developer_id,
                    e.event_type,
                    e.session_id,
                    json.dumps(e.metadata),
                    e.timestamp,
                    e.clearance_level,
                )
                for e in self._batch
            ],
        )
        conn.commit()
        conn.close()
        print(f"   💾 Flushed {len(self._batch)} events to DB")
        self._batch.clear()

    # ── Feature extraction (feeds the live LightGBM inference) ───────────────

    def compute_live_features(
        self, developer_id: str, window_minutes: int = 60
    ) -> dict:
        """
        Compute real-time SPACE-E features from recent telemetry.
        These are passed to predict() in train_model.py at inference time.
        
        Returns a dict that partially covers the model's feature_cols.
        Missing cols are filled with NaN → then median-imputed before prediction.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT event_type, metadata
               FROM telemetry_events
               WHERE developer_id = ?
                 AND datetime(timestamp) >= datetime('now', ?)
               ORDER BY timestamp""",
            (developer_id, f"-{window_minutes} minutes"),
        ).fetchall()
        conn.close()

        if not rows:
            return {}

        events = [(r[0], json.loads(r[1])) for r in rows]
        type_counts = {}
        for etype, _ in events:
            type_counts[etype] = type_counts.get(etype, 0) + 1

        keystroke_bursts = type_counts.get("keystroke_burst", 0)
        context_switches = type_counts.get("context_switch", 0)
        ai_queries = type_counts.get("ai_query", 0)
        file_saves = type_counts.get("file_save", 0)

        # flow_ratio: high = lots of coding, few interruptions = flow state
        flow_ratio = keystroke_bursts / max(context_switches + 1, 1)

        return {
            # Maps to SPACE-E dimension features the model knows about
            "eff_ai_tool_count": ai_queries,
            "act_coding_activity_count": file_saves,
            # Runtime-only features (model trained without these,
            # but you can add them in v2 after collecting real telemetry):
            "_rt_context_switches": context_switches,
            "_rt_flow_ratio": round(flow_ratio, 2),
            "_rt_keystroke_bursts": keystroke_bursts,
        }


# ── Demo / smoke test ─────────────────────────────────────────────────────────

async def _demo():
    pipeline = TelemetryPipeline(db_path="demo_telemetry.db")

    sample_events = [
        TelemetryEvent("dev_001", "keystroke_burst", {"chars": 342, "lang": "python"}),
        TelemetryEvent("dev_001", "file_save", {"file": "train_model.py", "lines": 120}),
        TelemetryEvent("dev_001", "ai_query", {"tool": "Copilot", "accepted": True}),
        TelemetryEvent("dev_001", "context_switch", {"from": "vscode", "to": "browser"}),
        TelemetryEvent("dev_001", "keystroke_burst", {"chars": 512, "lang": "python"}),
        TelemetryEvent("dev_001", "keystroke_burst", {"chars": 280, "lang": "python"}),
        TelemetryEvent("dev_002", "debug_session", {"duration_s": 180, "breakpoints": 3}),
        TelemetryEvent("dev_002", "ai_query", {"tool": "Claude", "accepted": False}),
    ]

    worker = asyncio.create_task(pipeline.run())

    for event in sample_events:
        await pipeline.ingest(event)
        await asyncio.sleep(0.01)

    await pipeline.queue.join()
    worker.cancel()

    print("\n📊 dev_001 live features (last 5 min window):")
    feats = pipeline.compute_live_features("dev_001", window_minutes=5)
    for k, v in feats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(_demo())
