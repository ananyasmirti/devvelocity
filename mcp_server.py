"""
mcp_server.py
MCP (Model Context Protocol) server for DevVelocity AI.

The AI coding agent calls these tools at runtime. Each tool is:
  1. Decorated with @app.tool() to expose it via MCP
  2. Wrapped with @require_clearance() to enforce RBAC
  
The agent can ONLY see data the calling developer is cleared for.
This is the "safely decoupled AI context layer" from the resume bullet.

Run with:
    python mcp_server.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent))
from telemetry_pipeline import TelemetryPipeline
from rbac_middleware import ClearanceLevel, get_clearance, require_clearance


# ── Server setup ──────────────────────────────────────────────────────────────

app = FastMCP("DevVelocity AI")
pipeline = TelemetryPipeline()

try:
    bundle = joblib.load("models/devvelocity_model.pkl")
    model = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]
    label_map: dict = bundle["label_map"]
    print("✅ LightGBM model loaded")
except FileNotFoundError:
    model = None
    feature_cols = []
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    print("⚠️  No model found — run train_model.py first")


def _run_inference(developer_id: str) -> dict:
    """Pull live telemetry features and run LightGBM inference."""
    if model is None:
        return {"error": "Model not loaded. Run: python src/ml/train_model.py"}

    live_feats = pipeline.compute_live_features(developer_id, window_minutes=60)
    row = {col: live_feats.get(col, np.nan) for col in feature_cols}
    df_row = pd.DataFrame([row]).fillna(0)

    pred = int(model.predict(df_row)[0])
    proba = model.predict_proba(df_row)[0]

    return {
        "efficiency_class": label_map[pred],
        "confidence": round(float(proba[pred]), 3),
        "probabilities": {label_map[i]: round(float(p), 3) for i, p in enumerate(proba)},
    }


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@app.tool()
@require_clearance("own_metrics")
async def get_my_metrics(caller_id: str, clearance: ClearanceLevel) -> dict:
    """
    [VIEWER+] Fetch the calling developer's own telemetry summary.
    Any developer can call this for themselves.
    """
    features = pipeline.compute_live_features(caller_id, window_minutes=60)
    return {"developer_id": caller_id, "session_summary": features}


@app.tool()
@require_clearance("own_efficiency_score")
async def get_my_efficiency_score(caller_id: str, clearance: ClearanceLevel) -> dict:
    """
    [VIEWER+] Get the calling developer's current SPACE efficiency score.
    Runs real-time LightGBM inference on live telemetry features.
    """
    result = _run_inference(caller_id)
    result["developer_id"] = caller_id
    return result


@app.tool()
@require_clearance("team_aggregate")
async def get_team_aggregate(caller_id: str, clearance: ClearanceLevel) -> dict:
    """
    [DEVELOPER+] Get anonymized team-level productivity stats.
    Returns aggregates only — no individual data exposed.
    """
    import sqlite3

    conn = sqlite3.connect("telemetry.db")
    rows = conn.execute(
        """SELECT developer_id, COUNT(*) as events,
                  SUM(CASE WHEN event_type='context_switch' THEN 1 ELSE 0 END) as switches,
                  SUM(CASE WHEN event_type='ai_query' THEN 1 ELSE 0 END) as ai_queries
           FROM telemetry_events
           GROUP BY developer_id"""
    ).fetchall()
    conn.close()

    if not rows:
        return {"team_size": 0, "message": "No telemetry data yet"}

    total_events = sum(r[1] for r in rows)
    total_switches = sum(r[2] for r in rows)
    total_ai = sum(r[3] for r in rows)

    return {
        "team_size": len(rows),
        "total_events": total_events,
        "avg_events_per_dev": round(total_events / len(rows), 1),
        "avg_context_switches": round(total_switches / len(rows), 1),
        "avg_ai_queries": round(total_ai / len(rows), 1),
        "flow_ratio": round(
            (total_events - total_switches) / max(total_switches, 1), 2
        ),
    }


@app.tool()
@require_clearance("team_individual")
async def get_developer_score(
    target_dev_id: str,
    caller_id: str,
    clearance: ClearanceLevel,
) -> dict:
    """
    [LEAD+] Get a specific developer's efficiency score.
    Only Team Leads and Admins can view other developers' individual scores.
    """
    result = _run_inference(target_dev_id)
    result["developer_id"] = target_dev_id
    result["queried_by"] = caller_id
    return result


@app.tool()
@require_clearance("raw_telemetry")
async def get_raw_telemetry(
    target_dev_id: str,
    limit: int,
    caller_id: str,
    clearance: ClearanceLevel,
) -> dict:
    """
    [LEAD+] Retrieve raw telemetry events for a specific developer.
    Max 100 records per call.
    """
    import json
    import sqlite3

    safe_limit = min(abs(limit), 100)
    conn = sqlite3.connect("telemetry.db")
    rows = conn.execute(
        """SELECT event_type, metadata, timestamp
           FROM telemetry_events
           WHERE developer_id = ?
           ORDER BY timestamp DESC
           LIMIT ?""",
        (target_dev_id, safe_limit),
    ).fetchall()
    conn.close()

    return {
        "developer_id": target_dev_id,
        "record_count": len(rows),
        "events": [
            {"type": r[0], "metadata": json.loads(r[1]), "timestamp": r[2]}
            for r in rows
        ],
    }


@app.tool()
@require_clearance("model_retrain")
async def trigger_model_retrain(caller_id: str, clearance: ClearanceLevel) -> dict:
    """
    [ADMIN] Trigger a model retraining job.
    Only admins can retrain the production model.
    """
    # In production: enqueue a Celery/RQ job, or call a training API
    return {
        "status": "queued",
        "message": f"Retrain job queued by {caller_id}",
        "note": "Connect to your job queue (Celery/RQ) to execute",
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run()
