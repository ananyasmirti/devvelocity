"""
rbac_middleware.py
Role-Based Access Control for the MCP server.

Every MCP tool function is decorated with @require_clearance("resource_name").
The decorator checks the caller's clearance before executing the tool —
the AI coding agent cannot retrieve data above its clearance level.

Clearance levels:
  VIEWER    (1) → own metrics only
  DEVELOPER (2) → team-level aggregates
  LEAD      (3) → individual data for direct reports
  ADMIN     (4) → full dataset access
"""

from enum import IntEnum
from functools import wraps
from typing import Any, Callable


# ── Clearance levels ──────────────────────────────────────────────────────────

class ClearanceLevel(IntEnum):
    VIEWER    = 1
    DEVELOPER = 2
    LEAD      = 3
    ADMIN     = 4


# ── Resource → minimum required clearance ────────────────────────────────────

RESOURCE_PERMISSIONS: dict[str, ClearanceLevel] = {
    "own_metrics":           ClearanceLevel.VIEWER,
    "own_efficiency_score":  ClearanceLevel.VIEWER,
    "team_aggregate":        ClearanceLevel.DEVELOPER,
    "team_ai_adoption":      ClearanceLevel.DEVELOPER,
    "model_scores_team":     ClearanceLevel.DEVELOPER,
    "team_individual":       ClearanceLevel.LEAD,
    "raw_telemetry":         ClearanceLevel.LEAD,
    "all_data":              ClearanceLevel.ADMIN,
    "model_retrain":         ClearanceLevel.ADMIN,
}


# ── RBAC decorator ────────────────────────────────────────────────────────────

def require_clearance(resource: str):
    """
    Decorator that enforces RBAC before an MCP tool executes.
    
    The decorated function must accept:
      caller_id: str           — who is calling
      clearance: ClearanceLevel — their clearance level

    Raises:
      PermissionError if clearance < required level for the resource
    
    Usage:
        @app.tool()
        @require_clearance("team_individual")
        async def get_dev_score(target_id: str, caller_id: str, clearance: ClearanceLevel):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(
            *args,
            caller_id: str,
            clearance: ClearanceLevel,
            **kwargs,
        ) -> Any:
            required = RESOURCE_PERMISSIONS.get(resource, ClearanceLevel.ADMIN)

            if clearance < required:
                _audit_log(caller_id, resource, func.__name__, granted=False)
                raise PermissionError(
                    f"[RBAC] Access DENIED for '{caller_id}': "
                    f"'{resource}' requires {required.name} (level {int(required)}), "
                    f"caller has {clearance.name} (level {int(clearance)})."
                )

            _audit_log(caller_id, resource, func.__name__, granted=True)
            return await func(*args, caller_id=caller_id, clearance=clearance, **kwargs)

        # Attach metadata so MCP server can expose permission requirements
        wrapper._rbac_resource = resource
        wrapper._rbac_required = RESOURCE_PERMISSIONS.get(resource, ClearanceLevel.ADMIN)
        return wrapper

    return decorator


# ── Audit log (replace with DB write in production) ──────────────────────────

def _audit_log(caller_id: str, resource: str, func: str, granted: bool):
    status = "✅ GRANTED" if granted else "❌ DENIED"
    print(f"[RBAC] {status} | caller={caller_id} | resource={resource} | tool={func}")


# ── Developer registry ────────────────────────────────────────────────────────
# In production: replace with DB lookup or OAuth/JWT claims

_REGISTRY: dict[str, ClearanceLevel] = {
    "alice":  ClearanceLevel.ADMIN,
    "bob":    ClearanceLevel.LEAD,
    "carol":  ClearanceLevel.DEVELOPER,
    "david":  ClearanceLevel.VIEWER,
}


def get_clearance(developer_id: str) -> ClearanceLevel:
    """Look up a developer's clearance level. Defaults to VIEWER if unknown."""
    return _REGISTRY.get(developer_id, ClearanceLevel.VIEWER)


def register_developer(developer_id: str, clearance: ClearanceLevel):
    """Register or update a developer's clearance."""
    _REGISTRY[developer_id] = clearance
    print(f"[RBAC] Registered {developer_id} → {clearance.name}")
