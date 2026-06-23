"""
Test script: DEVELOPER-level clearance
Tests what a DEVELOPER can access via MCP tools
"""

import asyncio
from rbac_middleware import ClearanceLevel, get_clearance

# Use an existing test developer from the registry
# carol is a DEVELOPER in the test registry
caller_id = "carol"
clearance = get_clearance(caller_id)

async def test_developer_access():
    """Test DEVELOPER-level tool access"""
    print(f"\n🧪 Testing as: {caller_id} (Clearance: {clearance.name} = level {int(clearance)})\n")
    
    # Import after setting up paths
    from mcp_server import (
        get_my_metrics,
        get_my_efficiency_score,
        get_team_aggregate,
        get_developer_score,
    )
    
    # ✅ DEVELOPER can access: own_metrics
    print("1️⃣  get_my_metrics (VIEWER+) — ✅ ALLOWED")
    try:
        result = await get_my_metrics(caller_id=caller_id, clearance=clearance)
        print(f"   Result: {result}\n")
    except PermissionError as e:
        print(f"   ❌ DENIED: {e}\n")
    
    # ✅ DEVELOPER can access: own_efficiency_score
    print("2️⃣  get_my_efficiency_score (VIEWER+) — ✅ ALLOWED")
    try:
        result = await get_my_efficiency_score(caller_id=caller_id, clearance=clearance)
        print(f"   Result: {result}\n")
    except PermissionError as e:
        print(f"   ❌ DENIED: {e}\n")
    
    # ✅ DEVELOPER can access: team_aggregate
    print("3️⃣  get_team_aggregate (DEVELOPER+) — ✅ ALLOWED")
    try:
        result = await get_team_aggregate(caller_id=caller_id, clearance=clearance)
        print(f"   Result: {result}\n")
    except PermissionError as e:
        print(f"   ❌ DENIED: {e}\n")
    
    # ❌ DEVELOPER cannot access: team_individual (LEAD+ only)
    print("4️⃣  get_developer_score (LEAD+) — ❌ DENIED")
    try:
        result = await get_developer_score(target_dev_id="bob", caller_id=caller_id, clearance=clearance)
        print(f"   Result: {result}\n")
    except PermissionError as e:
        print(f"   ❌ DENIED (as expected): {e}\n")

if __name__ == "__main__":
    asyncio.run(test_developer_access())
