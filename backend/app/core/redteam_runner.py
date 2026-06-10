from datetime import datetime, timezone
from uuid import uuid4

from backend.app.models.redteam_models import LaunchCampaignRequest


_CAMPAIGNS: dict[str, dict[str, object]] = {}


def launch_campaign(payload: LaunchCampaignRequest) -> dict[str, object]:
    campaign_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    _CAMPAIGNS[campaign_id] = {
        "status": "running",
        "progress_pct": 5,
        "last_poll": 0,
        "log_lines": [
            f"[{now}] Campaign initialized",
            f"Strategy: {payload.strategy}",
            "Defense layers configured",
        ],
    }

    return {
        "campaign_id": campaign_id,
        "status": "started",
        "timestamp": now,
    }


def get_campaign_status(campaign_id: str) -> dict[str, object]:
    campaign = _CAMPAIGNS.get(campaign_id)
    if campaign is None:
        raise KeyError(f"Unknown campaign_id '{campaign_id}'.")

    if campaign["status"] == "running":
        campaign["progress_pct"] = min(int(campaign["progress_pct"]) + 15, 100)
        campaign["log_lines"].append(
            f"[{datetime.now(timezone.utc).isoformat()}] Progress {campaign['progress_pct']}%"
        )
        if campaign["progress_pct"] >= 100:
            campaign["status"] = "completed"
            campaign["log_lines"].append("Campaign completed")

    log_lines: list[str] = campaign["log_lines"]
    last_poll = int(campaign["last_poll"])
    new_lines = log_lines[last_poll:]
    campaign["last_poll"] = len(log_lines)

    return {
        "campaign_id": campaign_id,
        "status": campaign["status"],
        "log_lines": new_lines,
        "progress_pct": campaign["progress_pct"],
    }


def stop_campaign(campaign_id: str) -> dict[str, object]:
    campaign = _CAMPAIGNS.get(campaign_id)
    if campaign is None:
        raise KeyError(f"Unknown campaign_id '{campaign_id}'.")

    campaign["status"] = "stopped"
    campaign["log_lines"].append(
        f"[{datetime.now(timezone.utc).isoformat()}] Campaign stopped by user"
    )
    return {
        "success": True,
        "campaign_id": campaign_id,
        "status": "stopped",
    }
