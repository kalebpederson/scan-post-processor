"""Desktop notifications via notify-send (best effort: never raises)."""
from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger("scanpp")


def notify(summary: str, body: str = "", urgent: bool = False, expire_ms: int = 6000) -> bool:
    """Public entry point (tests monkeypatch this name); see send_notification."""
    return send_notification(summary, body, urgent, expire_ms)


def send_notification(summary: str, body: str = "", urgent: bool = False, expire_ms: int = 6000) -> bool:
    """Run notify-send. Refuses to show real notifications when SCANPP_NOTIFY=0 or under pytest; an explicit
    SCANPP_NOTIFY_CMD (a test double) is always honoured."""
    cmd = os.environ.get("SCANPP_NOTIFY_CMD")
    if not cmd:
        if os.environ.get("SCANPP_NOTIFY") == "0" or "PYTEST_CURRENT_TEST" in os.environ:
            log.info("notifications disabled; not sending: %s", summary)
            return False
        cmd = "notify-send"
    args = [cmd, "-a", "scanpp", "-t", str(0 if urgent else expire_ms), "-u", "critical" if urgent else "low",
            summary, body]
    try:
        subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10, check=True)
        return True
    except Exception as e:  # noqa: BLE001  (missing binary, no session bus, timeout...)
        log.info("notification failed (%s): %s", type(e).__name__, e)
        return False
