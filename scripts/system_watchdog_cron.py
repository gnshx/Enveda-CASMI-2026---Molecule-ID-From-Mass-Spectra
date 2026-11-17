#!/usr/bin/env python3
"""
scripts/system_watchdog_cron.py
=============================================================================
System OS Watchdog for CASMI 2026
Runs from Linux crontab every 10 minutes on the PC host OS (outside AGY).
Ensures system_os_scheduler is continuously running in the background.
=============================================================================
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
LOG_FILE = ROOT / "logs" / "system_watchdog.log"
SCHEDULER_SCRIPT = ROOT / "scripts" / "system_os_scheduler.py"
PID_FILE = ROOT / "logs" / "system_scheduler.pid"

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def is_scheduler_running():
    res = subprocess.run(
        "pgrep -f system_os_scheduler.py",
        shell=True,
        capture_output=True,
        text=True
    )
    pids = res.stdout.strip().split()
    return len(pids) > 0, pids

def start_scheduler():
    log("Scheduler not detected. Launching system_os_scheduler.py...")
    cmd = f"/usr/bin/python3 -u {SCHEDULER_SCRIPT} >> {ROOT}/logs/system_scheduler.log 2>&1 &"
    subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(ROOT),
        start_new_session=True,
        close_fds=True
    )
    log("✓ system_os_scheduler launched with new session.")

if __name__ == "__main__":
    running, pids = is_scheduler_running()
    if running:
        log(f"Watchdog check: system_os_scheduler is ALIVE (PID {pids}).")
    else:
        log("Watchdog check: system_os_scheduler is DOWN.")
        start_scheduler()
