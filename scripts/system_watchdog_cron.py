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
SCIENCE_SCRIPT = ROOT / "scripts" / "autonomous_science_loop.py"

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def is_scheduler_running():
    res = subprocess.run(
        "pgrep -af '^/usr/bin/python3 -u /home/dlcv/Enveda-CASMI-2026/scripts/system_os_scheduler.py$'",
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

def is_science_loop_running():
    res = subprocess.run(
        "pgrep -af '^/usr/bin/python3 -u /home/dlcv/Enveda-CASMI-2026/scripts/autonomous_science_loop.py$'",
        shell=True,
        capture_output=True,
        text=True
    )
    return bool(res.stdout.strip()), res.stdout.strip().splitlines()

def start_science_loop():
    log("Offline science loop not detected. Launching autonomous_science_loop.py...")
    cmd = f"/usr/bin/python3 -u {SCIENCE_SCRIPT} >> {ROOT}/logs/science_loop.log 2>&1 &"
    subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(ROOT),
        start_new_session=True,
        close_fds=True
    )
    log("✓ autonomous_science_loop launched with new session.")

if __name__ == "__main__":
    running, pids = is_scheduler_running()
    if running:
        log(f"Watchdog check: system_os_scheduler is ALIVE (PID {pids}).")
    else:
        log("Watchdog check: system_os_scheduler is DOWN.")
        start_scheduler()

    science_running, science_pids = is_science_loop_running()
    if science_running:
        log(f"Offline science loop is ALIVE: {science_pids}.")
    else:
        start_science_loop()
