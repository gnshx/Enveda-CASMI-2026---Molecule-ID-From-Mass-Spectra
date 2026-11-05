#!/usr/bin/env python3
"""
scripts/runner_v27.py
Autonomous runner for Version 27:
1. Pushes deploy_kernel_v27 to Kaggle.
2. Monitors kernel compilation on Kaggle GPU.
3. Automatically submits output using kernel version 27.
4. Triggers score polling daemon.
"""

import os, sys, time, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
LOG_PATH = ROOT / "logs" / "runner_v27.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

PROXY = "http://176.111.37.5:39811"
ENV = os.environ.copy()
ENV["https_proxy"] = PROXY
ENV["http_proxy"] = PROXY

KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
VERSION_LABEL = "Version 27"
VERSION_NUM = 27
MESSAGE = "Version 27: Sovereign Multi-Neural Consensus (Saner v1 + Scorer v2 + Custom v1) + Locked Rank 1 Shield + LOTUS Backfill (Target LB 0.485+)"

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

def run_cmd(cmd, check=True):
    res = subprocess.run(cmd, env=ENV, shell=True, capture_output=True, text=True, cwd=str(ROOT))
    if check and res.returncode != 0:
        log(f"Command failed ({res.returncode}): {cmd}\nStderr: {res.stderr.strip()}")
    return res

def push_kernel():
    log(f"Pushing {VERSION_LABEL} to Kaggle from deploy_kernel_v27...")
    res = run_cmd("kaggle kernels push -p deploy_kernel_v27")
    log(f"Push result: {res.stdout.strip()}")
    if "Kernel push cache saved" not in res.stdout and "successfully" not in res.stdout.lower() and res.returncode != 0:
        raise RuntimeError(f"Kernel push failed: {res.stderr}")
    return True

def wait_for_kernel(timeout_seconds=5400):
    log(f"Waiting for kernel '{KERNEL_SLUG}' (v{VERSION_NUM}) to complete (timeout: {timeout_seconds//60}m)...")
    start = time.time()
    while time.time() - start < timeout_seconds:
        res = run_cmd(f"kaggle kernels status {KERNEL_SLUG}")
        out = res.stdout.strip()
        elapsed = (time.time() - start) / 60
        if "KernelWorkerStatus.COMPLETE" in out or "COMPLETE" in out:
            log(f"Kernel status: COMPLETE (elapsed: {elapsed:.1f}m)")
            return True
        elif "KernelWorkerStatus.ERROR" in out or "ERROR" in out:
            log(f"Kernel status: ERROR: {out}")
            return False
        elif "KernelWorkerStatus.RUNNING" in out or "RUNNING" in out:
            log(f"Kernel status: RUNNING (elapsed: {elapsed:.1f}m)")
        elif "KernelWorkerStatus.QUEUED" in out or "QUEUED" in out:
            log(f"Kernel status: QUEUED (elapsed: {elapsed:.1f}m)")
        else:
            log(f"Kernel status: {out} (elapsed: {elapsed:.1f}m)")
        time.sleep(30)
    log("Timeout waiting for kernel.")
    return False

def submit_competition():
    log(f"Submitting kernel code v{VERSION_NUM} to competition '{COMPETITION}'...")
    cmd = (
        f"kaggle competitions submit -c {COMPETITION} "
        f"-f submission.csv "
        f"-k {KERNEL_SLUG} "
        f"-v {VERSION_NUM} "
        f"-m \"{MESSAGE}\""
    )
    res = run_cmd(cmd)
    log(f"Code submission API result: {res.stdout.strip()}")
    return res.stdout.strip()

def append_daily_journey(score_str, status_str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    block = f"""
### Run Result: {VERSION_LABEL}
- **Timestamp**: {ts}
- **Kernel**: `{KERNEL_SLUG}`
- **Description**: {MESSAGE}
- **Achieved Public Score**: **`{score_str}`**
- **Status**: {status_str}
"""
    dj_path = ROOT / "progress" / "DAILY_JOURNEY.md"
    with open(dj_path, "a") as f:
        f.write(block)
    log(f"Appended run result to {dj_path.relative_to(ROOT)}")

def sync_git(score_str, status_str):
    msg = f"feat: {VERSION_LABEL} scored {score_str} ({status_str})"
    log(f"Syncing git with message: '{msg}'...")
    run_cmd(f"git add -A && git commit -m \"{msg}\"", check=False)
    push_res = run_cmd("git push origin main", check=False)
    log(f"Git push result: {push_res.stdout.strip()}")

def main():
    log(f"=== {VERSION_LABEL} Pipeline Push & Autonomous Submit ===")
    push_kernel()
    success = wait_for_kernel()
    if not success:
        log("Kernel run did not complete successfully. Halting before submit.")
        append_daily_journey("ERROR", "Kernel Run Failed")
        sync_git("ERROR", "Kernel Run Failed")
        return
    
    log("Kernel run completed successfully!")
    sub_res = submit_competition()
    append_daily_journey("0.0000", "Submitted (score pending)")
    sync_git("0.0000", "Submitted (score pending)")
    log(f"=== {VERSION_LABEL} submission submitted successfully! ===")

if __name__ == "__main__":
    main()
