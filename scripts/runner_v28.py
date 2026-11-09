#!/usr/bin/env python3
"""
scripts/runner_v28.py
Autonomous runner for Version 28:
1. Checks if kernel is already running or complete; pushes deploy_kernel_v28 if needed.
2. Monitors kernel run until complete on GPU Tesla T4.
3. Submits kernel output to Kaggle competition.
4. Updates DAILY_JOURNEY.md and ROADMAP_AND_PROGRESS.md.
5. Commits and pushes to GitHub.
"""

import os, sys, time, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
LOG_PATH = ROOT / "logs" / "runner_v28.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

PROXY = "http://176.111.37.5:39811"
ENV = os.environ.copy()
ENV["https_proxy"] = PROXY
ENV["http_proxy"] = PROXY

KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
VERSION_LABEL = "Version 28"
VERSION_NUM = 28
MESSAGE = "Version 28: Apex Sovereign Dual-Tier ExtLib (Gold/Silver) + 0.65/0.35 Scorer-v5 Ensemble + Strict Rank 1 Shield (Target LB 0.490+)"

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

def push_kernel_if_needed():
    res = run_cmd(f"kaggle kernels status {KERNEL_SLUG}")
    out = res.stdout.strip()
    if "RUNNING" in out or "QUEUED" in out:
        log(f"Kernel '{KERNEL_SLUG}' is already running/queued ({out}). Skipping push.")
        return True
    log(f"Pushing {VERSION_LABEL} to Kaggle from deploy_kernel_v28...")
    push_res = run_cmd("kaggle kernels push -p deploy_kernel_v28")
    log(f"Push result: {push_res.stdout.strip()}")
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
        f'-m "{MESSAGE}"'
    )
    res = run_cmd(cmd)
    log(f"Code submission API result: {res.stdout.strip()}")
    if res.returncode != 0:
        log(f"Submission error: {res.stderr.strip()}")
        return False
    return True

def append_journey(score=0.0000, status="Submitted (scoring in progress)"):
    journey_path = ROOT / "progress" / "DAILY_JOURNEY.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"""
### Run Record: {VERSION_LABEL}
- **Timestamp**: {ts}
- **Configuration**: {MESSAGE}
- **Public LB Score**: **`{score:.4f}`**
- **Status**: {status}
- **Artifacts**: Deployed from `deploy_kernel_v28/`
"""
    with open(journey_path, "a") as f:
        f.write(entry)
    log(f"Appended run result to {journey_path.relative_to(ROOT)}")

def sync_git(score=0.0000, status="Submitted (scoring in progress)"):
    msg = f"feat: {VERSION_LABEL} pushed & submitted ({status})"
    log(f"Syncing git with message: '{msg}'...")
    run_cmd("git add -A")
    run_cmd(f'git commit -m "{msg}"', check=False)
    res = run_cmd("git push origin main", check=False)
    log(f"Git push result: {res.stdout.strip()}")

def main():
    log(f"=== {VERSION_LABEL} Pipeline Autonomous Runner Started ===")
    
    # 1. Push if not already executing
    push_kernel_if_needed()
    
    # 2. Wait for Kaggle GPU Worker to Complete
    success = wait_for_kernel()
    if not success:
        log("Kernel run failed or timed out. Aborting submission.")
        sys.exit(1)
        
    log("Kernel run completed successfully!")
    
    # 3. Submit Code to Competition
    sub_ok = submit_competition()
    if not sub_ok:
        log("Submission failed!")
        sys.exit(1)
        
    # 4. Update Logs & Git
    append_journey()
    sync_git()
    log(f"=== {VERSION_LABEL} submitted successfully! Score poller will record verified score. ===")

if __name__ == "__main__":
    main()
