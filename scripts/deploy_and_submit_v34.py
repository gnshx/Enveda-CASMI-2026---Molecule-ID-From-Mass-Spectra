#!/usr/bin/env python3
"""
scripts/deploy_and_submit_v34.py
=============================================================================
CASMI 2026: Version 34 Frontier Sovereign Consensus Deployment & Submission Runner
Integrates:
  1. Sovereign Rank-1 Shield (w_top = 0.00, PIN_OLD_LEADER = True)
  2. In-Test Molecular Networking with Hard Rank-1 Shield (tau >= 0.70, lambda = 0.25)
  3. Regime C Constitutional Regioisomers with MetFrag-lite bond-cleavage scoring
  4. Metric-Exact Canonical Tautomer InChIKey14 Deduplication across all 25 slots
Target: LB 0.495+
=============================================================================
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from kaggle.api.kaggle_api_extended import KaggleApi

PROXY = "http://176.111.37.5:39811"
os.environ["https_proxy"] = PROXY
os.environ["http_proxy"] = PROXY
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
DEPLOY_DIR = ROOT / "deploy_kernel_v34"
KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
LOG_FILE = ROOT / "logs" / "runner_v34.log"

VERSION_LABEL = (
    "Version 34: Frontier Sovereign Consensus (Rank-1 Shield + Constrained "
    "Molecular Network tau>=0.70 + Regime C Regioisomers + MIDX Dedup) (Target LB 0.495+)"
)

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_cmd(cmd: str, timeout_sec=180):
    env = os.environ.copy()
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=env,
            timeout=timeout_sec
        )
        return res
    except subprocess.TimeoutExpired:
        log(f"Command timed out ({timeout_sec}s): {cmd}")
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr="Timeout")

def get_kaggle_api():
    api = KaggleApi()
    api.authenticate()
    return api

def push_kernel():
    log("=" * 80)
    log("🚀 PUSHING KERNEL VERSION 34 TO KAGGLE")
    log("=" * 80)
    cmd = f"kaggle kernels push -p {DEPLOY_DIR}"
    res = run_cmd(cmd)
    log(f"Push stdout: {res.stdout.strip()}")
    if res.stderr.strip():
        log(f"Push stderr: {res.stderr.strip()}")
    if res.returncode != 0:
        log("❌ Kernel push failed!")
        return False
    log("✓ Kernel push succeeded.")
    return True

def wait_for_commit(api: KaggleApi, max_wait_sec=5400, poll_interval=30):
    log("\n⏳ Waiting for Kaggle cloud worker commit execution to complete...")
    start_time = time.time()
    while time.time() - start_time < max_wait_sec:
        status_res = api.kernels_status(KERNEL_SLUG)
        status_str = str(status_res.get("status", "")).lower()
        elapsed = time.time() - start_time
        log(f"  * Kernel status: {status_str} (elapsed: {elapsed:.1f}s)")

        if status_str == "complete":
            log("🎉 Kernel commit completed successfully!")
            return True
        elif status_str in ("error", "cancel_acknowledged"):
            log(f"❌ Kernel run ended with status: {status_str}")
            if status_res.get("failureMessage"):
                log(f"Failure message: {status_res.get('failureMessage')}")
            return False

        time.sleep(poll_interval)

    log(f"⚠️ Commit wait timed out after {max_wait_sec}s")
    return False

def submit_kernel(api: KaggleApi):
    log("\n" + "=" * 80)
    log("🚀 DISPATCHING SUBMISSION VERSION 34 TO ENVEDA CASMI 2026")
    log("=" * 80)

    # Fetch latest kernel output files
    log("Inspecting output files of latest kernel commit...")
    files_res = api.kernels_output(KERNEL_SLUG)
    out_files = files_res.get("files", [])
    sub_file = None
    for f in out_files:
        name = f.get("name", "")
        if name == "submission.csv":
            sub_file = f
            break

    if not sub_file:
        log("❌ Could not find submission.csv in kernel output!")
        return None

    log(f"✓ Found submission.csv: {sub_file}")

    # Trigger competition submission
    log(f"Submitting with description: '{VERSION_LABEL}'...")
    sub_res = api.competition_submit(
        file_name="submission.csv",
        message=VERSION_LABEL,
        competition=COMPETITION
    )
    log(f"Submission API response: {sub_res}")
    return sub_res

def main():
    parser = argparse.ArgumentParser(description="Deploy and Submit Version 34 to Kaggle")
    parser.add_argument("--submit-only", action="store_true", help="Skip kernel push and wait/submit directly")
    args = parser.parse_args()

    api = get_kaggle_api()

    if not args.submit_only:
        pushed = push_kernel()
        if not pushed:
            sys.exit(1)

    ok = wait_for_commit(api)
    if not ok:
        sys.exit(1)

    sub_res = submit_kernel(api)
    if not sub_res:
        sys.exit(1)

    log("✓ Version 34 dispatched successfully.")

if __name__ == "__main__":
    main()
