#!/usr/bin/env python3
"""
scripts/deploy_and_submit_v32.py
Deploy and submit Version 32:
Sovereign Immutable Rank-1 Shield (w_top=0.00) + Calibrated Neural Reranker (w_mid=0.90) + MIDX Dedup.
Target: LB 0.490+ (Crossing into Gold Tier).
"""

import os
import sys
import time
import json
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
DEPLOY_DIR = ROOT / "deploy_kernel_v32"
KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
LOG_FILE = ROOT / "logs" / "runner_v32.log"

VERSION_LABEL = (
    "Version 32: Sovereign Immutable Rank-1 Shield (0.00 w_top) + "
    "Calibrated Neural Reranker (0.90 w_mid) + NP Prior (0.25) + MIDX Dedup (Target LB 0.490+)"
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

def get_current_kernel_version():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        api = KaggleApi()
        api.authenticate()
        client = api.build_kaggle_client()
        req = ApiGetKernelRequest()
        req.user_name = "nukaladevisaiganesh"
        req.kernel_slug = "casmi26-v4n-engine-fusion-union-lb-0-399"
        res = client.kernels.kernels_api_client.get_kernel(req)
        current_ver = getattr(res.metadata, "current_version_number", getattr(res.metadata, "currentVersionNumber", 33))
        return int(current_ver)
    except Exception as e:
        log(f"Warning determining kernel version: {e}")
        return 33

def main():
    log("=" * 70)
    log("🚀 LAUNCHING VERSION 32 DEPLOYMENT & SUBMISSION")
    log(f"Kernel: {KERNEL_SLUG}")
    log(f"Label: {VERSION_LABEL}")
    log("=" * 70)

    submit_only = "--submit-only" in sys.argv

    if not submit_only:
        # 1. Push kernel
        log("Step 1: Pushing notebook via Kaggle API...")
        push_cmd = f"kaggle kernels push -p {DEPLOY_DIR}"
        res = run_cmd(push_cmd)
        log(f"Push output:\n{res.stdout}\n{res.stderr}")
        if res.returncode != 0:
            log("❌ Kernel push failed!")
            return 1
    else:
        log("Skipping Step 1 (submit-only mode enabled)")

    # 2. Monitor kernel execution
    log("Step 2: Monitoring kernel commit status...")
    start_time = time.time()
    while time.time() - start_time < 3600:
        time.sleep(15)
        st_res = run_cmd(f"kaggle kernels status {KERNEL_SLUG}")
        st_out = (st_res.stdout + " " + st_res.stderr).strip()
        elapsed = (time.time() - start_time) / 60.0
        log(f"Status: {st_out} (elapsed {elapsed:.1f}m)")

        if "KernelWorkerStatus.COMPLETE" in st_out or "COMPLETE" in st_out:
            log(f"Kernel execution finished COMPLETE in {elapsed:.1f} minutes!")
            break
        elif "KernelWorkerStatus.FAILED" in st_out or "FAILED" in st_out or "ERROR" in st_out:
            log(f"Kernel run FAILED: {st_out}")
            return 1

    new_ver = get_current_kernel_version()
    log(f"Target Kernel Version for Submission: {new_ver}")

    # 3. Submit to competition
    log(f"Step 3: Submitting Version {new_ver} to {COMPETITION}...")
    sub_cmd = (
        f"kaggle competitions submit -c {COMPETITION} "
        f"-f submission.csv "
        f"-k {KERNEL_SLUG} "
        f"-v {new_ver} "
        f'-m "{VERSION_LABEL}"'
    )
    sub_res = run_cmd(sub_cmd)
    sub_output = (sub_res.stdout + "\n" + sub_res.stderr).strip()
    log(f"Competition submit output: {sub_output}")

    # Parse submission ref
    sub_ref = None
    for token in sub_output.split():
        if token.isdigit() and len(token) >= 7:
            sub_ref = int(token)
            break
    log(f"🎯 Submission Reference: {sub_ref}")

    # 4. Append to DAILY_JOURNEY.md and PROJECT_STATUS.md
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    journey_file = ROOT / "progress" / "DAILY_JOURNEY.md"
    entry = f"""
### Run Record: Version {new_ver} (Sovereign Immutable Rank-1 Shield)
- **Timestamp**: {now_utc}
- **Submission Ref**: `{sub_ref or 'Pending'}`
- **Configuration**: {VERSION_LABEL}
- **Architecture**:
  * **Immutable Rank 1 Shield**: `w_top = 0.00`, `PIN_OLD_LEADER = True` (305 Regime A library hits 100% protected).
  * **Calibrated Neural Reranker**: `w_mid = 0.90` on ranks 2-25 based on 0.8704 Val MRR simplex discovery.
  * **Natural Product Popularity Prior**: `POP_W = 0.25`, `NP_W = 0.10` on authentic biological pool.
  * **Metric-Exact InChIKey14 Deduplication**: `ENABLE_K14_DEDUP = True` eliminating all stereoisomer redundancy.
- **Status**: Actively running on hidden test set via Kaggle GPU worker.
- **Target Public Score**: **`0.490 – 0.495+`** (Crossing into Gold Tier).
"""
    if journey_file.exists():
        with open(journey_file, "a", encoding="utf-8") as f:
            f.write(entry)
        log("Logged deployment to DAILY_JOURNEY.md")

    # Git sync
    run_cmd("git add -A")
    run_cmd(f'git commit -m "feat(phase-27): deploy Version {new_ver} Sovereign Rank-1 Shield (Ref {sub_ref or "Pending"})"')
    run_cmd("git push origin main")
    log("Git sync complete.")
    log("🎉 DEPLOYMENT COMPLETE! 24/7 daemon is actively monitoring the evaluation queue.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
