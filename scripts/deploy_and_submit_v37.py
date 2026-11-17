#!/usr/bin/env python3
"""
scripts/deploy_and_submit_v37.py
=============================================================================
CASMI 2026: Version 37 Unified Grand Champion Deployment Runner
Integrates:
  1. Tri-Neural Multi-Capacity Backbone (Saner v1 + Scorer v2 consensus)
  2. Phase 36 Collision-Energy & Peak-Count Weighted Multi-Spectrum Consensus
  3. High-Resolution Isotope Pattern Consistency Engine (chemistry/isotope_engine.py)
  4. 4-Channel Learned Multimodal Weighting (w_neural=1.63, w_frag=1.60, w_iso=0.75, w_mass=0.50, w_plaus=0.30)
  5. MetFrag-lite Bond Cleavage Engine (<= 2 breaks, sqrt intensity)
  6. Bruker timsTOF -1.50 ppm Quadrupole Empirical Mass Calibration Prior
  7. Sovereign Immutable Rank-1 Shield (w_top = 0.00, 100% frozen gold library hits)
  8. Phase 28 Regime C Constitutional Regioisomer Generation on depleted pools (< 25)
  9. Phase 29 Constrained In-Test Molecular Networking (tau >= 0.70, lambda = 0.20)
 10. Metric-Exact Canonical Tautomer InChIKey14 Deduplication across all 25 candidate slots
Target: LB 0.525 – 0.540+ (Grand Champion Tier)
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
DEPLOY_DIR = ROOT / "deploy_kernel_v37"
KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
LOG_FILE = ROOT / "logs" / "runner_v37.log"

VERSION_LABEL = (
    "Version 37: Unified Grand Champion Multi-Modal Consensus (Tri-Neural + Phase 36 Multi-CE + "
    "Isotope Engine + Regioisomers + Molecular Network + Pure Shield) (Target LB 0.525+)"
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
    log("👑 PUSHING KERNEL VERSION 37 TO KAGGLE")
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
        current_ver = getattr(res.metadata, "current_version_number", getattr(res.metadata, "currentVersionNumber", 37))
        return int(current_ver)
    except Exception as e:
        log(f"Warning determining kernel version: {e}")
        return 37

def wait_for_commit(max_wait_sec=5400, poll_interval=15):
    log("\n⏳ Waiting for Kaggle cloud worker commit execution to complete...")
    start_time = time.time()
    while time.time() - start_time < max_wait_sec:
        st_res = run_cmd(f"kaggle kernels status {KERNEL_SLUG}")
        st_out = (st_res.stdout + " " + st_res.stderr).strip()
        elapsed = (time.time() - start_time) / 60.0
        log(f"  * Status: {st_out} (elapsed {elapsed:.1f}m)")

        if "KernelWorkerStatus.COMPLETE" in st_out or "COMPLETE" in st_out:
            log("🎉 Kernel commit completed successfully!")
            return True
        elif "KernelWorkerStatus.FAILED" in st_out or "FAILED" in st_out or "ERROR" in st_out:
            log(f"❌ Kernel run FAILED: {st_out}")
            return False

        time.sleep(poll_interval)

    log(f"⚠️ Commit wait timed out after {max_wait_sec}s")
    return False

def submit_kernel(version_num: int):
    log("\n" + "=" * 80)
    log(f"🚀 DISPATCHING SUBMISSION VERSION 37 (Kernel Version {version_num}) TO ENVEDA CASMI 2026")
    log("=" * 80)

    sub_cmd = (
        f"kaggle competitions submit -c {COMPETITION} "
        f"-f submission.csv "
        f"-k {KERNEL_SLUG} "
        f"-v {version_num} "
        f'-m "{VERSION_LABEL}"'
    )
    sub_res = run_cmd(sub_cmd)
    sub_output = (sub_res.stdout + "\n" + sub_res.stderr).strip()
    log(f"Competition submit output:\n{sub_output}")

    sub_ref = None
    for token in sub_output.split():
        if token.isdigit() and len(token) >= 7:
            sub_ref = int(token)
            break
    log(f"🎯 Submission Reference: {sub_ref}")
    return sub_ref

def main():
    parser = argparse.ArgumentParser(description="Deploy and Submit Version 37 to Kaggle")
    parser.add_argument("--submit-only", action="store_true", help="Skip kernel push and wait/submit directly")
    args = parser.parse_args()

    api = get_kaggle_api()

    if not args.submit_only:
        pushed = push_kernel()
        if not pushed:
            sys.exit(1)

    ok = wait_for_commit()
    if not ok:
        sys.exit(1)

    ver = get_current_kernel_version()
    sub_ref = submit_kernel(ver)
    if not sub_ref:
        sys.exit(1)

    log("✓ Version 37 dispatched successfully.")

if __name__ == "__main__":
    main()
