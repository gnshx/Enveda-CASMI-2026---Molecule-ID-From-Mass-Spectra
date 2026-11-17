#!/usr/bin/env python3
"""
scripts/auto_submit_v34_v35_v36.py
=============================================================================
CASMI 2026: Master Autonomous 3-Version Submission & Scoring Pipeline (24/7)
Orchestrates:
  1. Version 34 (Ref 58462901): Monitor PENDING -> COMPLETE on Kaggle GPU, record score.
  2. Version 35 (Kernel v35): Monitor RUNNING -> COMPLETE on Kaggle GPU, submit automatically.
  3. Version 36 (Kernel v36): Push deploy_kernel_v36, monitor commit, submit automatically.
  4. Continuous telemetry, git synchronization, and score reporting.
=============================================================================
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

PROXY = "http://176.111.37.5:39811"
os.environ["https_proxy"] = PROXY
os.environ["http_proxy"] = PROXY
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
LOG_FILE = ROOT / "logs" / "auto_submit_3ver.log"
KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"

REF_V34 = "58462901"

DESC_V35 = (
    "Version 35: Frontier Sovereign Constrained Molecular Network (tau >= 0.70, lambda = 0.20) + "
    "Restored Dual-Neural Ensemble + Strict Rank-1 Shield (Target LB 0.495+)"
)

DESC_V36 = (
    "Version 36: Apex Sovereign Frontier Milestone 3 Consensus (Regime C Regioisomer Backfill + "
    "Dual-Neural + Pure Shield) (Target LB 0.510+)"
)

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_cmd(cmd: str, timeout_sec=180, unset_proxy_for_git=True):
    env = os.environ.copy()
    if unset_proxy_for_git and "git" in cmd:
        for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
            env.pop(k, None)
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

def get_submission_status(ref: str):
    cmd = f"kaggle competitions submissions -c {COMPETITION}"
    res = run_cmd(cmd)
    if res.returncode != 0:
        return None, None
    for line in res.stdout.split("\n"):
        if ref in line:
            parts = line.split()
            # ref fileName date time desc ... status publicScore
            for i, p in enumerate(parts):
                if "SubmissionStatus." in p:
                    status = p
                    score = parts[i+1] if (i+1 < len(parts) and not parts[i+1].startswith("-")) else None
                    return status, score
    return None, None

def get_kernel_status():
    cmd = f"kaggle kernels status {KERNEL_SLUG}"
    res = run_cmd(cmd)
    out = (res.stdout + " " + res.stderr).strip()
    return out

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
        return int(res.metadata.current_version_number)
    except Exception as e:
        log(f"Warning determining kernel version: {e}")
        return 35

def submit_kernel_version(ver: int, desc: str):
    log(f"🚀 SUBMITTING KERNEL VERSION {ver} TO {COMPETITION}...")
    cmd = (
        f'kaggle competitions submit -c {COMPETITION} '
        f'-k {KERNEL_SLUG} -v {ver} -m "{desc}" -f submission.csv'
    )
    res = run_cmd(cmd, timeout_sec=120)
    log(f"Submission stdout: {res.stdout.strip()}")
    if res.stderr.strip():
        log(f"Submission stderr: {res.stderr.strip()}")
    
    # Extract submission ref
    out = res.stdout + " " + res.stderr
    ref = None
    for word in out.split():
        if word.isdigit() and len(word) >= 7:
            ref = word
            break
    if not ref and "Successfully submitted" in out:
        time.sleep(5)
        # Check submissions list for top ref
        s_res = run_cmd(f"kaggle competitions submissions -c {COMPETITION}")
        for l in s_res.stdout.split("\n")[:5]:
            p = l.split()
            if p and p[0].isdigit() and len(p[0]) >= 7:
                ref = p[0]
                break
    return ref

def sync_git(msg: str):
    log(f"Syncing git: {msg}")
    run_cmd("git add -A")
    run_cmd(f'git commit -m "{msg}"')
    run_cmd("git push origin main")

def main():
    log("=" * 80)
    log("🚀 CASMI 2026: AUTONOMOUS 3-VERSION SUBMISSION ORCHESTRATOR LAUNCHED")
    log("=" * 80)

    # 1. State tracking
    v34_score = None
    v35_submitted = True
    v35_ref = "58464539"
    v35_score = None
    v36_pushed = False
    v36_submitted = False
    v36_ref = None
    v36_score = None

    start_loop = time.time()
    
    while True:
        # Check Version 34 score if pending
        if v34_score is None:
            st, sc = get_submission_status(REF_V34)
            log(f"Version 34 (Ref {REF_V34}) status: {st}, score: {sc}")
            if st == "SubmissionStatus.COMPLETE" and sc is not None:
                v34_score = sc
                log(f"🎉 VERSION 34 EVALUATION COMPLETE! Public Score: {sc}")
                sync_git(f"docs(score): record Version 34 public score {sc} on hidden test")
            elif st == "SubmissionStatus.ERROR":
                v34_score = "ERROR"
                log("❌ Version 34 evaluated with error.")

        # Check Version 35 Kernel Status
        if not v35_submitted:
            kst = get_kernel_status()
            log(f"Version 35 Kernel status: {kst}")
            if "COMPLETE" in kst:
                log("✓ Version 35 Kernel Commit COMPLETE on Kaggle cloud GPU!")
                cur_v = get_current_kernel_version()
                log(f"Confirmed Version Number: {cur_v}")
                ref = submit_kernel_version(cur_v, DESC_V35)
                if ref:
                    v35_submitted = True
                    v35_ref = ref
                    log(f"🎉 VERSION 35 SUBMISSION DISPATCHED! Ref: {ref}")
                    sync_git(f"feat(phase-32): submit Version 35 Ref {ref} to competition")
            elif "FAILED" in kst or "ERROR" in kst:
                log(f"❌ Version 35 Kernel commit failed: {kst}")
                v35_submitted = True # Stop retrying

        # If Version 35 submitted, check its evaluation score
        if v35_submitted and v35_ref and v35_score is None:
            st, sc = get_submission_status(v35_ref)
            log(f"Version 35 (Ref {v35_ref}) status: {st}, score: {sc}")
            if st == "SubmissionStatus.COMPLETE" and sc is not None:
                v35_score = sc
                log(f"🎉 VERSION 35 EVALUATION COMPLETE! Public Score: {sc}")
                sync_git(f"docs(score): record Version 35 public score {sc} on hidden test")

        # Once Version 35 commit is complete (or submitted), deploy Version 36
        if v35_submitted and not v36_pushed:
            log("=" * 80)
            log("🚀 PUSHING VERSION 36 KERNEL TO KAGGLE (deploy_kernel_v36)...")
            log("=" * 80)
            push_res = run_cmd("kaggle kernels push -p deploy_kernel_v36")
            log(f"V36 Push stdout: {push_res.stdout.strip()}")
            if push_res.returncode == 0:
                v36_pushed = True
                log("✓ Version 36 Kernel successfully pushed! Monitoring commit...")
                sync_git("feat(phase-33): push Version 36 kernel to Kaggle cloud GPU")
            else:
                log(f"❌ Version 36 Push failed: {push_res.stderr}")

        # Check Version 36 Kernel Status and submit
        if v36_pushed and not v36_submitted:
            kst = get_kernel_status()
            log(f"Version 36 Kernel status: {kst}")
            if "COMPLETE" in kst:
                log("✓ Version 36 Kernel Commit COMPLETE on Kaggle cloud GPU!")
                cur_v = get_current_kernel_version()
                log(f"Confirmed Version Number: {cur_v}")
                ref = submit_kernel_version(cur_v, DESC_V36)
                if ref:
                    v36_submitted = True
                    v36_ref = ref
                    log(f"🎉 VERSION 36 SUBMISSION DISPATCHED! Ref: {ref}")
                    sync_git(f"feat(phase-33): submit Version 36 Ref {ref} to competition")
            elif "FAILED" in kst or "ERROR" in kst:
                log(f"❌ Version 36 Kernel commit failed: {kst}")
                v36_submitted = True

        # If Version 36 submitted, check its score
        if v36_submitted and v36_ref and v36_score is None:
            st, sc = get_submission_status(v36_ref)
            log(f"Version 36 (Ref {v36_ref}) status: {st}, score: {sc}")
            if st == "SubmissionStatus.COMPLETE" and sc is not None:
                v36_score = sc
                log(f"🎉 VERSION 36 EVALUATION COMPLETE! Public Score: {sc}")
                sync_git(f"docs(score): record Version 36 public score {sc} on hidden test")

        # Check if all 3 versions are scored
        if v34_score is not None and v35_score is not None and v36_score is not None:
            log("=" * 80)
            log("🏆 ALL 3 VERSIONS EVALUATED AND SCORED SUCCESSFULLY!")
            log(f"  * Version 34 Score: {v34_score}")
            log(f"  * Version 35 Score: {v35_score}")
            log(f"  * Version 36 Score: {v36_score}")
            log("=" * 80)
            break

        time.sleep(30)

if __name__ == "__main__":
    main()
