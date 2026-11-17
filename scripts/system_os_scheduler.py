#!/usr/bin/env python3
"""
scripts/system_os_scheduler.py
=============================================================================
CASMI 2026: Independent OS Background Scheduler & Hidden-Test Submitter
Runs entirely in the Linux host OS (PC system, completely detached from AGY).

Key Timers Managed Directly on the PC:
  1. Timer 1 (Pre-run Countdown Timer):
     - Monitors exact UTC time until 00:00:15 UTC (2026-11-18 daily quota reset).
     - Periodically checks Kaggle API for quota availability.
  2. Timer 2 (Kaggle Cloud Kernel Execution Timer):
     - Automatically pushes the version (v37, v38, v39).
     - Polls `kaggle kernels status` every 30s until `KernelWorkerStatus.COMPLETE`.
  3. Timer 3 (Hidden Cases Submission Timer):
     - Immediately upon kernel completion, fetches the generated kernel version number.
     - Automatically triggers `kaggle competitions submit -c ... -k ... -v ... -m "..."`.
  4. Timer 4 (Hidden Test Evaluation Score Tracker):
     - Polls `kaggle competitions submissions` until `SubmissionStatus.COMPLETE`.
     - Logs the verified public score and moves to the next version in sequence.

Logging: logs/system_scheduler.log
PID File: logs/system_scheduler.pid
Status: logs/system_scheduler_status.json
=============================================================================
"""

import os
import sys
import time
import json
import signal
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Enforce competition proxy
PROXY = "http://176.111.37.5:39811"
os.environ["https_proxy"] = PROXY
os.environ["http_proxy"] = PROXY
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
LOG_FILE = ROOT / "logs" / "system_scheduler.log"
PID_FILE = ROOT / "logs" / "system_scheduler.pid"
STATUS_FILE = ROOT / "logs" / "system_scheduler_status.json"

KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"

VERSIONS_QUEUE = [
    {
        "version_tag": "v37",
        "deploy_dir": ROOT / "deploy_kernel_v37",
        "label": "Version 37: Sovereign Regime B Rank-1 Promotion (0.75 w_top) + High-Res Isotope Engine + Pure Shield (Target LB 0.515+)",
        "target_lb": "0.515+"
    },
    {
        "version_tag": "v38",
        "deploy_dir": ROOT / "deploy_kernel_v38",
        "label": "Version 38: Regime B Rank-1 Promotion + Active Regime C Regioisomer Substitution (Ranks 20-25) + Pure Shield (Target LB 0.525+)",
        "target_lb": "0.525+"
    },
    {
        "version_tag": "v39",
        "deploy_dir": ROOT / "deploy_kernel_v39",
        "label": "Version 39: Sovereign Tri-Regime Frontier + Molecular Network Promotion + Pure Shield (Target LB 0.535+)",
        "target_lb": "0.535+"
    }
]

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_status(state: dict):
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def run_cmd(cmd: str, timeout_sec=300):
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
        log(f"⚠️ Command timed out ({timeout_sec}s): {cmd}")
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr="Timeout")

def get_kernel_version_number():
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
        ver = getattr(res.metadata, "current_version_number", getattr(res.metadata, "currentVersionNumber", None))
        if ver is not None:
            return int(ver)
    except Exception as e:
        log(f"Note: API kernel version query ({e})")
    return None

def check_quota_available():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        subs = api.competition_submissions(COMPETITION)
        today_utc = datetime.now(timezone.utc).date()
        today_subs = [s for s in subs if getattr(s, "date", datetime.now(timezone.utc)).date() == today_utc]
        rem = max(0, 5 - len(today_subs))
        return rem
    except Exception as e:
        log(f"Quota check note: {e}")
        return 0

def wait_until_quota_reset():
    log("=" * 80)
    log("⏳ [TIMER 1: PC COUNTDOWN TIMER] Initiating background wait until 00:00 UTC quota refresh...")
    log("=" * 80)
    
    while True:
        now_utc = datetime.now(timezone.utc)
        target_utc = datetime(now_utc.year, now_utc.month, now_utc.day, 0, 0, 20, tzinfo=timezone.utc)
        if now_utc >= target_utc:
            target_utc += timedelta(days=1)
        
        remaining_seconds = (target_utc - now_utc).total_seconds()
        
        # Check if already past 00:00:15 UTC or if submissions available
        if remaining_seconds > 86300 or remaining_seconds <= 10:
            log("🎉 Quota reset threshold reached (00:00 UTC)! Proceeding to version deployment.")
            break
        
        hours, rem = divmod(int(remaining_seconds), 3600)
        minutes, seconds = divmod(rem, 60)
        
        write_status({
            "phase": "TIMER_1_COUNTDOWN",
            "target_utc": target_utc.isoformat(),
            "remaining_seconds": int(remaining_seconds),
            "remaining_formatted": f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
        })
        
        if int(remaining_seconds) % 300 < 30:
            log(f"  * [Timer 1 Countdown]: {hours:02d}h {minutes:02d}m {seconds:02d}s remaining until daily quota reset (00:00 UTC).")
        
        time.sleep(30)

def deploy_version(v_info: dict):
    v_tag = v_info["version_tag"]
    deploy_dir = v_info["deploy_dir"]
    label = v_info["label"]
    
    log("\n" + "=" * 80)
    log(f"🚀 [DEPLOYING {v_tag.upper()}]: {label}")
    log(f"   Directory: {deploy_dir}")
    log("=" * 80)
    
    write_status({
        "phase": f"DEPLOYING_{v_tag.upper()}",
        "version_tag": v_tag,
        "label": label
    })
    
    # 1. Push kernel
    push_res = run_cmd(f"kaggle kernels push -p {deploy_dir}")
    log(f"Kernel push stdout: {push_res.stdout.strip()}")
    if push_res.stderr.strip():
        log(f"Kernel push stderr: {push_res.stderr.strip()}")
    if push_res.returncode != 0:
        log(f"❌ Kernel push failed for {v_tag}!")
        return False
    
    log(f"✓ Kernel push succeeded for {v_tag}. Waiting for execution on Kaggle GPU...")
    time.sleep(15)
    
    # 2. Timer 2: Wait for kernel commit to finish running
    log("\n⏳ [TIMER 2: KAGGLE COMMIT RUNNER] Monitoring cloud worker execution...")
    start_commit = time.time()
    commit_ok = False
    
    while time.time() - start_commit < 5400: # 90 minutes max
        st_res = run_cmd(f"kaggle kernels status {KERNEL_SLUG}")
        st_out = (st_res.stdout + " " + st_res.stderr).strip()
        elapsed_m = (time.time() - start_commit) / 60.0
        
        write_status({
            "phase": f"RUNNING_COMMIT_{v_tag.upper()}",
            "elapsed_minutes": round(elapsed_m, 1),
            "status": st_out
        })
        
        if int(time.time() - start_commit) % 60 < 15:
            log(f"  * [Commit Runner Timer]: {st_out} (elapsed {elapsed_m:.1f}m)")
            
        if "KernelWorkerStatus.COMPLETE" in st_out or "COMPLETE" in st_out:
            log(f"🎉 Kernel commit completed successfully for {v_tag} in {elapsed_m:.1f}m!")
            commit_ok = True
            break
        elif "KernelWorkerStatus.ERROR" in st_out or "ERROR" in st_out or "FAILED" in st_out:
            log(f"❌ Kernel commit failed on Kaggle: {st_out}")
            return False
            
        time.sleep(15)
        
    if not commit_ok:
        log("❌ Kernel commit timed out after 90m.")
        return False
        
    # Determine the kernel version number
    k_ver = get_kernel_version_number()
    log(f"✓ Kernel version number on Kaggle: {k_ver}")
    
    # 3. Timer 3: Submit for Hidden Cases
    log("\n📥 [TIMER 3: SUBMITTING HIDDEN CASES] Triggering hidden test evaluation...")
    sub_cmd = f'kaggle competitions submit -c {COMPETITION} -k {KERNEL_SLUG} '
    if k_ver is not None:
        sub_cmd += f'-v {k_ver} '
    sub_cmd += f'-m "{label}"'
    
    sub_res = run_cmd(sub_cmd)
    log(f"Submission response: {sub_res.stdout.strip()}")
    if sub_res.stderr.strip():
        log(f"Submission stderr: {sub_res.stderr.strip()}")
        
    time.sleep(20)
    
    # 4. Timer 4: Track Hidden Evaluation Score
    log("\n📊 [TIMER 4: TRACKING HIDDEN EVALUATION SCORE] Awaiting Kaggle scoring pass...")
    eval_start = time.time()
    score_found = None
    
    while time.time() - eval_start < 7200: # 120 minutes max
        chk_res = run_cmd(f"kaggle competitions submissions -c {COMPETITION}")
        lines = chk_res.stdout.strip().splitlines()
        
        if len(lines) > 2:
            top_line = lines[2]
            parts = top_line.split()
            # ref is parts[0], status is parts[-3] or parts[-2], score is parts[-2] or parts[-1]
            elapsed_e = (time.time() - eval_start) / 60.0
            
            write_status({
                "phase": f"SCORING_{v_tag.upper()}",
                "elapsed_minutes": round(elapsed_e, 1),
                "latest_line": top_line[:120]
            })
            
            if "SubmissionStatus.COMPLETE" in top_line:
                # Extract public score
                try:
                    for tok in reversed(parts):
                        try:
                            val = float(tok)
                            score_found = val
                            break
                        except ValueError:
                            pass
                except Exception:
                    pass
                log(f"🎉 [EVALUATION COMPLETE]: {v_tag} Public Score = {score_found}!")
                break
            elif "SubmissionStatus.ERROR" in top_line or "ERROR" in top_line:
                log(f"❌ Submission evaluation returned error: {top_line}")
                break
                
        time.sleep(25)
        
    return True

def main():
    PID_FILE.write_text(str(os.getpid()))
    log("=" * 80)
    log(f"🤖 CASMI 2026 SYSTEM BACKGROUND SCHEDULER ACTIVE (PID {os.getpid()})")
    log("   Running directly on PC host OS (outside AGY session).")
    log("=" * 80)
    
    # Step 1: Wait until 00:00 UTC quota refresh
    wait_until_quota_reset()
    
    # Step 2: Sequentially deploy and submit versions
    for v_info in VERSIONS_QUEUE:
        log(f"\nProceeding to execute queue item: {v_info['version_tag']}")
        success = deploy_version(v_info)
        if success:
            log(f"✓ Queue item {v_info['version_tag']} finished. Waiting 90s before next version...")
            time.sleep(90)
        else:
            log(f"⚠️ Queue item {v_info['version_tag']} encountered an issue. Waiting 180s...")
            time.sleep(180)
            
    log("\n🏁 ALL STAGED VERSIONS SUCCESSFULLY PROCESSED ON KAGGLE HIDDEN TEST SET!")
    write_status({"phase": "ALL_VERSIONS_COMPLETE"})

if __name__ == "__main__":
    main()
