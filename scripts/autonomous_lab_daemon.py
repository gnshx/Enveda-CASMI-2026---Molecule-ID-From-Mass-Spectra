#!/usr/bin/env python3
"""
scripts/autonomous_lab_daemon.py
=============================================================================
24/7 CASMI 2026 Autonomous Research Lab Master Daemon.
=============================================================================
Continuous 24/7 background operation:
1. Submission & Leaderboard Telemetry:
   - Continuous polling of active Kaggle submissions (Ref 58442788 V31+).
   - Automatic recording of verified scores to PROJECT_STATUS.md, EXPERIMENT_LOG.md,
     DAILY_JOURNEY.md, and ROADMAP_AND_PROGRESS.md.
   - Automated git commit and push to origin/main upon score completion.
2. Hardware & Resource Telemetry Watchdog:
   - Continuous polling of NVIDIA GeForce RTX 3060 (12GB VRAM) and system load.
   - Structured JSON telemetry logging to logs/telemetry.json.
3. Autonomous Scientific Validation Loop:
   - Periodic offline benchmark sweeps across Enveda-180 rehearsal candidates.
   - Validating candidate reranking weights and bond-cleavage consistency filters.
4. Swarm State Synchronization:
   - Maintains logs/swarm_state.json for multi-agent coordination across the team.
=============================================================================
"""

import os
import sys
import time
import json
import signal
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Set proxy environment variables for all network calls
PROXY_URL = "http://176.111.37.5:39811"
os.environ["https_proxy"] = PROXY_URL
os.environ["http_proxy"] = PROXY_URL
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL

REPO_ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
sys.path.insert(0, str(REPO_ROOT))
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DAEMON_LOG = LOGS_DIR / "lab_daemon.log"
TELEMETRY_FILE = LOGS_DIR / "telemetry.json"
SWARM_STATE_FILE = LOGS_DIR / "swarm_state.json"
PID_FILE = LOGS_DIR / "lab_daemon.pid"

COMPETITION_SLUG = "enveda-CASMI26-molecule-id-mass-spectra"
POLL_INTERVAL_SEC = 45
TELEMETRY_INTERVAL_SEC = 120
VALIDATION_INTERVAL_SEC = 7200  # 2 hours

def log(msg: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] [DAEMON-24x7] {msg}"
    print(line, flush=True)
    with open(DAEMON_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_cmd(cmd: str, cwd=None, timeout_sec=60) -> tuple[int, str]:
    try:
        env = os.environ.copy()
        # Unset proxy for git commands so GitHub connects directly
        if "git" in cmd:
            env.pop("https_proxy", None)
            env.pop("http_proxy", None)
            env.pop("HTTP_PROXY", None)
            env.pop("HTTPS_PROXY", None)
        res = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(cwd or REPO_ROOT),
            env=env,
            timeout=timeout_sec
        )
        output = (res.stdout + "\n" + res.stderr).strip()
        return res.returncode, output
    except subprocess.TimeoutExpired:
        log(f"⚠️ Command timed out ({timeout_sec}s): {cmd}")
        return -1, "TIMEOUT"
    except Exception as e:
        log(f"⚠️ Command exception: {e}")
        return -1, str(e)

def update_telemetry():
    """Poll hardware and disk resources."""
    telemetry: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {},
        "system": {}
    }
    
    # 1. GPU Telemetry via nvidia-smi
    code, out = run_cmd("nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits")
    if code == 0 and out:
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        if len(parts) >= 6:
            telemetry["gpu"] = {
                "name": parts[0],
                "temp_c": int(parts[1]),
                "utilization_gpu_pct": int(parts[2]),
                "utilization_mem_pct": int(parts[3]),
                "mem_used_mb": int(parts[4]),
                "mem_total_mb": int(parts[5]),
                "mem_free_mb": int(parts[5]) - int(parts[4])
            }
            
    # 2. Disk & System Telemetry
    disk = shutil.disk_usage(REPO_ROOT)
    telemetry["system"] = {
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
    }
    
    with open(TELEMETRY_FILE, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

def get_kaggle_api(retries=3, delay=3):
    from kaggle.api.kaggle_api_extended import KaggleApi
    for attempt in range(retries):
        try:
            api = KaggleApi()
            api.authenticate()
            return api
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(delay)

def record_verified_score(v_name: str, desc: str, score: float, ref: int):
    """Update all markdown ledgers and commit to git."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log(f"🏆 CELEBRATION: Official Score Verified for {v_name} (Ref {ref}): {score:.4f}!")
    
    # 0. Write atomic score event for swarm notification
    event_file = LOGS_DIR / "last_score_event.json"
    event_data = {
        "timestamp_utc": now_utc,
        "version": v_name,
        "ref": ref,
        "score": score,
        "description": desc,
        "status": "COMPLETE"
    }
    with open(event_file, "w", encoding="utf-8") as f:
        json.dump(event_data, f, indent=2)

    # 1. Update PROJECT_STATUS.md
    status_file = REPO_ROOT / "PROJECT_STATUS.md"
    if status_file.exists():
        content = status_file.read_text(encoding="utf-8")
        if str(ref) in content:
            content = content.replace("**`PENDING`**", f"**`{score:.4f}`**")
            content = content.replace("`PENDING`", f"**`{score:.4f}`**")
            content = content.replace("`SubmissionStatus.PENDING`", f"`SubmissionStatus.COMPLETE` (**`{score:.4f}`**)")
            content = content.replace("SubmissionStatus.PENDING", f"SubmissionStatus.COMPLETE ({score:.4f})")
            if score > 0.4800:
                old_pb_line = "| **Our Verified PB** | **Version 24 (Ref `57982390`)** | **`0.4800`** | Dual-Neural Z-Score Ensemble (Saner v1 + Scorer v2) + MIDX Dedup |"
                new_pb_line = f"| **Our Verified PB** | **{v_name} (Ref `{ref}`)** | **`{score:.4f}`** | {desc} |"
                if old_pb_line in content:
                    content = content.replace(old_pb_line, new_pb_line)
            status_file.write_text(content, encoding="utf-8")
            log("Updated PROJECT_STATUS.md with verified score.")
            
    # 2. Update EXPERIMENT_LOG.md
    exp_file = REPO_ROOT / "EXPERIMENT_LOG.md"
    if exp_file.exists():
        content = exp_file.read_text(encoding="utf-8")
        if str(ref) in content:
            content = content.replace("| **Pending** | **PENDING (Active)** |", f"| **`{score:.4f}`** | **COMPLETE** |")
            content = content.replace("**Pending**", f"**`{score:.4f}`**")
            content = content.replace("**PENDING (Active)**", "**COMPLETE**")
            if score > 0.4800:
                content = content.replace(
                    "**Current Personal Best**: `0.4800` (Version 24, Ref `57982390`)",
                    f"**Current Personal Best**: `{score:.4f}` ({v_name}, Ref `{ref}`)"
                )
            exp_file.write_text(content, encoding="utf-8")
            log("Updated EXPERIMENT_LOG.md with verified score.")
            
    # 3. Update DAILY_JOURNEY.md
    journey_file = REPO_ROOT / "progress" / "DAILY_JOURNEY.md"
    if journey_file.exists():
        entry = f"""
### Official Score Verified: {v_name}
- **Timestamp**: {now_utc}
- **Submission Ref**: `{ref}`
- **Description**: {desc}
- **Official Public Score**: **`{score:.4f}`**
- **Status**: Complete & Verified on Public Leaderboard
"""
        with open(journey_file, "a", encoding="utf-8") as f:
            f.write(entry)
        log("Appended entry to progress/DAILY_JOURNEY.md.")

    # 4. Update ROADMAP_AND_PROGRESS.md
    roadmap_file = REPO_ROOT / "progress" / "ROADMAP_AND_PROGRESS.md"
    if roadmap_file.exists():
        r_content = roadmap_file.read_text(encoding="utf-8")
        target_str = f"Version 31** (Ref `{ref}`): Apex Sovereign Frontier (N-1c Molecular Network + ExtLib TAU=0.80 + Learned Logistic Reranker) -> **Target LB 0.488 - 0.492+** [DEPLOYED & SCORING]"
        if target_str in r_content:
            new_r_str = f"Version 31** (Ref `{ref}`): Apex Sovereign Frontier -> **Verified Public LB: {score:.4f}** [COMPLETE]"
            r_content = r_content.replace(target_str, new_r_str)
        row = f"| {now_utc[:10]} | {v_name} (Ref {ref}) | {desc} | **{score:.4f}** | Succeeded |\n"
        if f"Ref {ref}" not in r_content:
            r_content += row
        roadmap_file.write_text(r_content, encoding="utf-8")
        log("Updated progress/ROADMAP_AND_PROGRESS.md with verified score.")
        
    # 5. Git Commit and Push
    run_cmd("git add -A")
    run_cmd(f'git commit -m "docs: record verified public score {score:.4f} for {v_name} (Ref {ref})"')
    code, push_out = run_cmd("git push origin main")
    log(f"Git push result: {push_out[:120]}")

def check_kaggle_submissions(seen_completed: set, zombie_refs: set):
    """Check competition submission queue."""
    try:
        api = get_kaggle_api()
        subs = api.competition_submissions(COMPETITION_SLUG)
        if not subs:
            return
            
        active_found = False
        for sub in subs[:5]:
            ref = getattr(sub, 'ref', None)
            if not ref or ref in zombie_refs:
                continue
                
            status = str(getattr(sub, 'status', ''))
            score_str = str(getattr(sub, 'public_score', getattr(sub, 'publicScore', '')) or '').strip()
            desc = getattr(sub, 'description', '')
            v_name = f"Ref {ref}"
            if "Version " in desc:
                v_name = desc.split(":")[0].strip()
                
            if "COMPLETE" in status and score_str and ref not in seen_completed:
                try:
                    score = float(score_str)
                    record_verified_score(v_name, desc, score, ref)
                    seen_completed.add(ref)
                except ValueError:
                    pass
            elif "PENDING" in status:
                active_found = True
                log(f"Active Submission: {v_name} (Ref {ref}) is actively computing on Kaggle GPU...")
            elif ("ERROR" in status or "FAILED" in status) and ref not in seen_completed:
                err_desc = getattr(sub, 'errorDescription', 'Evaluation failed')
                log(f"⚠️ Submission {v_name} (Ref {ref}) reported error: {err_desc}")
                seen_completed.add(ref)
                
        # Calculate daily submission quota (5 per UTC day)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        subs_today = sum(1 for s in subs if getattr(s, 'date', None) and str(s.date).startswith(today_str))
        quota_remaining = max(0, 5 - subs_today)

        # Write swarm state file
        swarm_state = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "active_submission": active_found,
            "latest_sub_ref": getattr(subs[0], 'ref', None) if subs else None,
            "latest_status": str(getattr(subs[0], 'status', None)) if subs else None,
            "latest_score": str(getattr(subs[0], 'public_score', None)) if subs else None,
            "submissions_today": subs_today,
            "quota_remaining": quota_remaining,
            "quota_total": 5,
            "daemon_alive": True
        }
        with open(SWARM_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(swarm_state, f, indent=2)
            
    except Exception as e:
        log(f"Exception checking Kaggle API: {e}")

def run_background_validation_sweep():
    """Execute local offline validation sweep to discover optimal candidate weights."""
    benchmark_path = REPO_ROOT / "data" / "benchmark" / "main_candidates.parquet"
    if not benchmark_path.exists():
        return
        
    try:
        from validation.metrics import mrr_at_25
        import pandas as pd
        import numpy as np
        
        df = pd.read_parquet(benchmark_path)
        log(f"🔬 Running periodic validation sweep across {len(df)} candidate rows...")
        
        # Test candidate popularity and spectral gating grid
        best_cfg = None
        best_mrr = -1.0
        
        for pop_w in [0.10, 0.15, 0.20, 0.25]:
            for np_w in [0.10, 0.15, 0.20]:
                preds, gt = {}, {}
                for mol, g in df.groupby("molecule", observed=True):
                    gt[str(mol)] = str(mol)
                    base_term = 1.0 / (3.0 + g["rank"])
                    scores = g["ranker_score"].to_numpy(dtype=float)
                    z_score = (scores - scores.mean()) / (scores.std() + 1e-6)
                    comp = base_term + 0.75 * z_score + pop_w * (g["popularity"] / 20.0) + np_w * (g["pool_source"] > 0)
                    g_temp = g.assign(comp_score=comp).sort_values("comp_score", ascending=False)
                    keys = g_temp["candidate_key"].tolist()
                    preds[str(mol)] = keys[:25]
                mrr = mrr_at_25(preds, gt)
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_cfg = {"pop_w": pop_w, "np_w": np_w, "val_mrr": round(float(mrr), 5)}
                    
        log(f"🔬 Validation sweep complete! Best config on benchmark: {best_cfg}")
        
        history_file = REPO_ROOT / "experiments" / "validation_sweep_best.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "best": best_cfg}, f, indent=2)
            
    except Exception as e:
        log(f"Validation sweep exception: {e}")

def daemon_main():
    # Write PID file
    pid = os.getpid()
    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(pid))
        
    log("=" * 70)
    log(f"🚀 CASMI 2026 24/7 AUTONOMOUS RESEARCH LAB MASTER DAEMON STARTED (PID {pid})")
    log("=" * 70)
    
    seen_completed = {57982390, 57983294, 57992067, 57992109, 58021007, 58168204, 58168293, 58442788, 58451032}
    zombie_refs = {58168204, 58168293}
    
    last_telemetry_time = 0.0
    last_validation_time = 0.0
    iteration = 0
    
    def handle_exit(signum, frame):
        log(f"Daemon received signal {signum}. Shutting down cleanly.")
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)
    
    try:
        while True:
            iteration += 1
            now = time.time()
            
            # 1. Telemetry Update (every 2 minutes)
            if now - last_telemetry_time >= TELEMETRY_INTERVAL_SEC:
                update_telemetry()
                last_telemetry_time = now
                
            # 2. Check Kaggle Submissions (every 45 seconds)
            check_kaggle_submissions(seen_completed, zombie_refs)
            
            # 3. Validation Sweep (every 2 hours)
            if now - last_validation_time >= VALIDATION_INTERVAL_SEC:
                run_background_validation_sweep()
                last_validation_time = now
                
            time.sleep(POLL_INTERVAL_SEC)
            
    except Exception as e:
        log(f"Fatal exception in daemon loop: {e}")
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)

if __name__ == "__main__":
    daemon_main()
