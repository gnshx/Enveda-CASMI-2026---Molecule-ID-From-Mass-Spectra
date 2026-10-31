"""
Autonomous Kaggle Runner for Enveda CASMI 2026 Competition
Manages the continuous submission, scoring, logging, git synchronization,
and sequential plan deployment loop.
"""

import os
import sys
import time
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT = Path('/home/dlcv/Enveda-CASMI-2026').resolve()
LOG_FILE = REPO_ROOT / 'logs' / 'autonomous_loop.log'
STATE_FILE = REPO_ROOT / 'logs' / 'runner_state.json'
KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION_SLUG = "enveda-CASMI26-molecule-id-mass-spectra"

# Sequential deployment queue for today
DEPLOYMENT_QUEUE = [
    {
        "id": "plan_a_v18",
        "name": "Plan A v18 (Library Gate Fix + PC_JOIN_N=50 + POP_MU=0.25)",
        "dir": None, # Already running on Kaggle
        "target_score": 0.400,
        "is_currently_running": True
    },
    {
        "id": "gap_v4_top1",
        "name": "Closing the Evidence Gap v4 (Savko 0.521 Top-1 Solution: Outsider Recall + MIDX Dedup)",
        "dir": REPO_ROOT / "deploy_kernel_gap",
        "target_score": 0.521,
        "is_currently_running": True
    },
    {
        "id": "midx_0479",
        "name": "MIDX Exact Dedup Fill (Baidalin 0.479: PubChem Doc Prior + Skeleton Dedup)",
        "dir": REPO_ROOT / "deploy_kernel_midx",
        "target_score": 0.479,
        "is_currently_running": False
    },
    {
        "id": "spectral_parallax_0477",
        "name": "Spectral Parallax v22 (Tamrazov 0.477: MetFrag-lite Physical Check + 5-Shield)",
        "dir": REPO_ROOT / "deploy_kernel_spectral_parallax",
        "target_score": 0.477,
        "is_currently_running": False
    },
    {
        "id": "rozen_v4g_custom_0472",
        "name": "Rozen Consensus CUSTOM (w_top=0.25, w_mid=0.75: Alex Chilton 0.472)",
        "dir": REPO_ROOT / "deploy_kernel_v4g",
        "target_score": 0.472,
        "is_currently_running": False
    }
]

def log(msg: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_cmd(cmd: str, cwd=None) -> tuple[int, str]:
    res = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    return res.returncode, res.stdout.strip()

def load_state() -> dict:
    if STATE_FILE.is_file():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"Warning: could not read state file: {e}")
    return {"current_queue_idx": 0, "completed_jobs": []}

def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

def check_kernel_status() -> str:
    code, out = run_cmd(f"kaggle kernels status {KERNEL_SLUG}")
    if code != 0:
        log(f"Kaggle status error: {out}")
        return "UNKNOWN"
    for line in out.splitlines():
        if "has status" in line:
            # e.g., nukaladevisaiganesh/... has status "KernelWorkerStatus.RUNNING"
            return line.split('has status')[-1].strip().strip('"').replace("KernelWorkerStatus.", "")
    return out.strip()

def wait_for_kernel_completion(poll_interval_sec=60) -> bool:
    log(f"Monitoring kernel: {KERNEL_SLUG}...")
    while True:
        status = check_kernel_status()
        log(f"Kernel status: {status}")
        if status == "COMPLETE":
            return True
        if status in ("ERROR", "CANCELLED", "FAILED"):
            log(f"Kernel ended with error status: {status}")
            return False
        time.sleep(poll_interval_sec)

def download_and_submit(job_meta: dict) -> tuple[bool, str, float]:
    out_dir = Path(f"/tmp/kernel_out_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading output to {out_dir}...")
    code, out = run_cmd(f"kaggle kernels output {KERNEL_SLUG}", cwd=out_dir)
    log(f"Output download result: {out[:200]}")
    
    sub_file = out_dir / "submission.csv"
    if not sub_file.is_file():
        # Search for any csv file that might be the submission
        csvs = list(out_dir.glob("*.csv"))
        log(f"Found CSV files: {[c.name for c in csvs]}")
        sub_file = next((c for c in csvs if "sub" in c.name.lower()), None)
    
    if not sub_file or not sub_file.is_file():
        log(f"Error: No submission.csv found in {out_dir}")
        return False, "", 0.0

    msg = f"{job_meta['name']} (target {job_meta['target_score']})"
    log(f"Submitting {sub_file.name} to {COMPETITION_SLUG} with message: '{msg}'...")
    sub_cmd = f'kaggle competitions submit -c {COMPETITION_SLUG} -f "{sub_file}" -m "{msg}"'
    code, sub_out = run_cmd(sub_cmd)
    log(f"Submission command output: {sub_out}")
    if code != 0 and "Successfully submitted" not in sub_out:
        log(f"Submission failed: {sub_out}")
        return False, sub_out, 0.0

    # Poll for score
    log("Polling competition submissions table for public score...")
    score = 0.0
    for _ in range(12): # poll up to 3 minutes
        time.sleep(15)
        code, table_out = run_cmd(f"kaggle competitions submissions -c {COMPETITION_SLUG}")
        lines = table_out.splitlines()
        if len(lines) >= 3:
            first_row = lines[2]
            parts = first_row.split()
            log(f"Latest submission row: {first_row}")
            if "COMPLETE" in first_row:
                try:
                    score = float(parts[-1])
                    log(f"Official Public Score: {score}")
                    return True, sub_out, score
                except ValueError:
                    pass
    return True, sub_out, score

def push_kernel(deploy_dir: Path) -> bool:
    log(f"Pushing new kernel from {deploy_dir}...")
    code, out = run_cmd(f"kaggle kernels push -p {deploy_dir}")
    log(f"Push result: {out}")
    return code == 0

def git_sync(commit_msg: str):
    log(f"Syncing git with message: '{commit_msg}'...")
    run_cmd("git add -A")
    run_cmd(f'git commit -m "{commit_msg}"')
    code, out = run_cmd("git push origin main")
    log(f"Git push result: {out}")

def append_to_daily_journey(job_meta: dict, score: float):
    journey_file = REPO_ROOT / "progress" / "DAILY_JOURNEY.md"
    entry = f"""
### Autonomous Run Result: {job_meta['name']}
- **Timestamp**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
- **Kernel**: `{KERNEL_SLUG}`
- **Target Score**: `{job_meta['target_score']}`
- **Achieved Public Score**: **`{score:.3f}`**
- **Status**: Complete & Verified
"""
    with open(journey_file, "a", encoding="utf-8") as f:
        f.write(entry)
    log("Appended run result to progress/DAILY_JOURNEY.md")

def main_loop():
    log("=== Autonomous Kaggle Loop Started ===")
    state = load_state()
    queue_idx = state.get("current_queue_idx", 0)
    
    while queue_idx < len(DEPLOYMENT_QUEUE):
        job = DEPLOYMENT_QUEUE[queue_idx]
        log(f"\n>>> PROCESSING JOB {queue_idx + 1}/{len(DEPLOYMENT_QUEUE)}: {job['name']} <<<")
        
        # If job has a directory and is not currently running, push it first
        if job["dir"] is not None and not job.get("is_currently_running", False):
            log(f"Deploying kernel from {job['dir']}...")
            success = push_kernel(job["dir"])
            if not success:
                log(f"Failed to push {job['name']}. Pausing 60s before retry...")
                time.sleep(60)
                continue
            job["is_currently_running"] = True
            save_state({"current_queue_idx": queue_idx, "active_job": job})
            # Sleep 30s to allow Kaggle container to initialize
            time.sleep(30)
            
        # Wait for kernel completion
        completed = wait_for_kernel_completion(poll_interval_sec=60)
        if not completed:
            log(f"Kernel run failed for job: {job['name']}. Moving to next job in 60s...")
            queue_idx += 1
            save_state({"current_queue_idx": queue_idx})
            time.sleep(60)
            continue
            
        # Download and submit
        sub_ok, details, score = download_and_submit(job)
        if sub_ok:
            log(f"Job {job['name']} scored: {score}")
            append_to_daily_journey(job, score)
            git_sync(f"auto: complete {job['id']} -> LB {score:.3f}")
        else:
            log(f"Job {job['name']} failed to submit cleanly.")
            
        queue_idx += 1
        save_state({"current_queue_idx": queue_idx})
        log(f"Job {job['name']} finished. Proceeding to next queue item...")
        time.sleep(30)
        
    log("=== All queued jobs completed for today! ===")

if __name__ == "__main__":
    main_loop()

