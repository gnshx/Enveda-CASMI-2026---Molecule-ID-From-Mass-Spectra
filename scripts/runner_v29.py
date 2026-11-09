#!/usr/bin/env python3
"""
scripts/runner_v29.py
Autonomous runner for Version 29:
1. Pushes deploy_kernel_v29 to Kaggle.
2. Monitors kernel run until complete on GPU Tesla T4.
3. Submits kernel output to Kaggle competition.
4. Monitors submission status until public score is released.
5. Updates DAILY_JOURNEY.md and ROADMAP_AND_PROGRESS.md.
6. Commits and pushes to GitHub.
"""

import os, sys, time, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
LOG_PATH = ROOT / "logs" / "runner_v29.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

PROXY = "http://176.111.37.5:39811"
ENV = os.environ.copy()
ENV["https_proxy"] = PROXY
ENV["http_proxy"] = PROXY

KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
VERSION_LABEL = "Version 29"
VERSION_NUM = 29
MESSAGE = "Version 29: Apex Sovereign Dual-Tier ExtLib (Gold/Silver) + timsTOF Mass Prior + v5 Scorer Ensemble (Target LB 0.487+)"

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
    log(f"Pushing {VERSION_LABEL} to Kaggle from deploy_kernel_v29...")
    res = run_cmd("kaggle kernels push -p deploy_kernel_v29")
    log(f"Push result: {res.stdout.strip()}")
    if "successfully" not in res.stdout.lower() and "cache saved" not in res.stdout.lower() and res.returncode != 0:
        raise RuntimeError(f"Kernel push failed: {res.stderr}")
    return True

def wait_for_kernel(timeout_seconds=5400):
    log(f"Waiting for kernel '{KERNEL_SLUG}' compilation on Kaggle GPU...")
    start_time = time.time()
    poll_count = 0
    while time.time() - start_time < timeout_seconds:
        poll_count += 1
        res = run_cmd(f"kaggle kernels status {KERNEL_SLUG}", check=False)
        out = res.stdout.strip()
        log(f"Poll #{poll_count} ({int(time.time() - start_time)}s elapsed): {out}")
        
        if "KernelWorkerStatus.COMPLETE" in out or '"complete"' in out.lower():
            log("🎉 Kernel execution completed successfully on Kaggle!")
            return True
        elif "KernelWorkerStatus.ERROR" in out or '"error"' in out.lower():
            log(f"❌ Kernel failed with error:\n{out}")
            run_cmd(f"kaggle kernels output {KERNEL_SLUG} -p logs/kernel_v29_err")
            raise RuntimeError(f"Kernel run errored: {out}")
        elif "KernelWorkerStatus.CANCEL" in out or '"cancel"' in out.lower():
            raise RuntimeError("Kernel was cancelled.")
        
        time.sleep(25)
    raise TimeoutError(f"Kernel did not finish within {timeout_seconds} seconds.")

def submit_to_competition():
    log(f"Submitting {VERSION_LABEL} output to competition {COMPETITION}...")
    cmd = f'kaggle competitions submit -c {COMPETITION} -f submission.csv -k {KERNEL_SLUG} -m "{MESSAGE}"'
    res = run_cmd(cmd)
    log(f"Submission command output:\n{res.stdout.strip()}")
    if res.returncode != 0:
        log(f"Submission failed:\n{res.stderr.strip()}")
        raise RuntimeError("Submission failed.")
    return True

def monitor_submission(timeout_seconds=7200):
    log("Monitoring Kaggle competition submissions for public score...")
    start_time = time.time()
    poll_count = 0
    ref = None
    
    while time.time() - start_time < timeout_seconds:
        poll_count += 1
        res = run_cmd(f"kaggle competitions submissions -c {COMPETITION}", check=False)
        lines = res.stdout.strip().split("\n")
        
        for line in lines[2:]:
            parts = [p.strip() for p in line.split() if p.strip()]
            if len(parts) >= 6:
                r, fname, dt, desc_start = parts[0], parts[1], parts[2], parts[3]
                if fname == "submission.csv" and (ref is None or r == ref):
                    ref = r
                    status_part = [p for p in parts if "status" in p.lower() or "complete" in p.lower() or "pending" in p.lower()]
                    status = status_part[0] if status_part else "UNKNOWN"
                    score = ""
                    for p in parts:
                        try:
                            val = float(p)
                            if 0.0 < val < 1.0:
                                score = f"{val:.4f}"
                                break
                        except ValueError:
                            pass
                    
                    log(f"Poll #{poll_count}: {VERSION_LABEL} (Ref {ref}) Status={status}, Score={score}")
                    if "complete" in status.lower() and score:
                        log(f"🏆 SUCCESS: {VERSION_LABEL} (Ref {ref}) Scored {score}!")
                        return ref, float(score)
                    elif "error" in status.lower():
                        log(f"❌ Submission failed with status {status}")
                        return ref, None
                    break
        time.sleep(30)
    log("Submission monitoring timed out.")
    return ref, None

def update_documentation(ref, score):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    journey_path = ROOT / "progress" / "DAILY_JOURNEY.md"
    score_str = f"{score:.4f}" if score is not None else "Pending / Eval"
    
    entry = f"""
### Run Reference: {ref} — {VERSION_LABEL} (Official Public Score: **{score_str}**)
- **Date & Time**: {ts}
- **Configuration**:
  - Architecture: Apex Sovereign Dual-Tier ExtLib (Gold $\\ge 0.825$ $\\rightarrow$ Rank 1, Silver $\\ge 0.72$ $\\rightarrow$ Rank 2)
  - Precursor Instrument Prior: Bruker timsTOF mass accuracy centered at -1.5 ppm (`V8 = 'MASS'`)
  - Multi-Neural Ensemble: 200-epoch v5 Scorer (320-width) + Saner v1 via z-score fusion ($0.65 / 0.35$)
  - Prior Weights: `POP_W=0.12, SPEC_W=0.12, NP_W=0.12, LIB_HIT_GATE=0.75`
  - Zero Decoy Injection: `qwertykeypadapple/casmi26-lotus-candidates` removed from candidate pools
- **Public Leaderboard Score**: **{score_str}**
- **Outcome Assessment**: {'New personal best and leaderboard breakthrough!' if score and score > 0.480 else 'Evaluated on Kaggle test set.'}
"""
    if journey_path.is_file():
        text = journey_path.read_text()
        journey_path.write_text(text + entry)
        log(f"Logged {VERSION_LABEL} ({score_str}) to DAILY_JOURNEY.md")
    
    # Sync git
    run_cmd(f"git add deploy_kernel_v29/ scripts/prepare_v29.py scripts/runner_v29.py progress/DAILY_JOURNEY.md")
    commit_msg = f"feat: deploy {VERSION_LABEL} - score {score_str} (Ref {ref})"
    run_cmd(f'git commit -m "{commit_msg}"')
    push_res = run_cmd("git push origin main")
    log(f"Git push result: {push_res.stdout.strip()}")

def main():
    log(f"=== Starting Autonomous Runner for {VERSION_LABEL} ===")
    # Check current status before pushing
    st = run_cmd(f"kaggle kernels status {KERNEL_SLUG}", check=False).stdout
    if "KernelWorkerStatus.RUNNING" in st or "KernelWorkerStatus.QUEUED" in st:
        log(f"Kernel is already {st.strip()}; skipping push and monitoring run...")
    else:
        push_kernel()
    wait_for_kernel()
    submit_to_competition()
    ref, score = monitor_submission()
    update_documentation(ref, score)
    log(f"=== {VERSION_LABEL} Execution Complete ===")

if __name__ == "__main__":
    main()
