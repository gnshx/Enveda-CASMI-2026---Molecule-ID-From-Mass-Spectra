#!/usr/bin/env python3
"""
Continuous Score Poller for Submissions 57982390 (V24) and 57983294 (V25).
Polls competition submissions every 30 seconds until both finish scoring,
records official scores to DAILY_JOURNEY.md and ROADMAP_AND_PROGRESS.md,
and commits/pushes to GitHub.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ.setdefault('https_proxy', 'http://176.111.37.5:39811')
os.environ.setdefault('http_proxy', 'http://176.111.37.5:39811')

REPO_ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
LOG_FILE = REPO_ROOT / "logs" / "score_poller.log"
COMPETITION_SLUG = "enveda-CASMI26-molecule-id-mass-spectra"

TARGET_SUBS = {
    57982390: {"version": "Version 24", "desc": "Version 24: Dual-Neural Z-Score Ensemble (Saner v1 + Scorer v2) + MIDX Dedup"},
    57983294: {"version": "Version 25", "desc": "Version 25: Tri-Neural Multi-Capacity Backbone + LOTUS 150k KB + MIDX Dedup"},
    57992067: {"version": "Version 26", "desc": "Version 26: Apex Sovereign Balanced Tri-Force (Saner v1 + Scorer v5 320-dim) + In-Pool InChIKey14 Dedup"},
    57992109: {"version": "Version 26", "desc": "Version 26: Apex Sovereign Balanced Tri-Force (Saner v1 + Scorer v5 320-dim) + In-Pool InChIKey14 Dedup"}
}

def log(msg: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_cmd(cmd: str, cwd=None, timeout_sec=60) -> tuple[int, str]:
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout_sec)
        return res.returncode, (res.stdout + "\n" + res.stderr).strip()
    except Exception as e:
        log(f"Command error: {e}")
        return -1, str(e)

def get_kaggle_api(retries=5, delay=5):
    for attempt in range(retries):
        try:
            api = KaggleApi()
            api.authenticate()
            return api
        except Exception as e:
            log(f"Auth attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Failed to authenticate")

def update_records(v_name: str, desc: str, score: float, ref: int):
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    journey_file = REPO_ROOT / "progress" / "DAILY_JOURNEY.md"
    entry = f"""
### Official Score Verified: {v_name}
- **Timestamp**: {now_utc}
- **Submission Ref**: `{ref}`
- **Description**: {desc}
- **Official Public Score**: **`{score:.4f}`**
- **Status**: Complete & Verified
"""
    if journey_file.exists():
        with open(journey_file, "a", encoding="utf-8") as f:
            f.write(entry)
        log(f"Logged {v_name} ({score:.4f}) to DAILY_JOURNEY.md")

    roadmap_file = REPO_ROOT / "progress" / "ROADMAP_AND_PROGRESS.md"
    if roadmap_file.exists():
        row = f"| {now_utc[:10]} | {v_name} (Ref {ref}) | {desc} | **{score:.4f}** | Succeeded |\n"
        content = roadmap_file.read_text(encoding="utf-8")
        if "| Date | Version |" in content:
            content += row
            roadmap_file.write_text(content, encoding="utf-8")
            log(f"Logged {v_name} ({score:.4f}) to ROADMAP_AND_PROGRESS.md")

    run_cmd("git add -A", cwd=REPO_ROOT)
    run_cmd(f'git commit -m "docs: record {v_name} official public score {score:.4f} (Ref {ref})"', cwd=REPO_ROOT)
    code, out = run_cmd("git push origin main", cwd=REPO_ROOT)
    log(f"Git push result: {out[:150]}")

def poll_scores():
    log("=== Starting Continuous Score Poller for V24 and V25 ===")
    completed = set()
    poll_count = 0
    while len(completed) < len(TARGET_SUBS):
        poll_count += 1
        try:
            api = get_kaggle_api()
            subs = api.competition_submissions(COMPETITION_SLUG)
            for sub in subs:
                ref = getattr(sub, 'ref', None)
                if ref in TARGET_SUBS and ref not in completed:
                    status = str(getattr(sub, 'status', ''))
                    score_str = str(getattr(sub, 'public_score', getattr(sub, 'publicScore', '')))
                    info = TARGET_SUBS[ref]
                    log(f"Poll #{poll_count}: {info['version']} (Ref {ref}) Status={status}, Score={score_str}")
                    if "COMPLETE" in status and score_str.strip():
                        try:
                            score = float(score_str)
                            log(f"🏆 SUCCESS: {info['version']} Scored {score:.4f}!")
                            update_records(info['version'], info['desc'], score, ref)
                            completed.add(ref)
                        except (ValueError, TypeError):
                            pass
                    elif "ERROR" in status or "FAILED" in status:
                        err = getattr(sub, 'errorDescription', 'Scoring error')
                        log(f"❌ ERROR: {info['version']} failed: {err}")
                        completed.add(ref)
        except Exception as e:
            log(f"Exception during poll #{poll_count}: {e}")

        if len(completed) < len(TARGET_SUBS):
            time.sleep(30)

    log("=== All target submissions resolved. Poller exiting cleanly. ===")

if __name__ == "__main__":
    poll_scores()

