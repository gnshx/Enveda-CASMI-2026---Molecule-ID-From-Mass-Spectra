#!/usr/bin/env python3
"""
scripts/score_poller.py
Continuous Dynamic Score Poller.
Polls competition submissions every 30 seconds for new or pending submissions,
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
        content = journey_file.read_text(encoding="utf-8")
        if f"Submission Ref**: `{ref}`" not in content or "0.0000" in content:
            with open(journey_file, "a", encoding="utf-8") as f:
                f.write(entry)
            log(f"Logged {v_name} ({score:.4f}) to DAILY_JOURNEY.md")
        else:
            log(f"Ref {ref} already documented in DAILY_JOURNEY.md, skipping duplicate append.")

    roadmap_file = REPO_ROOT / "progress" / "ROADMAP_AND_PROGRESS.md"
    if roadmap_file.exists():
        content = roadmap_file.read_text(encoding="utf-8")
        if f"Ref {ref}" not in content:
            row = f"| {now_utc[:10]} | {v_name} (Ref {ref}) | {desc} | **{score:.4f}** | Succeeded |\n"
            if "| Date | Version |" in content:
                content += row
                roadmap_file.write_text(content, encoding="utf-8")
                log(f"Logged {v_name} ({score:.4f}) to ROADMAP_AND_PROGRESS.md")

    run_cmd("git add -A", cwd=REPO_ROOT)
    run_cmd(f'git commit -m "docs: record {v_name} official public score {score:.4f} (Ref {ref})"', cwd=REPO_ROOT)
    code, out = run_cmd("git push origin main", cwd=REPO_ROOT)
    log(f"Git push result: {out[:150]}")

def poll_latest_loop(interval_sec=30):
    log("=== Continuous Dynamic Score Poller Started ===")
    seen_completed = {57982390, 57983294, 57992067, 57992109, 58021007}
    poll_count = 0
    while True:
        poll_count += 1
        try:
            api = get_kaggle_api()
            subs = api.competition_submissions(COMPETITION_SLUG)
            if subs:
                for sub in subs[:5]:
                    ref = getattr(sub, 'ref', None)
                    status = str(getattr(sub, 'status', ''))
                    score_str = str(getattr(sub, 'public_score', getattr(sub, 'publicScore', '')))
                    desc = getattr(sub, 'description', '')
                    v_name = f"Ref {ref}"
                    if "Version " in desc:
                        v_name = desc.split(":")[0]

                    if "COMPLETE" in status and score_str.strip() and ref not in seen_completed:
                        try:
                            score = float(score_str)
                            log(f"🏆 SUCCESS: {v_name} (Ref {ref}) Scored {score:.4f}!")
                            update_records(v_name, desc, score, ref)
                            seen_completed.add(ref)
                        except (ValueError, TypeError):
                            pass
                    elif "PENDING" in status:
                        log(f"Poll #{poll_count}: {v_name} (Ref {ref}) Status=PENDING, Score={score_str}")
                    elif "ERROR" in status or "FAILED" in status:
                        if ref not in seen_completed:
                            err = getattr(sub, 'errorDescription', 'Scoring error')
                            log(f"❌ ERROR: {v_name} (Ref {ref}) failed: {err}")
                            seen_completed.add(ref)
        except Exception as e:
            log(f"Exception during poll #{poll_count}: {e}")
        time.sleep(interval_sec)

if __name__ == "__main__":
    poll_latest_loop()
