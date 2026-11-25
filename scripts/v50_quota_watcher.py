#!/usr/bin/env python3
"""
scripts/v50_quota_watcher.py
=============================================================================
CASMI 2026 — Automated 5/Day Kaggle Kernel Runner & Leaderboard Submitter

Queue (Best 5 Staged Versions):
  1. deploy_kernel_v50  : V50 Selected Calibrated Ensemble (beta=0.20, delta=0.35)
  2. deploy_kernel_v50b : V50b Conservative Gate (beta=0.15, delta=0.60)
  3. deploy_kernel_v50d : V50d Ranks 2-25 Calibrated + Strict Rank-1 Lock
  4. deploy_kernel_v50c : V50c Bold Gate (beta=0.25, delta=0.15)
  5. deploy_kernel_v42  : V42 Audited Exact V24 0.4800 PB Control

Behavior:
  - Checks daily competition submission quota (5/day UTC). If 0 remaining,
    sleeps until UTC midnight rollover + 60s.
  - Attempts `kaggle kernels push -p <deploy_dir>`. If Kaggle weekly GPU quota
    is full (`Maximum weekly GPU quota`), sleeps `retry_sec` (default 900s)
    and retries automatically.
  - Once pushed, polls `kaggle kernels status` every 30s until COMPLETE,
    fetches the new kernel version number, submits to the competition
    immediately, and polls until `publicScore` posts.
  - Persists state in `logs/v50_queue_state.json` and logs in `logs/v50_quota_watcher.log`.
=============================================================================
"""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_BIN = str(ROOT / ".venv" / "bin" / "kaggle")
KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
LOG_FILE = ROOT / "logs" / "v50_quota_watcher.log"
STATE_FILE = ROOT / "logs" / "v50_queue_state.json"

QUEUE = [
    {
        "tag": "v50",
        "dir": str(ROOT / "deploy_kernel_v50"),
        "label": "Version 50: Calibrated Ensemble + Candidate-Aware Gate (beta=0.20, delta=0.35)",
    },
    {
        "tag": "v50b",
        "dir": str(ROOT / "deploy_kernel_v50b"),
        "label": "Version 50b: Conservative Rank-1 Challenger Gate (beta=0.15, delta=0.60)",
    },
    {
        "tag": "v50d",
        "dir": str(ROOT / "deploy_kernel_v50d"),
        "label": "Version 50d: Ranks 2-25 Calibrated V50 + Strict Rank-1 Lock",
    },
    {
        "tag": "v50c",
        "dir": str(ROOT / "deploy_kernel_v50c"),
        "label": "Version 50c: Bold Rank-1 Challenger Gate (beta=0.25, delta=0.15)",
    },
    {
        "tag": "v42",
        "dir": str(ROOT / "deploy_kernel_v42"),
        "label": "Version 42: Exact V24 0.4800 PB Control + Stage Execution Audit",
    },
]


def log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_env() -> dict:
    env = os.environ.copy()
    tok_file = Path.home() / ".kaggle" / "access_token"
    if tok_file.is_file() and "KAGGLE_API_TOKEN" not in env:
        env["KAGGLE_API_TOKEN"] = tok_file.read_text(encoding="utf-8").strip()
    return env


def run_cli(args: list[str], env: dict, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([KAGGLE_BIN] + args, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)


def remaining_daily_submissions(env: dict) -> int:
    os.environ["KAGGLE_API_TOKEN"] = env.get("KAGGLE_API_TOKEN", "")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    subs = api.competition_submissions(COMPETITION)
    today = datetime.now(timezone.utc).date()
    used = sum(1 for s in subs if getattr(s, "date", datetime.now(timezone.utc)).date() == today)
    return max(0, 5 - used)


def get_kernel_version(env: dict) -> int:
    os.environ["KAGGLE_API_TOKEN"] = env.get("KAGGLE_API_TOKEN", "")
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = KERNEL_SLUG.split("/", 1)
    res = client.kernels.kernels_api_client.get_kernel(req)
    ver = getattr(res.metadata, "current_version_number", getattr(res.metadata, "currentVersionNumber", None))
    return int(ver)


def load_state() -> dict:
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"completed": {}, "failed": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def sleep_until_utc_midnight() -> None:
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    sec = max(60, int((nxt - now).total_seconds()))
    log(f"Daily 5/5 submission quota reached. Sleeping {sec}s until UTC rollover ({nxt.isoformat()})...")
    time.sleep(sec)


def main():
    env = make_env()
    state = load_state()
    retry_sec = int(os.environ.get("KAGGLE_QUOTA_RETRY_SEC", "900"))
    log("Started V50 automated Kaggle kernel & 5/day leaderboard submission watcher.")

    for item in QUEUE:
        tag = item["tag"]
        if tag in state["completed"]:
            log(f"Skipping {tag} (already completed: {state['completed'][tag]})")
            continue

        while True:
            rem = remaining_daily_submissions(env)
            log(f"[{tag}] Daily submissions remaining today: {rem}/5")
            if rem <= 0:
                sleep_until_utc_midnight()
                continue

            push_res = run_cli(["kernels", "push", "-p", item["dir"]], env=env)
            push_out = (push_res.stdout + " " + push_res.stderr).strip()
            log(f"[{tag}] Push response: {push_out[:240]}")

            if push_res.returncode != 0 or "error" in push_out.lower():
                if "quota" in push_out.lower():
                    log(f"[{tag}] GPU quota full; retrying in {retry_sec}s...")
                    time.sleep(retry_sec)
                    continue
                state["failed"][tag] = push_out
                save_state(state)
                break

            # Wait for Kaggle commit completion
            deadline = time.time() + 10800
            kernel_ok = False
            while time.time() < deadline:
                time.sleep(30)
                st = run_cli(["kernels", "status", KERNEL_SLUG], env=env, timeout=60)
                st_out = (st.stdout + " " + st.stderr).strip()
                log(f"[{tag}] Kernel status: {st_out[:180]}")
                low = st_out.lower()
                if "complete" in low and "error" not in low:
                    kernel_ok = True
                    break
                if "error" in low or "failed" in low or "cancel" in low:
                    state["failed"][tag] = st_out
                    save_state(state)
                    break

            if not kernel_ok:
                break

            ver = get_kernel_version(env)
            log(f"[{tag}] Kernel COMPLETE (v{ver}). Submitting to {COMPETITION}...")
            sub_res = run_cli(
                [
                    "competitions", "submit",
                    "-c", COMPETITION,
                    "-f", "submission.csv",
                    "-k", KERNEL_SLUG,
                    "-v", str(ver),
                    "-m", item["label"],
                ],
                env=env,
            )
            sub_out = (sub_res.stdout + " " + sub_res.stderr).strip()
            log(f"[{tag}] Submit output: {sub_out[:240]}")
            if sub_res.returncode != 0:
                state["failed"][tag] = sub_out
                save_state(state)
                break

            # Poll for leaderboard score (up to 3h for hidden test rerun)
            from kaggle.api.kaggle_api_extended import KaggleApi

            os.environ["KAGGLE_API_TOKEN"] = env.get("KAGGLE_API_TOKEN", "")
            api = KaggleApi()
            api.authenticate()
            sc_deadline = time.time() + 10800
            result = {"kernel_version": ver, "status": "SUBMITTED"}
            while time.time() < sc_deadline:
                time.sleep(60)
                for s in api.competition_submissions(COMPETITION)[:10]:
                    desc = getattr(s, "description", "") or ""
                    if desc.startswith(item["label"][:35]):
                        st_s = str(getattr(s, "status", ""))
                        ps = getattr(s, "public_score", getattr(s, "publicScore", None))
                        ref = getattr(s, "ref", None)
                        if "COMPLETE" in st_s.upper() or "ERROR" in st_s.upper():
                            result = {"ref": ref, "kernel_version": ver, "status": st_s, "public_score": ps}
                            break
                if result["status"] != "SUBMITTED":
                    break

            state["completed"][tag] = result
            save_state(state)
            log(f"[{tag}] Final recorded result: {json.dumps(result)}")
            break

    log("All queued versions processed. Watcher exiting.")


if __name__ == "__main__":
    main()
