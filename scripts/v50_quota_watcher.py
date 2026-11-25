#!/usr/bin/env python3
"""
scripts/v50_quota_watcher.py
=============================================================================
CASMI 2026 — Resilient OS-Persistent 5/Day Kaggle Runner & Submitter

Survives AGY closing, network disconnects, and process restarts:
  - Single-instance exclusive lock via fcntl.flock (`logs/v50_quota_watcher.lock`)
  - Full try/except network retry loops around every Kaggle CLI/SDK call
  - Crash/reboot-safe state persistence (`logs/v50_queue_state.json`) tracking
    `active` pushed runs so a running kernel is resumed instead of re-pushed
  - Respects 5/day UTC submission quota (sleeps until UTC rollover when 0/5)

Queue (Best 5 Staged Versions):
  1. deploy_kernel_v50  : V50 Selected Calibrated Ensemble (beta=0.20, delta=0.35)
  2. deploy_kernel_v50b : V50b Conservative Gate (beta=0.15, delta=0.60)
  3. deploy_kernel_v50d : V50d Ranks 2-25 Calibrated + Strict Rank-1 Lock
  4. deploy_kernel_v50c : V50c Bold Gate (beta=0.25, delta=0.15)
  5. deploy_kernel_v42  : V42 Audited Exact V24 0.4800 PB Control
=============================================================================
"""

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_BIN = str(ROOT / ".venv" / "bin" / "kaggle")
KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
LOG_FILE = ROOT / "logs" / "v50_quota_watcher.log"
STATE_FILE = ROOT / "logs" / "v50_queue_state.json"
LOCK_FILE = ROOT / "logs" / "v50_quota_watcher.lock"

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


def run_cli(args: list[str], env: dict, timeout: int = 180) -> tuple[int, str]:
    try:
        res = subprocess.run([KAGGLE_BIN] + args, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
        return res.returncode, (res.stdout + " " + res.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "TimeoutExpired while contacting Kaggle"
    except Exception as exc:
        return -1, f"Network/Subprocess exception: {exc!r}"


def is_transient_network_err(text: str) -> bool:
    low = text.lower()
    return any(
        k in low
        for k in (
            "connection",
            "timeout",
            "timed out",
            "temporary failure",
            "name resolution",
            "network is unreachable",
            "max retries exceeded",
            "remote disconnected",
            "ssl",
            "502",
            "503",
            "504",
            "429",
        )
    )


def get_api(env: dict):
    os.environ["KAGGLE_API_TOKEN"] = env.get("KAGGLE_API_TOKEN", "")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def remaining_daily_submissions(env: dict) -> int | None:
    try:
        api = get_api(env)
        subs = api.competition_submissions(COMPETITION)
        today = datetime.now(timezone.utc).date()
        used = sum(1 for s in subs if getattr(s, "date", datetime.now(timezone.utc)).date() == today)
        return max(0, 5 - used)
    except Exception as exc:
        log(f"Network/API error checking quota ({exc!r}); will retry...")
        return None


def get_kernel_version(env: dict) -> int:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    api = get_api(env)
    client = api.build_kaggle_client()
    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = KERNEL_SLUG.split("/", 1)
    res = client.kernels.kernels_api_client.get_kernel(req)
    ver = getattr(res.metadata, "current_version_number", getattr(res.metadata, "currentVersionNumber", None))
    return int(ver)


def load_state() -> dict:
    if STATE_FILE.is_file():
        try:
            st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            st.setdefault("completed", {})
            st.setdefault("failed", {})
            st.setdefault("active", None)
            return st
        except Exception:
            pass
    return {"completed": {}, "failed": {}, "active": None}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def sleep_until_utc_midnight() -> None:
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    sec = max(60, int((nxt - now).total_seconds()))
    log(f"Daily 5/5 submission quota reached. Sleeping {sec}s until UTC rollover ({nxt.isoformat()})...")
    time.sleep(sec)


def wait_for_kernel_and_submit(item: dict, state: dict, env: dict, net_retry_sec: int) -> None:
    tag = item["tag"]
    active = state.get("active") or {"tag": tag, "stage": "pushed"}

    if active.get("stage") == "pushed":
        deadline = time.time() + 14400
        while time.time() < deadline:
            time.sleep(30)
            rc, st_out = run_cli(["kernels", "status", KERNEL_SLUG], env=env, timeout=60)
            if rc != 0 or is_transient_network_err(st_out):
                log(f"[{tag}] Status check interrupted ({st_out[:160]}); retrying in {net_retry_sec}s...")
                time.sleep(net_retry_sec)
                continue
            log(f"[{tag}] Kernel status: {st_out[:180]}")
            low = st_out.lower()
            if "complete" in low and "error" not in low:
                active["stage"] = "kernel_complete"
                state["active"] = active
                save_state(state)
                break
            if "error" in low or "failed" in low or "cancel" in low:
                state["failed"][tag] = st_out
                state["active"] = None
                save_state(state)
                return
        else:
            state["failed"][tag] = "Kernel commit polling timed out"
            state["active"] = None
            save_state(state)
            return

    if active.get("stage") == "kernel_complete":
        while True:
            rem = remaining_daily_submissions(env)
            if rem is None:
                time.sleep(net_retry_sec)
                continue
            if rem <= 0:
                sleep_until_utc_midnight()
                continue
            try:
                ver = get_kernel_version(env)
            except Exception as exc:
                log(f"[{tag}] Error fetching kernel version ({exc!r}); retrying in {net_retry_sec}s...")
                time.sleep(net_retry_sec)
                continue

            log(f"[{tag}] Submitting kernel v{ver} to {COMPETITION}...")
            rc, sub_out = run_cli(
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
            log(f"[{tag}] Submit output: {sub_out[:240]}")
            if rc != 0:
                if is_transient_network_err(sub_out):
                    log(f"[{tag}] Transient submit error; retrying in {net_retry_sec}s...")
                    time.sleep(net_retry_sec)
                    continue
                state["failed"][tag] = sub_out
                state["active"] = None
                save_state(state)
                return
            active["stage"] = "submitted"
            active["kernel_version"] = ver
            state["active"] = active
            save_state(state)
            break

    if active.get("stage") == "submitted":
        ver = active.get("kernel_version")
        sc_deadline = time.time() + 14400
        result = {"kernel_version": ver, "status": "SUBMITTED"}
        while time.time() < sc_deadline:
            time.sleep(60)
            try:
                api = get_api(env)
                subs = api.competition_submissions(COMPETITION)
            except Exception as exc:
                log(f"[{tag}] Network error polling score ({exc!r}); retrying in {net_retry_sec}s...")
                time.sleep(net_retry_sec)
                continue

            for s in subs[:10]:
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
        state["active"] = None
        save_state(state)
        log(f"[{tag}] Final recorded result: {json.dumps(result)}")


def acquire_lock():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fp = open(LOCK_FILE, "w", encoding="utf-8")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(0)
    fp.write(str(os.getpid()))
    fp.flush()
    return fp


def main():
    _lock_fp = acquire_lock()
    env = make_env()
    state = load_state()
    quota_retry_sec = int(os.environ.get("KAGGLE_QUOTA_RETRY_SEC", "900"))
    net_retry_sec = int(os.environ.get("KAGGLE_NET_RETRY_SEC", "60"))
    log(f"Started resilient V50 Kaggle watcher (PID={os.getpid()}).")

    # Resume any active pushed/submitted run first after disconnect/reboot
    if state.get("active"):
        a_tag = state["active"].get("tag")
        matched = [q for q in QUEUE if q["tag"] == a_tag]
        if matched:
            log(f"Resuming active run for {a_tag} (stage={state['active'].get('stage')})...")
            wait_for_kernel_and_submit(matched[0], state, env, net_retry_sec)

    for item in QUEUE:
        tag = item["tag"]
        if tag in state["completed"]:
            log(f"Skipping {tag} (already completed: {state['completed'][tag]})")
            continue

        while True:
            rem = remaining_daily_submissions(env)
            if rem is None:
                time.sleep(net_retry_sec)
                continue
            log(f"[{tag}] Daily submissions remaining today: {rem}/5")
            if rem <= 0:
                sleep_until_utc_midnight()
                continue

            rc, push_out = run_cli(["kernels", "push", "-p", item["dir"]], env=env)
            log(f"[{tag}] Push response: {push_out[:240]}")

            if rc != 0 or "error" in push_out.lower():
                if is_transient_network_err(push_out):
                    log(f"[{tag}] Network/API offline; retrying in {net_retry_sec}s...")
                    time.sleep(net_retry_sec)
                    continue
                if "quota" in push_out.lower():
                    log(f"[{tag}] GPU quota full; retrying in {quota_retry_sec}s...")
                    time.sleep(quota_retry_sec)
                    continue
                state["failed"][tag] = push_out
                save_state(state)
                break

            state["active"] = {"tag": tag, "stage": "pushed"}
            save_state(state)
            wait_for_kernel_and_submit(item, state, env, net_retry_sec)
            break

    log("All queued versions processed. Watcher exiting.")


if __name__ == "__main__":
    main()
