#!/usr/bin/env python3
"""
scripts/fast_submit_all.py
===========================================================================
CASMI 2026: Immediate 5-Version Sequential Push & Submit
- Fixes the kaggle PATH issue by using full absolute path
- Pushes v37, v38, v39, v40, v41 sequentially
- Monitors each kernel to completion
- Submits to hidden test immediately
- Runs completely independently of AGY
===========================================================================
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# ─── PATHS & CONFIG ────────────────────────────────────────────────────────
PROXY = "http://176.111.37.5:39811"
KAGGLE_BIN = "/home/dlcv/anaconda3/bin/kaggle"
ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
LOG_FILE = ROOT / "logs" / "fast_submit.log"
STATUS_FILE = ROOT / "logs" / "fast_submit_status.json"

KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"

VERSIONS_QUEUE = [
    {
        "version_tag": "v37",
        "deploy_dir": ROOT / "deploy_kernel_v37",
        "label": "Version 37: Sovereign Regime B Rank-1 Promotion (0.75 w_top) + High-Res Isotope Engine + Pure Shield (Target LB 0.515+)",
    },
    {
        "version_tag": "v38",
        "deploy_dir": ROOT / "deploy_kernel_v38",
        "label": "Version 38: V37 + Active Regioisomer Substitution (Ranks 20-25 Constitutional Isomers) (Target LB 0.520+)",
    },
    {
        "version_tag": "v39",
        "deploy_dir": ROOT / "deploy_kernel_v39",
        "label": "Version 39: V38 + Tuned Weights (POP_W=0.25, SPEC_W=0.10, NP_W=0.08) (Target LB 0.522+)",
    },
    {
        "version_tag": "v40",
        "deploy_dir": ROOT / "deploy_kernel_v40",
        "label": "Version 40: Stronger Regime B Promotion (0.80 w_top) + Higher POP_W (0.35) + NP_W (0.15) (Target LB 0.525+)",
    },
    {
        "version_tag": "v41",
        "deploy_dir": ROOT / "deploy_kernel_v41",
        "label": "Version 41: Maximum Regime B Promotion (0.90 w_top) + Maximum Pop Prior (0.40) + SPEC_W (0.15) (Target LB 0.528+)",
    },
]

# ─── ENV ────────────────────────────────────────────────────────────────────
def make_env():
    env = os.environ.copy()
    env["https_proxy"] = PROXY
    env["http_proxy"] = PROXY
    env["HTTP_PROXY"] = PROXY
    env["HTTPS_PROXY"] = PROXY
    env["PATH"] = f"/home/dlcv/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:{env.get('PATH','')}"
    return env

# ─── LOGGING ────────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_status(state: dict):
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(STATUS_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ─── RUN CMD ────────────────────────────────────────────────────────────────
def run_cmd(cmd_parts: list, timeout_sec=300):
    try:
        res = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=make_env(),
            timeout=timeout_sec
        )
        return res
    except subprocess.TimeoutExpired:
        log(f"⚠️ Timeout ({timeout_sec}s): {' '.join(cmd_parts)}")
        return subprocess.CompletedProcess(args=cmd_parts, returncode=-1, stdout="", stderr="Timeout")

# ─── CHECK QUOTA ────────────────────────────────────────────────────────────
def check_quota():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        subs = api.competition_submissions(COMPETITION)
        today_utc = datetime.now(timezone.utc).date()
        today_subs = [s for s in subs if getattr(s, "date", datetime.now(timezone.utc)).date() == today_utc]
        rem = max(0, 5 - len(today_subs))
        log(f"📊 Quota: {rem}/5 remaining today ({len(today_subs)} used)")
        return rem
    except Exception as e:
        log(f"⚠️ Quota check error: {e}")
        return 5  # assume available

# ─── PUSH KERNEL ────────────────────────────────────────────────────────────
def push_kernel(deploy_dir: Path, version_tag: str) -> bool:
    log(f"🚀 Pushing kernel {version_tag} from {deploy_dir}")
    res = run_cmd([KAGGLE_BIN, "kernels", "push", "-p", str(deploy_dir)], timeout_sec=120)
    log(f"   stdout: {res.stdout.strip()}")
    log(f"   stderr: {res.stderr.strip()}")
    if res.returncode != 0:
        log(f"❌ Push FAILED for {version_tag} (exit {res.returncode})")
        return False
    log(f"✅ Push SUCCESS for {version_tag}")
    return True

# ─── GET KERNEL VERSION NUMBER ───────────────────────────────────────────────
def get_kernel_version_number() -> int | None:
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
        ver = getattr(res.metadata, "current_version_number",
                      getattr(res.metadata, "currentVersionNumber", None))
        if ver is not None:
            return int(ver)
    except Exception as e:
        log(f"⚠️ Kernel version query error: {e}")
    return None

# ─── WAIT FOR KERNEL COMPLETION ──────────────────────────────────────────────
def wait_for_kernel(version_tag: str, max_wait_sec=9600):
    log(f"⏳ Waiting for kernel to complete (max {max_wait_sec//60} min)...")
    deadline = time.time() + max_wait_sec
    poll_interval = 60  # seconds

    while time.time() < deadline:
        time.sleep(poll_interval)
        res = run_cmd([KAGGLE_BIN, "kernels", "status", KERNEL_SLUG], timeout_sec=60)
        output = res.stdout.strip() + res.stderr.strip()
        log(f"   Kernel status: {output[:200]}")

        if "complete" in output.lower() and "error" not in output.lower():
            log(f"✅ Kernel {version_tag} COMPLETE!")
            return True
        elif "error" in output.lower() and "complete" not in output.lower():
            log(f"❌ Kernel {version_tag} ERROR!")
            return False
        elif "running" in output.lower() or "queued" in output.lower():
            log(f"   Still running... ({int((deadline - time.time()) / 60)} min remaining)")
        else:
            log(f"   Status unclear, continuing to poll...")

    log(f"⚠️ Kernel wait timeout for {version_tag}!")
    return False

# ─── SUBMIT ─────────────────────────────────────────────────────────────────
def submit_version(kernel_ver_num: int, label: str, version_tag: str) -> bool:
    log(f"📤 Submitting {version_tag} (kernel version {kernel_ver_num})")
    res = run_cmd([
        KAGGLE_BIN, "competitions", "submit",
        "-c", COMPETITION,
        "-f", "submission.csv",
        "-k", KERNEL_SLUG,
        "-v", str(kernel_ver_num),
        "-m", label
    ], timeout_sec=120)
    log(f"   stdout: {res.stdout.strip()}")
    log(f"   stderr: {res.stderr.strip()}")
    if res.returncode != 0 or "error" in res.stderr.lower():
        log(f"❌ Submit FAILED for {version_tag}")
        return False
    log(f"✅ Submit SUCCESS for {version_tag} (kernel v{kernel_ver_num})")
    return True

# ─── POLL FOR SCORE ─────────────────────────────────────────────────────────
def poll_score(version_tag: str, expected_description_prefix: str, max_wait=1200):
    log(f"📊 Polling for score ({version_tag})...")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        time.sleep(60)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            subs = api.competition_submissions(COMPETITION)
            if subs:
                latest = subs[0]
                status = str(getattr(latest, "status", ""))
                score = getattr(latest, "publicScore", None)
                log(f"   Latest submission status={status} score={score}")
                if "complete" in status.lower():
                    log(f"🏆 {version_tag} SCORED: {score}")
                    return score
        except Exception as e:
            log(f"⚠️ Score poll error: {e}")
    log(f"⚠️ Score poll timeout for {version_tag}")
    return None

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    log("=" * 80)
    log("🤖 CASMI 2026 FAST SUBMIT ALL — Starting")
    log(f"   Versions queued: {[v['version_tag'] for v in VERSIONS_QUEUE]}")
    log("=" * 80)

    # Add anaconda to Python sys.path so kaggle module imports work
    import sys
    anaconda_site = "/home/dlcv/anaconda3/lib/python3.12/site-packages"
    if anaconda_site not in sys.path:
        sys.path.insert(0, anaconda_site)

    # Check quota
    quota = check_quota()
    if quota == 0:
        log("🚫 No quota remaining today! Exiting.")
        return

    results = {}
    n_submitted = 0

    for ver in VERSIONS_QUEUE:
        if n_submitted >= quota:
            log(f"🛑 Quota exhausted ({n_submitted} submitted). Stopping.")
            break

        tag = ver["version_tag"]
        deploy_dir = ver["deploy_dir"]
        label = ver["label"]

        log(f"\n{'='*60}")
        log(f"▶️  PROCESSING {tag}")
        log(f"{'='*60}")
        write_status({"phase": f"pushing_{tag}", "results": results})

        # 1. Push kernel
        ok = push_kernel(deploy_dir, tag)
        if not ok:
            log(f"⚠️ Skipping {tag} due to push failure")
            results[tag] = {"status": "push_failed"}
            continue

        # 2. Get version number before waiting
        time.sleep(10)  # brief pause for Kaggle to register the push
        ver_num = get_kernel_version_number()
        log(f"   Kernel version number after push: {ver_num}")

        # 3. Wait for kernel to complete
        completed = wait_for_kernel(tag)
        if not completed:
            log(f"⚠️ {tag} kernel did not complete cleanly - still attempting submit")

        # 4. Get final version number
        final_ver = get_kernel_version_number()
        if final_ver is None:
            final_ver = ver_num
        log(f"   Final kernel version number: {final_ver}")

        if final_ver is None:
            log(f"❌ Cannot determine kernel version for {tag} - skipping submit")
            results[tag] = {"status": "version_unknown"}
            continue

        # 5. Submit
        write_status({"phase": f"submitting_{tag}", "results": results})
        sub_ok = submit_version(final_ver, label, tag)
        if sub_ok:
            n_submitted += 1
            write_status({"phase": f"polling_score_{tag}", "results": results})
            score = poll_score(tag, label)
            results[tag] = {"status": "submitted", "kernel_ver": final_ver, "score": score}
            log(f"✅ {tag} done. Score: {score}. Total submitted: {n_submitted}/{quota}")
        else:
            results[tag] = {"status": "submit_failed", "kernel_ver": final_ver}

        # Respect rate limiting between versions
        if n_submitted < quota and ver != VERSIONS_QUEUE[-1]:
            log("   Waiting 30s before next version...")
            time.sleep(30)

    log("\n" + "=" * 80)
    log("🏁 FAST SUBMIT ALL COMPLETE")
    for tag, r in results.items():
        log(f"   {tag}: {r}")
    log("=" * 80)
    write_status({"phase": "complete", "results": results})

if __name__ == "__main__":
    main()
