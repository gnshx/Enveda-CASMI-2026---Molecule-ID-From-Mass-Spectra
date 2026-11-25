#!/usr/bin/env python3
"""Push a deploy_kernel_vXX directory to Kaggle, wait for completion, submit, and poll score."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_BIN = str(ROOT / ".venv" / "bin" / "kaggle")
KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"


def make_env() -> dict:
    env = os.environ.copy()
    tok_file = Path.home() / ".kaggle" / "access_token"
    if tok_file.is_file() and "KAGGLE_API_TOKEN" not in env:
        env["KAGGLE_API_TOKEN"] = tok_file.read_text(encoding="utf-8").strip()
    return env


def run_cli(args: list[str], env: dict, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [KAGGLE_BIN] + args,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def get_kernel_version(env: dict) -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = KERNEL_SLUG.split("/", 1)
    res = client.kernels.kernels_api_client.get_kernel(req)
    ver = getattr(
        res.metadata,
        "current_version_number",
        getattr(res.metadata, "currentVersionNumber", None),
    )
    if ver is None:
        raise RuntimeError("Could not retrieve current_version_number from Kaggle")
    return int(ver)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-dir", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-wait-sec", type=int, default=7200)
    args = parser.parse_args()

    env = make_env()
    deploy_dir = Path(args.deploy_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Pushing {deploy_dir} to {KERNEL_SLUG}...", flush=True)
    push_res = run_cli(["kernels", "push", "-p", str(deploy_dir)], env=env, timeout=180)
    push_out = (push_res.stdout + "\n" + push_res.stderr).strip()
    print(push_out, flush=True)
    if push_res.returncode != 0 or "error" in push_out.lower():
        raise RuntimeError(f"Kernel push failed: {push_out}")

    print("[2/4] Polling kernel status...", flush=True)
    deadline = time.time() + args.max_wait_sec
    status_out = ""
    while time.time() < deadline:
        time.sleep(20)
        st = run_cli(["kernels", "status", KERNEL_SLUG], env=env, timeout=60)
        status_out = (st.stdout + " " + st.stderr).strip()
        print(f"  status: {status_out}", flush=True)
        low = status_out.lower()
        if "complete" in low and "error" not in low:
            break
        if "error" in low or "failed" in low or "cancel" in low:
            raise RuntimeError(f"Kernel failed: {status_out}")
    else:
        raise TimeoutError(f"Kernel timed out after {args.max_wait_sec}s: {status_out}")

    ver = get_kernel_version(env)
    print(f"[3/4] Kernel complete (version {ver}). Submitting to {COMPETITION}...", flush=True)
    sub_res = run_cli(
        [
            "competitions",
            "submit",
            "-c",
            COMPETITION,
            "-f",
            "submission.csv",
            "-k",
            KERNEL_SLUG,
            "-v",
            str(ver),
            "-m",
            args.message,
        ],
        env=env,
        timeout=180,
    )
    print(sub_res.stdout.strip(), sub_res.stderr.strip(), flush=True)
    if sub_res.returncode != 0:
        raise RuntimeError(f"Submission failed: {sub_res.stderr}")

    print("[4/4] Polling competition submission score...", flush=True)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    score_deadline = time.time() + 10800  # hidden rerun can take ~1-2h
    matched = None
    while time.time() < score_deadline:
        time.sleep(30)
        subs = api.competition_submissions(COMPETITION)
        for s in subs[:10]:
            desc = getattr(s, "description", "") or ""
            if desc.startswith(args.message[:35]):
                status_str = str(getattr(s, "status", ""))
                pub_score = getattr(s, "public_score", getattr(s, "publicScore", None))
                ref = getattr(s, "ref", None)
                print(f"  sub ref={ref} status={status_str} publicScore={pub_score}", flush=True)
                if "COMPLETE" in status_str.upper() or "ERROR" in status_str.upper():
                    matched = {
                        "ref": ref,
                        "kernel_version": ver,
                        "status": status_str,
                        "public_score": pub_score,
                        "description": desc,
                    }
                    break
        if matched is not None:
            break

    out_json.write_text(json.dumps(matched or {"kernel_version": ver, "status": "timeout"}, indent=2))
    print("DONE:", json.dumps(matched), flush=True)


if __name__ == "__main__":
    main()
