"""Run a command on a throwaway RunPod GPU, then terminate the pod.

    uv run python launch.py "python baseline.py --limit 500 --out runs/base.json"

Creates a pod on the cheapest card with enough VRAM, installs the project,
runs the command with output streaming live, copies anything under runs/ back
to this machine, and terminates the pod.

The pod is terminated in a `finally` block, so a crash, a failed command, or
Ctrl-C still shuts it down. An orphaned pod bills by the hour until noticed,
which on a $10 budget is the whole budget.

Requires RUNPOD_API_KEY in .env and an SSH public key registered under
Settings -> SSH Public Keys in your RunPod account.
"""

import argparse
import os
import subprocess
import time

import runpod
from dotenv import load_dotenv

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
WORKDIR = "/workspace/chessllm"
MIN_VRAM_GB = 24  # GRPO later needs policy + frozen reference + Adam states

# The working tree is uploaded as-is rather than cloned, so a run always
# reflects local edits -- no commit/push cycle to test a one-line change.
# .env is excluded deliberately: the API key must never land on a rented box.
# .venv is excluded because macOS wheels are useless on the pod's Linux.
NO_UPLOAD = [".git", ".venv", "__pycache__", ".env", "runs", ".ipynb_checkpoints"]

# Debian puts stockfish in /usr/games, which is not on root's PATH.
SYSTEM_SETUP = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq stockfish curl rsync
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p {WORKDIR}
echo "--- system setup done ---"
"""

PROJECT_SETUP = f"""
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/games:$PATH"
cd {WORKDIR} && uv sync -q
echo "--- deps installed ---"
"""


def price_of(detail):
    """Cheapest on-demand hourly price from a get_gpu() detail dict.

    Spot prices (minimumBidPrice) are deliberately ignored: a spot pod can be
    reclaimed mid-run, and a half-finished eval costs more than it saves.
    """
    lowest = detail.get("lowestPrice") or {}
    candidates = [
        lowest.get("uninterruptablePrice"),
        detail.get("communityPrice"),
        detail.get("securePrice"),
    ]
    prices = [p for p in candidates if isinstance(p, (int, float)) and p > 0]
    return min(prices) if prices else None


def priced_gpus(min_vram):
    """[(price, detail)] for available GPU types with >= min_vram GB, cheapest first.

    Two calls are needed: get_gpus() returns only id/displayName/memoryInGb
    with no pricing at all, so candidates have to be priced individually with
    get_gpu(). Filtering on VRAM first keeps that to a handful of calls.
    """
    candidates = [
        gpu for gpu in runpod.get_gpus()
        if (gpu.get("memoryInGb") or 0) >= min_vram
    ]
    if not candidates:
        raise SystemExit(f"no GPU type with >= {min_vram} GB VRAM")

    priced = []
    for gpu in candidates:
        detail = runpod.get_gpu(gpu["id"])
        price = price_of(detail)
        if price:
            priced.append((price, detail))

    return sorted(priced, key=lambda p: p[0])


def pick_gpu(min_vram):
    priced = priced_gpus(min_vram)
    if not priced:
        raise SystemExit(
            f"no GPU with >= {min_vram} GB is currently available. "
            "Try --min-vram 16, or --list-gpus to see what there is."
        )
    price, gpu = priced[0]
    print(f"picked {gpu['displayName']}  {gpu['memoryInGb']}GB  ${price:.2f}/hr")
    return gpu


def wait_for_ssh(pod_id, timeout=600):
    """Poll until the pod exposes a public SSH port. Returns (ip, port)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id)
        runtime = (pod or {}).get("runtime") or {}
        for port in runtime.get("ports") or []:
            if port.get("privatePort") == 22 and port.get("isIpPublic"):
                return port["ip"], int(port["publicPort"])
        print("  waiting for pod...")
        time.sleep(10)
    raise SystemExit("pod never exposed SSH within timeout")


def ssh(ip, port, script, check=True):
    """Run a bash script on the pod, streaming output to this terminal."""
    return subprocess.run(
        [
            "ssh", "-p", str(port), f"root@{ip}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "bash -s",
        ],
        input=script,
        text=True,
        check=check,
    )


def ssh_opts(port):
    return (
        f"ssh -p {port} -o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    )


def upload(ip, port):
    """Push the local working tree to the pod."""
    excludes = [arg for name in NO_UPLOAD for arg in ("--exclude", name)]
    subprocess.run(
        ["rsync", "-az", "--delete", "-e", ssh_opts(port), *excludes,
         "./", f"root@{ip}:{WORKDIR}/"],
        check=True,
    )
    print("working tree uploaded")


def fetch_results(ip, port):
    """Copy the pod's runs/ directory back into ./runs."""
    os.makedirs("runs", exist_ok=True)
    result = subprocess.run(
        ["rsync", "-az", "-e", ssh_opts(port),
         f"root@{ip}:{WORKDIR}/runs/", "runs/"],
        check=False,
    )
    if result.returncode == 0:
        print("results copied to runs/")
    else:
        print("no results to copy (nothing written to runs/ on the pod)")


def report_cost(pod_id):
    """Print what this run cost, from the pod's own uptime and hourly rate."""
    try:
        pod = runpod.get_pod(pod_id) or {}
        hours = (pod.get("uptimeSeconds") or 0) / 3600
        rate = pod.get("costPerHr") or 0
        print(f"\nuptime {hours * 60:.1f} min at ${rate:.2f}/hr = ${hours * rate:.2f}")
    except Exception as exc:  # never let accounting block termination
        print(f"(could not read cost: {exc})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?",
                        help="shell command to run inside the repo")
    parser.add_argument("--list-gpus", action="store_true",
                        help="show available GPUs with live prices, then exit")
    parser.add_argument("--gpu", default=None, help="gpu_type_id, overrides auto-pick")
    parser.add_argument("--min-vram", type=int, default=MIN_VRAM_GB)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--disk", type=int, default=40, help="container disk GB")
    parser.add_argument("--cloud", default="ALL", choices=["ALL", "COMMUNITY", "SECURE"])
    parser.add_argument("--keep", action="store_true",
                        help="leave the pod running (you must terminate it yourself)")
    args = parser.parse_args()

    load_dotenv()
    key = os.getenv("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY not set (put it in .env)")
    runpod.api_key = key

    if args.list_gpus:
        for price, gpu in priced_gpus(args.min_vram):
            print(f"  {gpu['displayName']:32s} {gpu['memoryInGb']:3d}GB  ${price:.2f}/hr")
        return

    if not args.command:
        raise SystemExit("give a command to run, or pass --list-gpus")

    gpu_id = args.gpu or pick_gpu(args.min_vram)["id"]

    pod = runpod.create_pod(
        name="chessllm",
        image_name=args.image,
        gpu_type_id=gpu_id,
        cloud_type=args.cloud,
        container_disk_in_gb=args.disk,
        support_public_ip=True,
        start_ssh=True,
        ports="22/tcp",
    )
    pod_id = pod["id"]
    print(f"pod {pod_id} created")

    try:
        ip, port = wait_for_ssh(pod_id)
        print(f"ssh root@{ip} -p {port}\n")

        ssh(ip, port, SYSTEM_SETUP)
        upload(ip, port)
        ssh(ip, port, PROJECT_SETUP)

        print(f"\n--- running: {args.command} ---\n")
        ssh(
            ip, port,
            f'export PATH="$HOME/.local/bin:/usr/games:$PATH"\n'
            f"cd {WORKDIR}\n"
            f"uv run {args.command}\n",
            check=False,
        )

        fetch_results(ip, port)
    finally:
        report_cost(pod_id)
        if args.keep:
            print(f"\npod {pod_id} left running -- terminate with:")
            print(f'  uv run python -c "import runpod,os;'
                  f'runpod.api_key=os.environ[\'RUNPOD_API_KEY\'];'
                  f"runpod.terminate_pod('{pod_id}')\"")
        else:
            runpod.terminate_pod(pod_id)
            print(f"\npod {pod_id} terminated")


if __name__ == "__main__":
    main()
