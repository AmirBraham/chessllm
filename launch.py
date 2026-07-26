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
from runpod.error import QueryError

IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"  # torch 2.8, cu12.8
WORKDIR = "/workspace/chessllm"
# Preference order, passphrase-less automation key first. A key with a
# passphrase cannot be used unattended: ssh silently skips it, falls back to
# password auth, and reports "Permission denied (publickey,password)" -- which
# looks identical to the key never having been registered.
SSH_KEYS = ("runpod_automation", "id_ed25519", "id_rsa")
MIN_VRAM_GB = 24  # GRPO later needs policy + frozen reference + Adam states

# The working tree is uploaded as-is rather than cloned, so a run always
# reflects local edits -- no commit/push cycle to test a one-line change.
# .env is excluded deliberately: the API key must never land on a rented box.
# .venv is excluded because macOS wheels are useless on the pod's Linux.
NO_UPLOAD = [".git", ".venv", "__pycache__", ".env", "runs", ".ipynb_checkpoints"]

# Deliberately not uv. `uv sync` reinstalls torch's whole dependency subtree --
# ~2GB of nvidia-*-cu13 wheels -- even with --no-install-package torch, because
# those are separate packages in the lock. Worse, they land ahead of the
# image's driver-matched torch on sys.path. pip installing only what the image
# lacks avoids the second CUDA stack entirely; transformers declares torch as
# an extra, so nothing here drags one in.
POD_PACKAGES = "chess transformers matplotlib pyarrow huggingface_hub tqdm"

# Debian puts stockfish in /usr/games, which is not on root's PATH.
SYSTEM_SETUP = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq stockfish rsync
mkdir -p {WORKDIR}
echo "--- system setup done ---"
"""

PROJECT_SETUP = f"""
set -euo pipefail
export PATH="/usr/games:$PATH"
cd {WORKDIR}

python -m pip install --no-cache-dir -q {POD_PACKAGES} \
  || python -m pip install --no-cache-dir -q --break-system-packages {POD_PACKAGES}

# Hard gate. torch falls back to CPU silently when its CUDA build is newer
# than the host driver -- a cu13 wheel on a 570.x driver (CUDA 12.8) reports
# cuda.is_available() == False and everything still "works", just far slower
# on a GPU you are paying for.
python - <<'PY'
import sys, torch
print(f"torch {{torch.__version__}}  cuda_available={{torch.cuda.is_available()}}")
if not torch.cuda.is_available():
    print(f"built for CUDA {{torch.version.cuda}}; this host's driver is too old")
    sys.exit(1)
print(f"device: {{torch.cuda.get_device_name(0)}}")
PY
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


def has_passphrase(private_key):
    """True if the key cannot be read without a passphrase."""
    return subprocess.run(
        ["ssh-keygen", "-y", "-P", "", "-f", private_key],
        capture_output=True,
    ).returncode != 0


def ssh_identity(explicit=None):
    """(private_key_path, public_key_text) for an unattended connection.

    Pods created through the web UI get the account's registered keys injected
    automatically. Pods created through the API do not -- the key must be
    handed over as PUBLIC_KEY, which the image's start script writes to
    authorized_keys before starting sshd.
    """
    names = [explicit] if explicit else [f"~/.ssh/{n}" for n in SSH_KEYS]

    for name in names:
        private = os.path.expanduser(name)
        public = f"{private}.pub"
        if not (os.path.exists(private) and os.path.exists(public)):
            continue
        if has_passphrase(private):
            print(f"  skipping {os.path.basename(private)}: passphrase-protected")
            continue
        with open(public) as f:
            return private, f.read().strip()

    raise SystemExit(
        "no passphrase-less SSH key found. Automation needs one:\n"
        "  ssh-keygen -t ed25519 -N '' -f ~/.ssh/runpod_automation"
    )


def create_pod(args, candidates, pubkey):
    """Create a pod on the first candidate GPU with actual capacity.

    A GPU type being listed does not mean one is free: capacity can vanish
    between pricing it and asking for it, and RunPod answers with "no longer
    any instances available". Walking down the price-sorted list turns that
    from a dead end into a few cents per hour.
    """
    for price, gpu in candidates:
        label = f"{gpu['displayName']}  {gpu['memoryInGb']}GB  ${price:.2f}/hr"
        try:
            pod = runpod.create_pod(
                name="chessllm",
                image_name=args.image,
                gpu_type_id=gpu["id"],
                cloud_type=args.cloud,
                container_disk_in_gb=args.disk,
                support_public_ip=True,
                start_ssh=True,
                ports="22/tcp",
                env={"PUBLIC_KEY": pubkey},
                allowed_cuda_versions=args.cuda,
            )
            print(f"got {label}")
            return pod
        except QueryError as exc:
            if "no longer any instances" not in str(exc):
                raise
            print(f"  {label} -- no capacity, trying next")

    raise SystemExit(
        "no GPU with capacity right now. Try --cloud COMMUNITY, "
        "--min-vram 16, or wait a few minutes."
    )


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


def ssh(ip, port, script, key=None, check=True):
    """Run a bash script on the pod, streaming output to this terminal."""
    return subprocess.run(
        ["ssh", *ssh_args(port, key), f"root@{ip}", "bash -s"],
        input=script,
        text=True,
        check=check,
    )


def ssh_args(port, key=None):
    """Common ssh options.

    BatchMode=yes matters: without it, an unregistered SSH key makes ssh fall
    back to a password prompt that no one can answer, and the script hangs
    while the pod bills. With it, ssh fails immediately and the finally block
    tears the pod down.
    """
    args = [
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "BatchMode=yes",
        # Without keepalives a long silent step -- a big download, or minutes
        # of generation with no output -- lets the connection die, and ssh
        # exits 255 with the pod mid-task.
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=20",
    ]
    if key:
        args += ["-i", key]
    return args


def ssh_opts(port, key=None):
    """The same options as one string, for `rsync -e`."""
    return "ssh " + " ".join(ssh_args(port, key))


def upload(ip, port, key=None):
    """Push the local working tree to the pod."""
    excludes = [arg for name in NO_UPLOAD for arg in ("--exclude", name)]
    subprocess.run(
        ["rsync", "-az", "--delete", "-e", ssh_opts(port, key), *excludes,
         "./", f"root@{ip}:{WORKDIR}/"],
        check=True,
    )
    print("working tree uploaded")


def fetch_results(ip, port, key=None):
    """Copy the pod's runs/ directory back into ./runs."""
    os.makedirs("runs", exist_ok=True)
    result = subprocess.run(
        ["rsync", "-az", "-e", ssh_opts(port, key),
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
    parser.add_argument("--cuda", nargs="+", default=["12.8", "12.9"],
                        help="acceptable host CUDA versions; filters out hosts "
                             "whose driver is too old for the image's torch")
    parser.add_argument("--ssh-key", default=None,
                        help="private key to use (default: ssh picks it)")
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

    if args.gpu:
        candidates = [(0.0, runpod.get_gpu(args.gpu))]
    else:
        candidates = priced_gpus(args.min_vram)
        if not candidates:
            raise SystemExit(
                f"no GPU with >= {args.min_vram} GB listed. "
                "Try --min-vram 16, or --list-gpus."
            )

    identity, pubkey = ssh_identity(args.ssh_key)
    print(f"ssh key: {os.path.basename(identity)}")

    pod = create_pod(args, candidates, pubkey)
    pod_id = pod["id"]
    print(f"pod {pod_id} created")

    try:
        ip, port = wait_for_ssh(pod_id)
        print(f"ssh -i {identity} root@{ip} -p {port}\n")

        try:
            ssh(ip, port, SYSTEM_SETUP, identity)
        except subprocess.CalledProcessError:
            raise SystemExit(
                f"\nssh refused the key {os.path.basename(identity)}. The pod "
                "installs whatever PUBLIC_KEY it was created with, so this "
                "means the public half did not match, or sshd was not ready "
                "yet. Retry, or pass --ssh-key to choose a different key."
            )

        upload(ip, port, identity)
        ssh(ip, port, PROJECT_SETUP, identity)

        print(f"\n--- running: {args.command} ---\n")
        ssh(
            ip, port,
            f'export PATH="/usr/games:$PATH"\n'
            f"cd {WORKDIR}\n"
            f"{args.command}\n",
            key=identity,
            check=False,
        )

        fetch_results(ip, port, identity)
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
