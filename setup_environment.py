#!/usr/bin/env python3
"""Build the tested Python 3.11 venv and pinned FlashAttention 2 inside a GPU job."""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

REPO = Path(__file__).resolve().parent
ENV = Path(os.environ.get("ATTENTION_ENV", str(REPO / ".venv"))).resolve()
PYTHON = ENV / "bin/python"
OUTPUT = REPO / "outputs/environment"
SOURCE_URL = "https://files.pythonhosted.org/packages/01/7a/92a46e7cd6bbb4d7b2855a457c3b855df54a97af5656d98fc92e58e61065/flash_attn-2.8.3.post1.tar.gz"
SOURCE_SHA256 = "55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348"
DEFAULT_ARCHITECTURES = "80;90;100;120"


def command(args, **kwargs):
    print("RUN", " ".join(map(str, args)), flush=True)
    return subprocess.run([str(arg) for arg in args], cwd=REPO, check=True, **kwargs)


def write_json(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n")


def check_gpu():
    from importlib.metadata import version
    import torch
    import torchvision
    import flash_attn_2_cuda
    from flash_attn import flash_attn_func

    expected = {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128",
                "transformers": "5.19.0", "nnsight": "0.7.0", "flash-attn": "2.8.3.post1"}
    for name, pin in expected.items():
        if version(name) != pin:
            raise RuntimeError(f"Expected {name}=={pin}; found {version(name)}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU supporting BF16 is required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    x = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    if not torch.isfinite(x @ x).all():
        raise RuntimeError("Nonfinite GPU matmul")
    q, k, v = [torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    native = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
    external = flash_attn_func(q, k, v, dropout_p=0., causal=True)
    torch.cuda.synchronize()
    if not torch.isfinite(native).all() or not torch.isfinite(external).all():
        raise RuntimeError("Nonfinite SDPA/FA2 output")
    architectures = subprocess.check_output(["cuobjdump", "--list-elf", flash_attn_2_cuda.__file__], text=True)
    for arch in os.environ.get("FLASH_ATTN_CUDA_ARCHS", DEFAULT_ARCHITECTURES).split(";"):
        if f"sm_{arch}" not in architectures:
            raise RuntimeError(f"FA2 binary is missing sm_{arch}")
    manifest = {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                "packages": expected, "cuda_runtime": torch.version.cuda,
                "torch_architectures": torch.cuda.get_arch_list(),
                "flash_binary_sha256": hashlib.sha256(Path(flash_attn_2_cuda.__file__).read_bytes()).hexdigest(),
                "sdpa_fa2_max_abs": (native.transpose(1, 2).float() - external.float()).abs().max().item(),
                "nvcc": subprocess.check_output(["nvcc", "--version"], text=True)}
    write_json("preflight.json", manifest)
    print("GPU PREFLIGHT PASSED", json.dumps(manifest), flush=True)


def setup():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use Python 3.11; the pinned wheels are CPython 3.11 Linux x86_64")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not PYTHON.exists():
        command([sys.executable, "-m", "venv", ENV])
    requirements_sha = hashlib.sha256((REPO / "requirements.txt").read_bytes()).hexdigest()
    architectures = os.environ.get("FLASH_ATTN_CUDA_ARCHS", DEFAULT_ARCHITECTURES)
    os.environ.update(MAX_JOBS="1", NVCC_THREADS="1", FLASH_ATTENTION_FORCE_BUILD="TRUE",
                      FLASH_ATTN_CUDA_ARCHS=architectures, OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    marker = OUTPUT / "installed.json"
    recorded = json.loads(marker.read_text()) if marker.exists() else {}
    if (recorded.get("requirements_sha256") != requirements_sha or
            recorded.get("source_sha256") != SOURCE_SHA256 or
            recorded.get("architectures") != architectures):
        command([PYTHON, "-m", "pip", "install", "--disable-pip-version-check",
                 "pip==26.2.1", "setuptools==81.0.0", "wheel==0.48.0"])
        command([PYTHON, "-m", "pip", "install", "--disable-pip-version-check",
                 "-r", REPO / "requirements.txt", "--report", OUTPUT / "base-install.json"])
        command([PYTHON, "-m", "pip", "check"])
        # Node-local scratch holds the large temporary CUDA build tree.
        with tempfile.TemporaryDirectory(prefix="attn-reconstruction-fa2-", dir=os.environ.get("TMPDIR", "/tmp")) as scratch:
            with urlopen(SOURCE_URL, timeout=120) as response:
                archive = response.read()
            if hashlib.sha256(archive).hexdigest() != SOURCE_SHA256:
                raise RuntimeError("FlashAttention source archive hash mismatch")
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                tar.extractall(scratch, filter="data")
            setup_file = Path(scratch) / "flash_attn-2.8.3.post1/setup.py"
            original = setup_file.read_text()
            if original.count("-std=c++17") != 4:
                raise RuntimeError("Unexpected FA2 release build flags")
            patched = original.replace("-std=c++17", "-std=c++20")
            setup_file.write_text(patched)
            write_json("flash-build.json", {
                "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
                "build_only_patch": "-std=c++17 -> -std=c++20",
                "original_setup_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "patched_setup_sha256": hashlib.sha256(patched.encode()).hexdigest(),
                "architectures": architectures})
            command([PYTHON, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
                     "--no-cache-dir", "--force-reinstall", "--report", OUTPUT / "flash-install.json", setup_file.parent])
        write_json("installed.json", {"requirements_sha256": requirements_sha,
                                      "source_sha256": SOURCE_SHA256, "architectures": architectures})
    command([PYTHON, "-m", "pip", "check"])
    command([PYTHON, __file__, "--check"])
    with (OUTPUT / "freeze.txt").open("w") as stream:
        command([PYTHON, "-m", "pip", "freeze", "--all"], stdout=stream)
    write_json("ready.json", {"environment": str(ENV), "job": os.environ["SLURM_JOB_ID"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only verify installed packages and execute GPU kernels")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run environment setup and GPU verification inside a GPU Slurm allocation")
    check_gpu() if args.check else setup()


if __name__ == "__main__":
    main()
