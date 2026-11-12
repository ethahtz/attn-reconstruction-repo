#!/bin/bash -l
#SBATCH --job-name=attn-reconstruction
#SBATCH --account=default
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mode=run
if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: sbatch run.sbatch --setup
       sbatch run.sbatch [reproduce_blog.py arguments]
Default: run every blog experiment and export CSV tables.
Examples: --part part1; --part recovery; --part logits precision
Edit Slurm account/partition directives for your cluster.
ATTENTION_PYTHON: Python 3.11 executable (default python3.11)
ATTENTION_ENV: venv path (default .venv)
ATTENTION_CUDA_MODULE: module to load (default cuda/12.9.0-cinr; empty skips module loading)
FLASH_ATTN_CUDA_ARCHS: build targets (default 80;90;100;120)
No additional jobs are submitted.
HELP
    exit 0
fi
if [[ "${1:-}" == "--setup" ]]; then
    mode=setup
    shift
    if [[ "$#" != 0 ]]; then echo "--setup takes no further arguments" >&2; exit 2; fi
fi
attention_env="${ATTENTION_ENV:-.venv}"
# Planning uses only the standard library: no allocation, module, or venv needed.
for argument in "$@"; do
    if [[ "$argument" == "--dry-run" ]]; then
        exec "${ATTENTION_PYTHON:-python3.11}" reproduce_blog.py "$@"
    fi
done
: "${SLURM_JOB_ID:?Run GPU work with sbatch or inside a GPU Slurm allocation}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAX_JOBS=1 NVCC_THREADS=1
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80;90;100;120}"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR="${TMPDIR:-/tmp}/attn-reconstruction-pip-${SLURM_JOB_ID}"
export HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/attn-reconstruction-hf-${SLURM_JOB_ID}}"
unset PYTHONPATH
if [[ -n "${ATTENTION_CUDA_MODULE-cuda/12.9.0-cinr}" ]]; then
    module load "${ATTENTION_CUDA_MODULE-cuda/12.9.0-cinr}"
fi
if [[ "$mode" == "setup" ]]; then
    exec flock -n "/tmp/${USER}.agent-lock" "${ATTENTION_PYTHON:-python3.11}" setup_environment.py
fi
if [[ ! -x "$attention_env/bin/python" ]]; then
    echo "Missing $attention_env; run sbatch run.sbatch --setup first." >&2
    exit 2
fi
flock -n "/tmp/${USER}.agent-lock" "$attention_env/bin/python" setup_environment.py --check
# reproduce_blog.py holds the lock throughout all experiment subprocesses.
exec "$attention_env/bin/python" reproduce_blog.py "$@"
