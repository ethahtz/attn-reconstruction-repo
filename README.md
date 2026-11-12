# Attention reconstruction: blog reproduction

Reproduce the controlled experiments in [Are we interpreting the model of interest?](https://ethahtz.github.io/blogs/model_of_interest.html): attention-backend differences in logits and greedy generation, precision controls, and faithful attention extraction using NNsight identity-value passes.

`reproduce_blog.py` runs the complete suite by default and writes the quantitative tables as CSV. This repository contains only the experiment code, setup scripts, and fixed input manifests. Model weights and COCO images are downloaded when needed. Environments, downloaded inputs, raw tensors, and generated outputs stay outside Git.

## Setup and run

The tested environment uses Python 3.11 on Linux x86_64, PyTorch **2.11.0+cu128**, torchvision **0.26.0+cu128**, Transformers **5.19.0**, NNsight **0.7.0**, and FlashAttention **2.8.3.post1**. Direct and transitive Python dependencies are pinned in `requirements.txt`; the Torch wheels and FA2 source archive also have SHA-256 pins.

Provide an NVIDIA GPU supporting BF16, a compatible driver, and the CUDA toolkit. The verified build used CUDA **12.9.0** (`nvcc` 12.9.41). The supplied batch script loads `cuda/12.9.0-cinr` and requests any GPU in `gpu-he`; edit its account, partition and module settings for another cluster.

Read `run.sbatch`, then create the venv and compile FA2 in a GPU allocation:

```bash
sbatch --wait run.sbatch --setup
```

Run every experiment:

```bash
sbatch --wait run.sbatch
```

Or, inside an existing GPU Slurm allocation with the environment ready:

```bash
.venv/bin/python reproduce_blog.py
```

Setup uses Python `venv`, builds single-threaded in node-local scratch, and runs actual BF16 matmul, ordinary SDPA and external FA2 kernels before declaring the environment ready. The default FA2 build targets SM 80/90/100/120; this can take several hours. Build metadata and GPU checks are saved under `outputs/environment/`. No script submits additional jobs.

Environment overrides:

- `ATTENTION_PYTHON`: Python 3.11 executable for setup (default `python3.11`).
- `ATTENTION_ENV`: venv directory (default `.venv`).
- `ATTENTION_CUDA_MODULE`: CUDA module name; set to an empty string if CUDA is already available without a module.
- `FLASH_ATTN_CUDA_ARCHS`: semicolon-separated build targets (default `80;90;100;120`). Retain targets for every GPU on which the environment will be used.
- `HF_HOME`: optional existing Hugging Face cache. Otherwise the batch wrapper uses node-local scratch.

FA2 is built from the release archive pinned in `setup_environment.py`, with the tested build-only change from C++17 to C++20. Its attention source is otherwise unchanged.

## Run selected experiments

```bash
sbatch --wait run.sbatch --part part1
sbatch --wait run.sbatch --part recovery
sbatch --wait run.sbatch --part logits precision
```

The same arguments work with `.venv/bin/python reproduce_blog.py` inside a GPU allocation.

| Part | Experiments |
|---|---|
| `all` (default) | Every main experiment and precision appendix |
| `part1` | BF16 backend comparisons and VLM precision comparison |
| `part2` or `recovery` | Decoder attention recovery and real-value reconstruction, plus separate counterfactual controls |
| `generation` | BF16 greedy continuation comparisons across all three checkpoints |
| `logits` | BF16 fixed-prefix logit differences across all three checkpoints |
| `precision` | VLM BF16/FP16/FP32 comparisons |
| `text` | Qwen3-4B and GPT-2 BF16 controls |
| `appendix` | Precision comparisons across all three checkpoints |

`--output-dir PATH` selects a run directory. Compatible completed reports are reused. Partial or incompatible reports are preserved and produce an error; use a new output directory to restart. Do not combine reports from different environments.

Planning and analysis run directly on CPU, without Slurm:

```bash
python3 reproduce_blog.py --dry-run
.venv/bin/python reproduce_blog.py --mode analyze --output-dir outputs/blog-reproduction
```

Analysis requires the completed run's raw logit tensors and recovery report. No historical measurements from another checkout are required or bundled.

## Experiment protocol

- Checkpoints: **Qwen3-VL-4B-Instruct**, **Qwen3-4B**, and **GPT-2**, each pinned to an immutable checkpoint revision in `reproduce_blog.py`.
- Inputs: the fixed 20 COCO validation images in `coco20.json` with **“Describe the image.”**; Qwen3-4B and GPT-2 use the same 20 literal prose prefixes in `continuation20.json`, without a chat template or extra special tokens.
- Backends: HF eager, ordinary PyTorch SDPA with automatic dispatch, and external FA2 in BF16/FP16. FP32 compares eager and ordinary SDPA. We hold the checkpoint, processed input, masks and cache settings fixed within each comparison.
- Fixed-prefix logits and recovery use uncached prefixes. Greedy generation uses the cache and a 128-token cap.
- Recovery selects three decoder layers and three heads per layer: **3 models × 20 inputs × 3 backends × 9 heads = 1,620 real-value comparisons**. Vision heads are excluded. The three counterfactual trials per head yield **4,860 separate comparisons**.
- NNsight probes preserve Q/K, masks, dtype and backend, and restore the genuine pre-output-projection activation before downstream computation continues. Evaluation compares native pre-projection outputs with recovered `A @ V` using float64 analytical multiplication.
- Eager is a comparison reference. Finite-precision errors, repeat variation, actual attention operators, input hashes and source hashes are retained. Layer/token research arrays and image-token boundaries are saved with raw outputs.

The full default suite runs nine checkpoint/precision configurations and records 480 greedy continuations. The motivating behavioral benchmark anecdote is outside this controlled reproduction suite.

## Quantitative outputs

The default run writes to `outputs/blog-reproduction/`:

- `runs/`: comparison reports, processed inputs, logits, generations and research arrays.
- `attention/`: recovery report and saved attention/value/output tensors.
- `tables/`: CSV tables and detailed measurements.
- `figures/`: generated BF16 logit comparison figure.
- `provenance.json`: dependency versions and hashes of scripts, requirements, input manifests and reports.

| CSV in `tables/` | Result |
|---|---|
| `table_backend_continuations.csv` | Different BF16 greedy continuations |
| `table_vlm_bf16_logit_rmse.csv` | VLM next-token logit RMSE |
| `table_vlm_bf16_max_abs.csv` | VLM maximum logit differences |
| `table_bf16_pairwise_logits.csv` | Pairwise logit comparisons for all checkpoints |
| `table_vlm_precision.csv` | Main VLM precision comparison |
| `table_vlm_precision_full.csv` | Full FP16/FP32 VLM comparisons |
| `table_real_value_reconstruction.csv` | Real-value reconstruction by model/backend versus eager-returned weights |
| `table_real_value_reconstruction_aggregate.csv` | Aggregated decoder reconstruction errors |
| `table_eager_vlm_precision.csv` | Eager VLM precision matrix |
| `table_eager_text_precision.csv` | Eager text-model precision differences |
| `table_text_precision_backend_controls.csv` | Text-model precision/backend controls |

Additional CSVs contain generations/token IDs, first-token changes, per-head reconstruction errors, counterfactual decoder controls, worked identity-value examples, and backend errors/repeat variation. Columns ending in `_percent` contain percentages; count columns are integers.

The verified GPU rerun preserved the main trends using this environment. Its mean real-value reconstruction error was **0.182589%**, ranging from **0.072612% to 0.318774%** over the 1,620 selected decoder heads. Exact numbers and generated text can change with the GPU or software stack; the original blog used an earlier CUDA/PyTorch environment.

The reproduction entry point uses `compare.py` for backend comparisons, `evaluate_reconstruction.py` for the reconstruction measurements, and the local NNsight extractor in `src/faithful_attention_extraction/`. The entry point imports the extractor directly from this repository. The repository uses the MIT license.
