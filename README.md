# Attention reconstruction: blog reproduction

Reproduce the experiments in [Are we interpreting the model of interest?](https://ethahtz.github.io/blogs/model_of_interest.html): attention-backend differences in logits and greedy generation, precision controls, and faithful attention extraction using NNsight identity-value passes.

`reproduce_blog.py` runs the complete suite by default and writes the quantitative tables as CSV.

## Setup and run

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

`--output-dir PATH` selects a run directory.

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
