#!/usr/bin/env python3
"""Reproduce the current blog's experiments and export its tables as CSV.

Default: run all experiments inside a GPU Slurm allocation.
Examples (run from the repository root):
    .venv/bin/python reproduce_blog.py
    .venv/bin/python reproduce_blog.py --part part1
    .venv/bin/python reproduce_blog.py --part recovery
    .venv/bin/python reproduce_blog.py --mode analyze --part precision
    .venv/bin/python reproduce_blog.py --dry-run

Uses compare.py and evaluate_reconstruction.py from this checkout, preserving the
measured processing, masking, cache, dispatch, and restoration logic.
No jobs are submitted, no packages installed, and no scheduler polling occurs.
"""
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent
BACKENDS = ("eager", "sdpa", "fa2")
DTYPES = ("bfloat16", "float16", "float32")
CHECKPOINTS = {
    "vlm": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    "lm": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
}
IMPLEMENTATIONS = {"eager": "eager", "sdpa": "sdpa", "fa2": "flash_attention_2"}
EXPANSIONS = {
    "part1": {"generation", "logits", "precision", "text"},
    "part2": {"recovery"},
    "all": {"generation", "logits", "precision", "text", "recovery", "appendix"},
}
PARTS = tuple(EXPANSIONS) + ("generation", "logits", "precision", "text", "recovery", "appendix")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=PARTS, nargs="+", default=["all"],
                        help="Parts to run/export. Multiple parts may be combined. Default: all.")
    parser.add_argument("--mode", choices=("run", "analyze"), default="run",
                        help="run: fresh experiments, reusing matching completed outputs; "
                             "analyze: this output directory's completed runs only.")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/blog-reproduction",
                        help="Reports, raw tensors, CSV tables, and generated figures.")
    parser.add_argument("--dry-run", action="store_true", help="Print plans without inference or writing files.")
    return parser.parse_args(argv)


def expanded_parts(parts):
    result = set()
    for part in parts:
        result.update(EXPANSIONS.get(part, {part}))
    return result


def configurations(parts):
    required = set()
    if parts & {"generation", "logits", "text", "recovery"}:
        groups = ("lm", "gpt2") if parts == {"text"} else CHECKPOINTS
        required.update((group, "bfloat16") for group in groups)
    if "precision" in parts:
        required.update(("vlm", dtype) for dtype in DTYPES)
    if "appendix" in parts:
        required.update((group, dtype) for group in CHECKPOINTS for dtype in DTYPES)
    return [(group, dtype) for group in CHECKPOINTS for dtype in DTYPES if (group, dtype) in required]


def comparison_command(group, dtype, output):
    checkpoint, revision = CHECKPOINTS[group]
    args = [sys.executable, str(REPO / "compare.py"), "--model", checkpoint,
            "--revision", revision, "--dtype", dtype, "--max-new-tokens", "128"]
    if group == "vlm":
        args += ["--coco", str(REPO / "coco20.json"), "--prompt", "Describe the image."]
    else:
        args += ["--text-prompts", str(REPO / "continuation20.json"), "--raw-text"]
    return args + ["--backends", *(BACKENDS[:2] if dtype == "float32" else BACKENDS),
                   "--output", str(output)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_allocation(gpu=False):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run fresh GPU inference inside a Slurm allocation; CPU analysis needs no allocation.")
    if gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Fresh model inference requires an allocated CUDA GPU.")


def backend_rows(report, backend):
    matches = [r for r in report["results"] if r["backend"] == backend]
    if len(matches) != 1 or matches[0].get("status") != "ok":
        raise ValueError(f"Missing/failed backend {backend}: {matches}")
    rows = {str(row["image_id"]): row for row in matches[0]["images"]}
    if len(rows) != 20 or len(matches[0]["images"]) != 20:
        raise ValueError(f"Expected exactly 20 unique inputs for {backend}")
    return rows


def validate_report(report, group, dtype):
    if report["model"] != CHECKPOINTS[group][0] or report["revision"] != CHECKPOINTS[group][1]:
        raise ValueError(f"Wrong checkpoint/revision in {group}/{dtype}")
    if report["dtype"] != dtype or report["cache"] is not False:
        raise ValueError(f"Wrong precision/cache settings in {group}/{dtype}")
    generation = report["generation"]
    if (generation["do_sample"] is not False or generation["use_cache"] is not True
            or generation["max_new_tokens"] != 128 or generation["num_beams"] != 1):
        raise ValueError("Expected cached greedy generation with a 128-token cap")
    if group != "vlm" and report["input_format"] != "raw_text":
        raise ValueError("Text checkpoints must receive raw prefixes without chat templates")
    if group == "vlm" and report["prompt"] != "Describe the image.":
        raise ValueError("VLM prompt does not match the blog")
    expected_ids = ({str(x["image_id"]) for x in json.loads((REPO / "coco20.json").read_text())["images"]}
                    if group == "vlm" else {f"text-{i:02d}" for i in range(20)})
    examples = {str(x["image_id"]): x for x in report["examples"]}
    if set(examples) != expected_ids:
        raise ValueError("Report input IDs do not match the current manifest")
    if group != "vlm":
        prefixes = json.loads((REPO / "continuation20.json").read_text())
        if any(examples[f"text-{i:02d}"]["prompt"] != prefix for i, prefix in enumerate(prefixes)):
            raise ValueError("Report uses a different text-prefix manifest")
    for backend in BACKENDS[:2] if dtype == "float32" else BACKENDS:
        if set(backend_rows(report, backend)) != expected_ids:
            raise ValueError("Backend input IDs do not match the current manifest")


def write_csv(directory, filename, rows, fields=None):
    rows = list(rows)
    if fields is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty table {filename}")
        fields = list(rows[0])
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV {path.name}: {len(rows)} rows", flush=True)
    return path


def mean(values):
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected nonempty finite measurements")
    return math.fsum(values) / len(values)


class Reproduction:
    def __init__(self, args):
        self.args = args
        self.parts = expanded_parts(args.part)
        self.root = args.output_dir.resolve()
        self.csv_dir = self.root / "tables"
        self.paths = {(g, d): self.root / "runs" / f"{g}-{d}.json"
                      for g, d in configurations(self.parts)}
        self.reports = {}
        self.stats = {}
        self.table_sources = {}
        self.attention_sources = {}

    def export(self, filename, rows, fields=None):
        result = write_csv(self.csv_dir, filename, rows, fields)
        self.table_sources[filename] = str(result)
        return result

    def prepare(self):
        for (group, dtype), path in self.paths.items():
            if self.args.mode == "run" and not path.exists():
                require_allocation(gpu=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                command = comparison_command(group, dtype, path)
                print(shlex.join(command), flush=True)
                subprocess.run(command, cwd=REPO, check=True)
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed report: {path}")
            report = json.loads(path.read_text())
            validate_report(report, group, dtype)
            self.reports[group, dtype] = report

    def pairwise(self, group, dtype):
        key = (group, dtype)
        if key in self.stats:
            return self.stats[key]
        import torch
        report = self.reports[key]
        methods = BACKENDS[:2] if dtype == "float32" else BACKENDS
        rows = {b: backend_rows(report, b) for b in methods}
        ids = list(rows[methods[0]])
        pairs = {(a, b): {"rmse": [], "max_abs": [], "changed": 0, "first": 0}
                 for a, b in combinations(methods, 2)}
        for input_id in ids:
            logits = {}
            for backend in methods:
                raw = self.paths[key].with_suffix("") / f"{input_id}-{backend}.pt"
                if not raw.is_file():
                    raise FileNotFoundError(f"Raw logits are needed for pairwise metrics: {raw}")
                # Large hidden-state tensors remain memory mapped and are not copied.
                payload = torch.load(raw, map_location="cpu", weights_only=True, mmap=True)
                logits[backend] = payload["logits"][0].double().clone()
                del payload
            for (a, b), metrics in pairs.items():
                delta = logits[a] - logits[b]
                metrics["rmse"].append(delta.square().mean().sqrt().item())
                metrics["max_abs"].append(delta.abs().max().item())
                left, right = rows[a][input_id]["generated_ids"], rows[b][input_id]["generated_ids"]
                if not left or not right:
                    raise ValueError("Expected nonempty generated token sequences")
                metrics["changed"] += int(left != right)
                metrics["first"] += int(left[0] != right[0])
        result = [{"model": CHECKPOINTS[group][0], "dtype": dtype, "backend_a": a,
                   "backend_b": b, "inputs": len(ids), "mean_logit_rmse": mean(v["rmse"]),
                   "mean_per_input_max_abs": mean(v["max_abs"]),
                   "different_sequences": v["changed"], "different_first_tokens": v["first"]}
                  for (a, b), v in pairs.items()]
        self.stats[key] = result
        return result

    def export_generations(self):
        rows = []
        prefixes = json.loads((REPO / "continuation20.json").read_text())
        for (group, dtype), report in self.reports.items():
            for entry in report["results"]:
                if entry.get("status") != "ok" or entry["backend"] not in BACKENDS:
                    continue
                for row in entry["images"]:
                    key = str(row["image_id"])
                    prompt = report["prompt"] if group == "vlm" else prefixes[int(key.split("-")[-1])]
                    rows.append({"group": group, "dtype": dtype, "input": key,
                                 "backend": entry["backend"], "prompt": prompt,
                                 "generated_ids": json.dumps(row["generated_ids"]),
                                 "generated_text": row["generated_text"]})
        self.export("generation_examples.csv", rows)
        if self.parts & {"generation", "text"}:
            groups = CHECKPOINTS if "generation" in self.parts else ("lm", "gpt2")
            changes = []
            for group in groups:
                report = self.reports[group, "bfloat16"]
                reference = backend_rows(report, "eager")
                changes.append({"model": CHECKPOINTS[group][0], "inputs": 20, **{
                    f"{backend}_vs_eager_different_sequences": sum(
                        row["generated_ids"] != reference[key]["generated_ids"]
                        for key, row in backend_rows(report, backend).items())
                    for backend in ("sdpa", "fa2")}})
            self.export("table_backend_continuations.csv", changes)

    def export_logits(self):
        if not self.parts & {"logits", "text"}:
            return
        groups = CHECKPOINTS if "logits" in self.parts else ("lm", "gpt2")
        rows = [row for group in groups for row in self.pairwise(group, "bfloat16")]
        self.export("table_bf16_pairwise_logits.csv", rows)
        if "logits" in self.parts:
            vlm = self.pairwise("vlm", "bfloat16")
            self.export("table_vlm_bf16_logit_rmse.csv", [
                {k: r[k] for k in ("backend_a", "backend_b", "inputs", "mean_logit_rmse")} for r in vlm])
            self.export("table_vlm_bf16_max_abs.csv", [
                {k: r[k] for k in ("backend_a", "backend_b", "inputs", "mean_per_input_max_abs")} for r in vlm])
            self.plot_logits()

    def plot_logits(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        upper = max(row["mean_logit_rmse"] for group in CHECKPOINTS
                    for row in self.pairwise(group, "bfloat16"))
        for ax, group in zip(axes, CHECKPOINTS):
            matrix = [[0.0] * 3 for _ in range(3)]
            for row in self.pairwise(group, "bfloat16"):
                i, j = BACKENDS.index(row["backend_a"]), BACKENDS.index(row["backend_b"])
                matrix[i][j] = matrix[j][i] = row["mean_logit_rmse"]
            image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=upper)
            ax.set_xticks(range(3), ["Eager", "SDPA", "FA2"])
            ax.set_yticks(range(3), ["Eager", "SDPA", "FA2"])
            ax.set_title(CHECKPOINTS[group][0].split("/")[-1])
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{matrix[i][j]:.4f}", ha="center", va="center",
                            color="black" if matrix[i][j] > upper * .7 else "white")
        fig.colorbar(image, ax=axes, label="Mean next-token logit RMSE", shrink=.8)
        directory = self.root / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "bf16-model-logit-matrices.png", dpi=180)
        plt.close(fig)

    def export_precision(self):
        if "precision" in self.parts:
            rows = [next(r for r in self.pairwise("vlm", dtype)
                         if r["backend_a"] == "eager" and r["backend_b"] == "sdpa")
                    for dtype in DTYPES]
            self.export("table_vlm_precision.csv", rows)
            self.export("table_vlm_precision_full.csv", [
                row for dtype in ("float16", "float32") for row in self.pairwise("vlm", dtype)])
        if "appendix" not in self.parts:
            return
        summaries = []
        first_rows = []
        for group in CHECKPOINTS:
            sequence_counts = {}
            first_counts = {}
            examples = []
            for dtype in DTYPES:
                examples.append({str(x["image_id"]): x["token_ids"]
                                 for x in self.reports[group, dtype]["examples"]})
            if not examples[0] == examples[1] == examples[2]:
                raise ValueError("Prefix token IDs changed across precision runs")
            for a, b in combinations(DTYPES, 2):
                left = backend_rows(self.reports[group, a], "eager")
                right = backend_rows(self.reports[group, b], "eager")
                if left.keys() != right.keys():
                    raise ValueError("Precision runs use different input IDs")
                sequence_counts[a, b] = sum(left[k]["generated_ids"] != right[k]["generated_ids"] for k in left)
                first_counts[a, b] = sum(left[k]["generated_ids"][0] != right[k]["generated_ids"][0] for k in left)
                first_rows.append({"model": CHECKPOINTS[group][0], "precision_a": a, "precision_b": b,
                                   "inputs": 20, "different_sequences": sequence_counts[a, b],
                                   "different_first_tokens": first_counts[a, b]})
            if group == "vlm":
                matrix_rows = []
                for a in DTYPES:
                    values = {}
                    for b in DTYPES:
                        key = (a, b) if (a, b) in sequence_counts else (b, a)
                        values[b] = 0 if a == b else sequence_counts[key]
                    matrix_rows.append({"eager_precision": a, **values})
                self.export("table_eager_vlm_precision.csv", matrix_rows)
            else:
                summaries.append({"model": CHECKPOINTS[group][0], "inputs": 20, **{
                    f"{a}_vs_{b}": sequence_counts[a, b] for a, b in combinations(DTYPES, 2)}})
        self.export("table_eager_text_precision.csv", summaries)
        self.export("eager_precision_first_tokens.csv", first_rows)
        controls = []
        for group in ("lm", "gpt2"):
            for dtype in ("float16", "float32"):
                metrics = self.pairwise(group, dtype)
                sdpa = next(r for r in metrics if (r["backend_a"], r["backend_b"]) == ("eager", "sdpa"))
                fa2 = next((r for r in metrics if r["backend_b"] == "fa2" and r["backend_a"] == "eager"), None)
                controls.append({"model": CHECKPOINTS[group][0], "dtype": dtype, "inputs": 20,
                                 "sdpa_vs_eager_different_sequences": sdpa["different_sequences"],
                                 "fa2_vs_eager_different_sequences": fa2["different_sequences"] if fa2 else "Unavailable",
                                 "mean_sdpa_logit_rmse": sdpa["mean_logit_rmse"]})
        self.export("table_text_precision_backend_controls.csv", controls)

    def run_recovery(self, output):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
        # Research checks stay in the research validator, not in the published package.
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "src"))
        from evaluate_reconstruction import validate, validation_heads, forward_kwargs, _discover

        require_allocation(gpu=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {"dtype": "bfloat16", "seed": 20261013, "held_out_trials": 3,
                  "gpu": torch.cuda.get_device_name(), "job": os.environ.get("SLURM_JOB_ID"),
                  "scope": "decoder-only", "validator_sha256": sha256(REPO / "evaluate_reconstruction.py"),
                  "cases": []}
        for group, (checkpoint, revision) in CHECKPOINTS.items():
            snapshot = snapshot_download(checkpoint, revision=revision, max_workers=1,
                                         allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
            input_names = ([str(x["image_id"]) for x in json.loads((REPO / "coco20.json").read_text())["images"]]
                           if group == "vlm" else [f"text-{i:02d}" for i in range(20)])
            loader = AutoModelForImageTextToText if group == "vlm" else AutoModelForCausalLM
            eager_patterns = {}
            for backend in BACKENDS:
                model = loader.from_pretrained(
                    snapshot, dtype=torch.bfloat16, attn_implementation=IMPLEMENTATIONS[backend],
                ).to("cuda").eval().requires_grad_(False)
                heads = validation_heads(_discover(model), include_vision=False)
                if len(heads) != 9 or any(s != "text" for s, _, _ in heads):
                    raise ValueError("Expected nine fixed decoder heads")
                for index, input_id in enumerate(input_names):
                    input_file = self.paths[group, "bfloat16"].with_suffix("") / f"{input_id}-inputs.pt"
                    inputs = torch.load(input_file, map_location="cuda", weights_only=True)
                    with torch.no_grad(), torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as profiler:
                        model(**inputs, **forward_kwargs(model))
                    ops = sorted({e.key for e in profiler.key_averages()
                                  if "attention" in e.key.lower() or "flash" in e.key.lower()})
                    if backend == "sdpa" and not any("aten::scaled_dot_product_attention" in op for op in ops):
                        raise ValueError(f"Missing ordinary SDPA dispatch evidence: {ops}")
                    if backend == "fa2" and not any("flashattn" in op.replace("_", "").lower() for op in ops):
                        raise ValueError(f"Missing external FA2 dispatch evidence: {ops}")
                    result, raw = validate(model, inputs, heads, seed=20261013+index*1000000,
                                           eager_patterns=eager_patterns.get(input_id))
                    if backend == "eager":
                        eager_patterns[input_id] = raw["patterns"]
                    result.update(model_group=group, model=checkpoint, revision=revision, backend=backend,
                                  input=input_id, input_sha256=sha256(input_file), attention_ops=ops)
                    report["cases"].append(result)
                    torch.save(raw, output.parent / f"{group}-{input_id}-{backend}.pt")
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"Recovery: {group} {backend} {input_id}", flush=True)
                    del raw, result, inputs, profiler
                    gc.collect()
                del model
                gc.collect()
                torch.cuda.empty_cache()
        return report

    def recovery_cases(self):
        path = self.root / "attention/report.json"
        if self.args.mode == "run" and not path.exists():
            self.run_recovery(path)
        sources = [(path, set(CHECKPOINTS))]
        cases = []
        for path, groups in sources:
            data = json.loads(path.read_text())
            if data["dtype"] != "bfloat16":
                raise ValueError("Expected BF16 recovery")
            self.attention_sources[str(path)] = sha256(path)
            for case in data["cases"]:
                if case["model_group"] in groups:
                    if case["model"] != CHECKPOINTS[case["model_group"]][0]:
                        raise ValueError("Wrong recovery checkpoint")
                    if case["revision"] != CHECKPOINTS[case["model_group"]][1]:
                        raise ValueError("Wrong recovery revision")
                    heads = [h for h in case["heads"] if h["stream"] == "text"]
                    cases.append({**case, "heads": heads})
        addresses = [(c["model_group"], c["backend"], str(c["input"])) for c in cases]
        expected = Counter({(g, b): 20 for g in CHECKPOINTS for b in BACKENDS})
        if len(set(addresses)) != 180 or len(addresses) != 180:
            raise ValueError("Recovery report is incomplete or contains duplicate cases; keep it and rerun in a new output directory")
        if Counter((c["model_group"], c["backend"]) for c in cases) != expected:
            raise ValueError("Recovery cases do not match the experiment design")
        if any(len(c["heads"]) != 9 for c in cases):
            raise ValueError("Expected nine decoder heads per case")
        return cases

    def export_recovery(self):
        if "recovery" not in self.parts:
            return
        self.export_identity_examples()
        cases = self.recovery_cases()
        groups = defaultdict(list)
        detailed = []
        for case in cases:
            if case["restored_logit_max_abs"] != 0:
                raise ValueError("Nonzero restoration delta; inspect the saved report")
            for head in case["heads"]:
                groups[case["model_group"], case["backend"]].append(head)
                if case["backend"] == "eager" and head["native_eager"]["max_abs"] != 0:
                    raise ValueError("Recovered eager reference does not exactly match native weights")
                detailed.append({"model": case["model"], "backend": case["backend"], "input": case["input"],
                                 "layer": head["layer"], "head": head["head"],
                                 "recovered_error_percent": 100*head["original_v"]["relative_rmse"],
                                 "eager_error_percent": 100*head["original_v_eager"]["relative_rmse"]})
        self.export("real_value_reconstruction_per_head.csv", detailed)
        rows = []
        for group in CHECKPOINTS:
            for backend in BACKENDS:
                heads = groups[group, backend]
                own = [100*h["original_v"]["relative_rmse"] for h in heads]
                eager = [100*h["original_v_eager"]["relative_rmse"] for h in heads]
                rows.append({"model": CHECKPOINTS[group][0], "backend": backend, "comparisons": len(heads),
                             "recovered_error_percent": mean(own), "eager_error_percent": mean(eager),
                             "recovered_min_percent": min(own), "recovered_max_percent": max(own)})
        self.export("table_real_value_reconstruction.csv", rows)
        aggregate = []
        for backend in (*BACKENDS, None):
            selected = [r for r in detailed if backend is None or r["backend"] == backend]
            own = [r["recovered_error_percent"] for r in selected]
            aggregate.append({"backend": backend or "all", "comparisons": len(selected),
                              "recovered_error_percent": mean(own),
                              "eager_error_percent": mean([r["eager_error_percent"] for r in selected]),
                              "recovered_min_percent": min(own), "recovered_max_percent": max(own)})
        self.export("table_real_value_reconstruction_aggregate.csv", aggregate)
        # Preserve the separate original research evaluation without mixing its
        # three trials/head into the current blog's real-value denominator.
        counterfactual = []
        for group in CHECKPOINTS:
            for backend in BACKENDS:
                trials = [t for h in groups[group, backend] for t in h["held_out"]]
                if len(trials) != 540 or any(t["qk_max_abs"] != 0 for t in trials):
                    raise ValueError("Counterfactual count/QK preservation does not match the protocol")
                counterfactual.append({"model": CHECKPOINTS[group][0], "backend": backend, "comparisons": len(trials),
                                       "recovered_error_percent": mean([100*t["relative_rmse"] for t in trials]),
                                       "eager_error_percent": mean([100*t["eager_relative_rmse"] for t in trials]),
                                       "reversed_columns_error_percent": mean([100*t["scrambled_relative_rmse"] for t in trials])})
        self.export("research_counterfactual_decoder.csv", counterfactual)

    def export_identity_examples(self):
        import torch
        generator = torch.Generator().manual_seed(0)
        rows = []
        for length, head_dim in ((2, 4), (6, 2)):
            scores = torch.randn(length, length, generator=generator, dtype=torch.float64)
            attention = scores.softmax(-1)
            pieces = []
            for offset in range(0, length, head_dim):
                width = min(head_dim, length-offset)
                values = torch.zeros(length, head_dim, dtype=attention.dtype)
                values[offset:offset+width, :width] = torch.eye(width, dtype=attention.dtype)
                pieces.append((attention @ values)[:, :width])
            recovered = torch.cat(pieces, dim=-1)
            torch.testing.assert_close(recovered, attention, atol=0, rtol=0)
            rows.append({"sequence_length": length, "head_dimension": head_dim,
                         "passes": len(pieces), "max_abs_recovery_error": (attention-recovered).abs().max().item()})
        self.export("identity_value_examples.csv", rows)

    def export_audit(self):
        rows = []
        for (group, dtype), report in self.reports.items():
            for result in report["results"]:
                images = result.get("images", [])
                rows.append({"model": CHECKPOINTS[group][0], "dtype": dtype, "backend": result["backend"],
                             "status": result.get("status"), "error": result.get("error", ""),
                             "repeat_max_abs": max((r["repeat_max_abs"] for r in images), default=None),
                             "profile_max_abs": max((r["profile_max_abs"] for r in images), default=None),
                             "attention_ops": json.dumps(sorted({op for r in images for op in r.get("attention_ops", [])}))})
        self.export("backend_audit.csv", rows)
        packages = {}
        for name in ("torch", "transformers", "nnsight", "flash-attn", "numpy", "matplotlib"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        manifest = {"mode": self.args.mode, "parts": sorted(self.parts), "packages": packages,
                    "script_sha256": sha256(__file__),
                    "extractor_source_sha256": sha256(REPO / "src/faithful_attention_extraction/_core.py"),
                    "requirements_sha256": sha256(REPO / "requirements.txt"),
                    "comparison_source_sha256": sha256(REPO / "compare.py"),
                    "validator_source_sha256": sha256(REPO / "evaluate_reconstruction.py"),
                    "input_manifest_sha256": {name: sha256(REPO / name) for name in ("coco20.json", "continuation20.json")},
                    "reports": {f"{g}/{d}": {"path": str(p), "sha256": sha256(p)}
                                for (g, d), p in self.paths.items()},
                    "attention_reports": self.attention_sources, "csv_files": self.table_sources,
                    "units": "errors ending _percent are percentages; logits use raw logit units; counts are integers"}
        (self.root / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(self):
        self.prepare()
        self.export_generations()
        self.export_logits()
        self.export_precision()
        self.export_recovery()
        self.export_audit()


def main(argv=None):
    args = parse_args(argv)
    reproduction = Reproduction(args)
    if args.dry_run:
        print("Parts:", ", ".join(sorted(reproduction.parts)))
        for (group, dtype), output in reproduction.paths.items():
            print(shlex.join(comparison_command(group, dtype, output))
                  if args.mode == "run" else f"Read {output}")
        if "recovery" in reproduction.parts:
            print("Decoder recovery: 180 cases, 1,620 original-value comparisons; three separate counterfactual trials/head.")
        print("CSV directory:", reproduction.csv_dir)
        return 0
    if args.mode == "run":
        require_allocation(gpu=True)
    # Honor the project-wide single-agent lock. Hold the descriptor throughout
    # all child comparisons so another local agent cannot overlap this run.
    lock_path = Path("/tmp") / f"{os.environ.get('USER', 'attention-reproduction')}.agent-lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                          TOKENIZERS_PARALLELISM="false")
        import torch
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        reproduction.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
