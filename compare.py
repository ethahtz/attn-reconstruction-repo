"""Compare attention backends: next-token logits and every layer output."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
from urllib.request import urlopen
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
import re
from pathlib import Path
import sys
import subprocess

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
TEXT_MODEL = "Qwen/Qwen3-4B"
TEXT_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
GPT2_MODEL = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
FA2 = "flash_attention_2"
MAIN_BACKENDS = ("eager", "sdpa", "fa2")


def code_revision():
    try:
        result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                                 "rev-parse", "HEAD"], text=True, capture_output=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def summarize(reference, candidate):
    if not reference or len(reference) != len(candidate) or not all(map(math.isfinite, reference + candidate)):
        raise ValueError("Expected finite logits with matching nonempty vocabularies")
    delta = [c - r for r, c in zip(reference, candidate)]
    mr, mc = max(reference), max(candidate)
    zr = mr + math.log(math.fsum(math.exp(r - mr) for r in reference))
    zc = mc + math.log(math.fsum(math.exp(c - mc) for c in candidate))
    return {"max_abs": max(map(abs, delta)), "mean_abs": math.fsum(map(abs, delta)) / len(delta),
            "rmse": math.sqrt(math.fsum(d * d for d in delta) / len(delta)),
            "kl_ref_to_candidate": math.fsum(math.exp(r - zr) * (r - c + zc - zr) for r, c in zip(reference, candidate)),
            "same_argmax": max(range(len(reference)), key=reference.__getitem__) == max(range(len(candidate)), key=candidate.__getitem__)}


def hidden_difference(reference, candidate):
    import torch
    if reference.shape != candidate.shape or not reference.numel():
        raise ValueError("Hidden-state shapes must match and be nonempty")
    reference, candidate = reference.double(), candidate.double()
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Hidden states must be finite")
    delta = candidate - reference
    cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=-1, eps=1e-12)
    return {"max_abs": delta.abs().max().item(), "mean_abs": delta.abs().mean().item(),
            "rmse": delta.square().mean().sqrt().item(), "mean_token_cosine": cosine.mean().item()}


def measurement_layers(model, text_only=False):
    classes = {"Qwen3VLVisionBlock", "Qwen3VLTextDecoderLayer", "Qwen3VLVisionPatchMerger", "Qwen3DecoderLayer", "GPT2Block"}
    return {n for n, m in model.named_modules()
            if (type(m).__name__ in classes or n in {"model.language_model.norm", "model.norm", "transformer.ln_f"})
            and not (text_only and ".visual." in n)}


def collect_layers(model, names):
    states, handles = {}, []
    def capture(name):
        def hook(module, inputs, output):
            states[name] = output.detach().cpu().clone()
        return hook
    for name, module in model.named_modules():
        if name in names:
            handles.append(module.register_forward_hook(capture(name)))
    if len(handles) != len(names):
        raise ValueError("Some requested layers were not found")
    return states, handles


def token_arrays(reference, candidate):
    import numpy as np
    import torch
    if set(reference) != set(candidate):
        raise ValueError("Reference and candidate layer names differ")
    groups = {"vision": [], "vision_merged": [], "decoder": []}
    for name in candidate:
        group = "vision" if ".visual.blocks." in name else "vision_merged" if ".visual." in name else "decoder"
        groups[group].append(name)
    arrays, axes = {}, {}
    for group, names in groups.items():
        if not names:
            continue
        names.sort(key=lambda n: re.sub(r"\d+", lambda m: f"{int(m.group()):05d}", n))
        axes[group] = names
        rows = {"rmse": [], "max_abs": [], "cosine_deviation": []}
        for name in names:
            if reference[name].shape != candidate[name].shape:
                raise ValueError("Hidden-state shapes must match")
            a = reference[name].double().reshape(-1, reference[name].shape[-1])
            b = candidate[name].double().reshape_as(a)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Hidden states must be finite")
            delta = b - a
            cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-12).clamp(-1, 1)
            cosine[(delta == 0).all(-1)] = 1
            rows["rmse"].append(delta.square().mean(-1).sqrt().numpy())
            rows["max_abs"].append(delta.abs().max(-1).values.numpy())
            rows["cosine_deviation"].append((1 - cosine).numpy())
        for metric, values in rows.items():
            arrays[f"{group}_{metric}"] = np.stack(values)  # [layer, token position]
    return arrays, axes


def resolve_revision(model, revision):
    """An immutable commit already identifies the revision; aliases need lookup."""
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi
    return HfApi().model_info(model, revision=revision).sha


def attention_backends():
    """Normal HF choices plus explicitly named forced-kernel diagnostics."""
    from torch.nn.attention import SDPBackend
    return [("eager", "eager", None), ("sdpa", "sdpa", None),
            ("sdpa_math", "sdpa", SDPBackend.MATH),
            ("sdpa_flash", "sdpa", SDPBackend.FLASH_ATTENTION),
            ("fa2", FA2, None)]


def encode_text(tokenizer, prompt, raw_text=False, template_options=None):
    """Encode a literal continuation prefix or a user chat message."""
    if raw_text:
        return tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
        **(template_options or {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--snapshot", type=Path, help="Offline snapshot directory named by its pinned commit")
    parser.add_argument("--revision", help="HF revision; resolved to an immutable commit before loading")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path)
    source.add_argument("--text-only", action="store_true", help="Text input only, for either a VLM or a standalone LM")
    source.add_argument("--text-prompts", type=Path, help="JSON list of fixed text prompts")
    source.add_argument("--coco", type=Path, help="Fixed image URL manifest, e.g. coco20.json")
    parser.add_argument("--raw-text", action="store_true", help="Feed standalone LMs literal prefixes without chat or special tokens")
    parser.add_argument("--prompt", default="Describe the image.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--reference", type=Path, help="Reuse a completed run as the eager reference")
    parser.add_argument("--backends", nargs="+", choices=["eager", "sdpa", "sdpa_math", "sdpa_flash", "fa2"], default=list(MAIN_BACKENDS))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("outputs/results.json"))
    args = parser.parse_args()
    if "eager" not in args.backends and not args.reference:
        parser.error("A run without eager requires --reference")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run model inference inside a Slurm allocation")

    text_only = args.text_only or args.text_prompts is not None

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageDraw
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; verify the pinned wheel and assigned GPU driver")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("The assigned GPU does not support the requested BF16 baseline")
    dtype = getattr(torch, args.dtype)
    pin = args.revision or {MODEL: REVISION, TEXT_MODEL: TEXT_REVISION, GPT2_MODEL: GPT2_REVISION}.get(args.model)
    if args.snapshot and not re.fullmatch(r"[0-9a-f]{40}", pin or ""):
        parser.error("--snapshot requires an immutable --revision")
    revision = resolve_revision(args.model, pin)
    if args.snapshot:
        if not args.snapshot.is_dir() or args.snapshot.name != revision:
            parser.error("Snapshot directory must exist and be named by the pinned revision")
        snapshot = str(args.snapshot)
    else:
        snapshot = snapshot_download(args.model, revision=revision, max_workers=1,
                                     allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    config = AutoConfig.from_pretrained(snapshot)
    is_gpt2 = config.model_type == "gpt2"
    is_lm = config.model_type in {"qwen3", "gpt2"}
    if args.raw_text and (not is_lm or not text_only):
        parser.error("--raw-text requires a standalone LM with --text-prompts or --text-only")
    if is_lm and not text_only:
        parser.error("A standalone LM requires --text-only or --text-prompts")
    loader = AutoModelForCausalLM if is_lm else AutoModelForImageTextToText
    processor = AutoTokenizer.from_pretrained(snapshot) if is_lm else AutoProcessor.from_pretrained(snapshot, size={"shortest_edge": 3136, "longest_edge": 200704})
    tokenizer = processor if is_lm else processor.tokenizer
    template_options = {"enable_thinking": False} if config.model_type == "qwen3" and not args.raw_text else {}
    logit_kwargs = {"use_cache": False}
    if not is_lm:
        logit_kwargs["logits_to_keep"] = 1
    artifacts = args.output.with_suffix("")
    artifacts.mkdir(parents=True, exist_ok=True)
    fingerprint = lambda inputs: {k: hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k, v in inputs.items()}
    report = {"comparison_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "code_commit": code_revision(), "model": args.model, "revision": revision, "fa2_package": "flash-attn==2.8.3.post1", "dtype": args.dtype,
              "gpu": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda,
              "python": sys.version.split()[0], "packages": {p: version(p) for p in ("transformers", "torch", "torchvision", "huggingface-hub")},
              "prompt": None if args.text_prompts else args.prompt, "model_kind": "standalone_lm" if is_lm else "vlm", "chat_template_options": template_options, "input_format": "raw_text" if is_gpt2 or args.raw_text else "chat_template", "position": "last prompt token predicting the next token", "logit_projection": "full_prefix" if is_lm else "last_token", "cache": False, "seed": 0,
              "hidden_states": "All tokens at each vision/decoder block output, vision mergers, and final decoder norm; decoder block outputs precede DeepStack fusion",
              "generation": {"do_sample": False, "num_beams": 1, "use_cache": True, "max_new_tokens": args.max_new_tokens},
              "token_maps": "Arrays are [layer, token position]; decoder positions index the full prompt, vision positions follow native patch order",
              "examples": [], "results": [], "attention_config": config.to_dict()}
    try:
        report["packages"]["flash-attn"] = version("flash-attn")
    except PackageNotFoundError:
        report["packages"]["flash-attn"] = None
    reference_dir = args.reference.with_suffix("") if args.reference else artifacts
    reference_report = json.loads(args.reference.read_text()) if args.reference else None
    if reference_report:
        for field in ("model", "revision", "dtype", "prompt", "generation", "gpu", "cuda_runtime"):
            if report[field] != reference_report[field]:
                raise ValueError(f"Reference differs in {field}")
        for package in ("transformers", "torch", "torchvision"):
            if report["packages"][package] != reference_report["packages"][package]:
                raise ValueError(f"Reference differs in {package} version")
        if report["chat_template_options"] != reference_report.get("chat_template_options", {}):
            raise ValueError("Reference differs in chat template options")
        report["reference_run"] = {"path": str(args.reference), "sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest()}
    records = json.loads(args.coco.read_text())["images"] if args.coco else [{"image_id": "text" if text_only else "custom" if args.image else "synthetic"}]
    if args.coco:
        report["coco_manifest"] = json.loads(args.coco.read_text())
    if args.text_prompts:
        prompts = json.loads(args.text_prompts.read_text())
        if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            parser.error("--text-prompts must contain a nonempty JSON list of nonempty strings")
        records = [{"image_id": f"text-{i:02d}", "prompt": p} for i, p in enumerate(prompts)]
        report["text_manifest"] = {"path": str(args.text_prompts), "sha256": hashlib.sha256(args.text_prompts.read_bytes()).hexdigest(), "count": len(prompts)}
    cases = []
    for record in records:
        key = str(record["image_id"])
        if text_only:
            image = None
        elif args.coco:
            path = args.output.parent / "images" / f"{key}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            # COCO's own S3 bucket provides valid HTTPS for the original image URLs.
            url = record["url"].replace("http://images.cocodataset.org/", "https://s3.amazonaws.com/images.cocodataset.org/")
            if not path.exists():
                with urlopen(url, timeout=60) as response:
                    path.write_bytes(response.read())
            image = Image.open(path).convert("RGB")
        elif args.image:
            image = Image.open(args.image).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 50, 95, 145), fill="red")
            draw.ellipse((125, 50, 205, 130), fill="blue")
        prompt = record.get("prompt", args.prompt)
        if is_lm:
            inputs = encode_text(tokenizer, prompt, raw_text=is_gpt2 or args.raw_text, template_options=template_options)
        else:
            content = ([] if text_only else [{"type": "image", "image": image}]) + [{"type": "text", "text": prompt}]
            inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt", **template_options)
        inputs = {k: v.to(dtype=dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
        hashes = fingerprint(inputs)
        if reference_report:
            old = next(x for x in reference_report["examples"] if str(x["image_id"]) == key)
            if hashes != old["input_hashes"]:
                raise ValueError("Inputs differ from the saved eager reference")
        cases.append((key, inputs, hashes))
        torch.save(inputs, artifacts / f"{key}-inputs.pt")
        ids = inputs["input_ids"][0].tolist()
        image_positions = [] if text_only else [i for i, token in enumerate(ids) if token == tokenizer.convert_tokens_to_ids("<|image_pad|>")]
        image_span = {"start_marker": ids.index(tokenizer.convert_tokens_to_ids("<|vision_start|>")),
                      "end_marker": ids.index(tokenizer.convert_tokens_to_ids("<|vision_end|>")),
                      "image_start": min(image_positions), "image_end_exclusive": max(image_positions) + 1} if image_positions else None
        report["examples"].append({**record, "token_ids": ids, "tokens": tokenizer.convert_ids_to_tokens(ids), "image_span": image_span, "image_sha256": hashlib.sha256(image.tobytes()).hexdigest() if image is not None else None,
                                   "input_hashes": hashes, "input_shapes": {k: list(v.shape) for k, v in inputs.items()}})
    def save_report():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, implementation, kernel in attention_backends():
        if name not in args.backends:
            continue
        model, handles = None, []
        row = {"backend": name, "requested_implementation": implementation, "images": []}
        try:
            torch.manual_seed(0)
            model = loader.from_pretrained(snapshot, dtype=dtype, attn_implementation=implementation).to("cuda").eval()
            settings = {n: m.config._attn_implementation for n, m in model.named_modules() if type(m).__name__ in ("Qwen3VLVisionAttention", "Qwen3VLTextAttention", "Qwen3Attention", "GPT2Attention")}
            row["module_settings"] = settings
            row["attention_semantics"] = {n: {k: getattr(m, k, None) for k in
                ("is_causal", "sliding_window", "layer_type", "scaling", "scale_attn_weights", "scale_attn_by_inverse_layer_idx", "reorder_and_upcast_attn")}
                for n, m in model.named_modules() if n in settings}
            source_path = Path(sys.modules[type(model).__module__].__file__)
            row["model_source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if name == "fa2":
                import flash_attn_2_cuda
                with open(flash_attn_2_cuda.__file__, "rb") as binary:
                    row["fa2_binary_sha256"] = hashlib.file_digest(binary, "sha256").hexdigest()
                row["build_env"] = {k: os.environ.get(k) for k in ("CUDA_HOME", "FLASH_ATTN_CUDA_ARCHS", "MAX_JOBS", "NVCC_THREADS")}
            if not settings or any(setting != implementation for setting in settings.values()):
                raise ValueError("Attention module settings do not match the requested backend")
            layers = measurement_layers(model, text_only)
            if not layers:
                raise ValueError("No supported layers found for hidden-state collection")
            for key, cpu_inputs, hashes in cases:
                inputs = {k: v.to("cuda") for k, v in cpu_inputs.items()}
                states, handles = collect_layers(model, layers)
                samples = []
                with torch.inference_mode(), sdpa_kernel(kernel) if kernel is not None else nullcontext():
                    first_output = model(**inputs, **logit_kwargs)
                    logit_positions = first_output.logits.shape[1]
                    if is_lm and logit_positions != inputs["input_ids"].shape[1]:
                        raise ValueError("Standalone text models must project the full prefix")
                    samples.append(first_output.logits[0, -1].detach().cpu())
                    del first_output
                    for handle in handles:
                        handle.remove()
                    handles = []
                    for _ in range(2):
                        samples.append(model(**inputs, **logit_kwargs).logits[0, -1].detach().cpu())
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                        probe = model(**inputs, **logit_kwargs).logits[0, -1].detach().cpu()
                if fingerprint(inputs) != hashes:
                    raise ValueError("Processed inputs changed during the comparison")
                ops = sorted({e.key for e in profile.key_averages() if "attention" in e.key.lower() or "flash" in e.key.lower()})
                if name == "sdpa" and not any("aten::scaled_dot_product_attention" in op for op in ops):
                    raise ValueError("Missing profiler evidence for ordinary SDPA dispatch")
                if kernel is not None:
                    expected = "_scaled_dot_product_attention_math" if name == "sdpa_math" else "_scaled_dot_product_flash_attention"
                    if not any(expected in op for op in ops):
                        raise ValueError("Missing profiler evidence for the forced SDPA kernel")
                if name == "fa2" and not any("flashattn" in op.replace("_", "").lower() for op in ops):
                    raise ValueError("Missing profiler evidence for package FlashAttention")
                if set(states) != layers:
                    raise ValueError("Not all layer outputs were collected")
                arrays, axes = token_arrays(states, states) if name == "eager" else (None, None)
                with torch.inference_mode(), sdpa_kernel(kernel) if kernel is not None else nullcontext():
                    generated = model.generate(**inputs, do_sample=False, num_beams=1, num_return_sequences=1,
                                               max_new_tokens=args.max_new_tokens, use_cache=True, return_dict_in_generate=False)
                generated_ids = generated[0, inputs["input_ids"].shape[1]:].cpu().tolist()
                raw = {"logits": torch.stack(samples), "hidden_states": states, "generated_ids": generated_ids}
                reference = raw if name == "eager" else torch.load(reference_dir / f"{key}-eager.pt", weights_only=True)
                if set(reference["hidden_states"]) != set(states):
                    raise ValueError("Reference and candidate layer names differ")
                if name != "eager":
                    arrays, axes = token_arrays(reference["hidden_states"], states)
                map_path = artifacts / f"{key}-{name}-tokens.npz"
                np.savez_compressed(map_path, **arrays)
                ref_ids = reference["generated_ids"]
                divergence = next((i for i, (a, b) in enumerate(zip(ref_ids, generated_ids)) if a != b),
                                  min(len(ref_ids), len(generated_ids)) if len(ref_ids) != len(generated_ids) else None)
                result = {"image_id": key, "logit_positions": logit_positions, "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                          "generated_ids": generated_ids, "first_generated_divergence": divergence,
                          "token_maps": str(map_path), "token_axes": axes, **summarize(reference["logits"][0].float().tolist(), samples[0].float().tolist()),
                          "repeat_max_abs": max((s.float() - samples[0].float()).abs().max().item() for s in samples[1:]),
                          "profile_max_abs": (probe.float() - samples[0].float()).abs().max().item(), "attention_ops": ops,
                          "layers": {n: {"shape": list(states[n].shape), **hidden_difference(reference["hidden_states"][n], states[n])} for n in sorted(states)}}
                torch.save(raw, artifacts / f"{key}-{name}.pt")
                row["images"].append(result)
                print(json.dumps({"backend": name, **{k: v for k, v in result.items() if k not in ("layers", "attention_ops", "token_axes", "generated_ids", "generated_text")}, "layer_count": len(states)}), flush=True)
                if fingerprint(inputs) != hashes:
                    raise ValueError("Generation changed the processed inputs")
                del inputs, reference, raw, states, samples, probe, arrays, generated
            row["status"] = "ok"
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
            print(json.dumps({"backend": name, "error": row["error"]}), flush=True)
        finally:
            for handle in handles:
                handle.remove()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report["results"].append(row)
        save_report()
    return 0 if sum(r["status"] == "ok" for r in report["results"]) >= (1 if args.reference else 2) and (args.reference or report["results"][0]["status"] == "ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
