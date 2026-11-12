"""Local attention recovery through NNsight partial-identity value probes."""
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from nnsight import NNsight

Head = tuple[str, int, int]


@dataclass(frozen=True)
class _Spec:
    path: str
    value: str
    output: str
    heads: int
    kv_heads: int
    dim: int
    fused: bool = False


def _spec(path, module):
    if getattr(module, 'is_cross_attention', False) or any(
        part in path.split('.') for part in ('cross_attn', 'encoder_attn', 'crossattention')
    ):
        return None
    dim = getattr(module, 'head_dim', None)
    if not isinstance(dim, int) or dim <= 0:
        return None
    for output in ('o_proj', 'out_proj'):
        v = getattr(module, 'v_proj', None)
        o = getattr(module, output, None)
        if all(hasattr(module, name) for name in ('q_proj', 'k_proj')) and v is not None and o is not None:
            n, kv = getattr(o, 'in_features', 0), getattr(v, 'out_features', 0)
            if n > 0 and kv > 0 and n % dim == kv % dim == 0:
                return _Spec(path, 'v_proj', output, n // dim, kv // dim, dim)
    # Fused QKV layouts are explicit: Q, then K, then V, with equal head counts.
    if type(module).__name__ == 'GPT2Attention' and module.c_attn.nf == 3 * module.num_heads * dim:
        return _Spec(path, 'c_attn', 'c_proj', module.num_heads, module.num_heads, dim, True)
    if type(module).__name__ == 'Qwen3VLVisionAttention':
        return _Spec(path, 'qkv', 'proj', module.num_heads, module.num_heads, dim, True)
    return None


def _discover(model, attention_modules=None):
    groups = {}
    if attention_modules is not None:
        for stream, paths in attention_modules.items():
            groups[stream] = []
            for path in paths:
                try:
                    spec = _spec(path, model.get_submodule(path))
                except AttributeError:
                    spec = None
                if spec is None:
                    raise ValueError(f'No supported attention projection layout at {path!r}')
                groups[stream].append(spec)
    else:
        for path, module in model.named_modules():
            spec = _spec(path, module)
            if spec is not None:
                stream = 'vision' if any(part == 'visual' or part.startswith('vision')
                                         for part in path.split('.')) else 'text'
                groups.setdefault(stream, []).append(spec)
    if not any(groups.values()):
        raise ValueError('No supported attention projection layout found; see package compatibility')
    return groups


def _validate_inputs(model, inputs):
    if model.training or any(module.training for module in model.modules()):
        raise ValueError('Use evaluation mode: call model.eval()')
    if inputs.get('past_key_values') is not None:
        raise ValueError('Use an uncached full prefix; past_key_values is unsupported')
    config = getattr(model, 'config', None)
    configs = [config, getattr(config, 'text_config', None), getattr(config, 'vision_config', None)]
    if inputs.get('output_attentions') or any(getattr(c, 'output_attentions', False) for c in configs):
        raise ValueError('output_attentions=True can switch attention backends; disable it')
    ids, embeds = inputs.get('input_ids'), inputs.get('inputs_embeds')
    if not ((isinstance(ids, torch.Tensor) and ids.ndim == 2 and min(ids.shape) > 0) or
            (isinstance(embeds, torch.Tensor) and embeds.ndim == 3 and min(embeds.shape) > 0)):
        raise ValueError('Use nonempty input_ids [batch, tokens] or inputs_embeds [batch, tokens, dim]')
    if hasattr(model, '_orig_mod'):
        raise ValueError('Compiled models are outside the supported scope')


def _envoy(wrapped, path):
    node = wrapped
    for part in path.split('.'):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def extract_attention(
    model: torch.nn.Module,
    inputs: Mapping[str, Any],
    heads: Sequence[tuple[int, int] | Head] | None = None,
    *,
    attention_modules: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[Head, torch.Tensor], dict[str, Any]]:
    """Recover requested heads; return CPU [B,T,T] tensors and preservation checks.

    (L,H) addresses text heads; (stream,L,H) also addresses vision/custom streams.
    None selects every text head. Ordinary HF evaluation models only, uncached.
    Partial-identity V probes preserve Q/K, masks, backend and downstream values.
    Vision's packed tokens have a singleton batch axis. Reconstruction is finite
    precision and quadratic in token count; head selection limits saved memory.
    """
    _validate_inputs(model, inputs)
    specs = _discover(model, attention_modules)
    if heads is None:
        heads = [('text', layer, head) for layer, spec in enumerate(specs.get('text', []))
                 for head in range(spec.heads)]
    selected, grouped = [], {}
    for address in heads:
        if not isinstance(address, (tuple, list)) or len(address) not in (2, 3):
            raise ValueError('heads must contain (layer, head) or (stream, layer, head) addresses')
        stream, layer, head = ('text', *address) if len(address) == 2 else address
        if (not isinstance(stream, str) or type(layer) is not int or type(head) is not int or
            stream not in specs or not 0 <= layer < len(specs[stream]) or
            not 0 <= head < specs[stream][layer].heads):
            raise ValueError(f'Invalid heads address: {address!r}')
        key = (stream, layer, head)
        if key in selected:
            raise ValueError(f'Duplicate heads address: {key!r}')
        selected.append(key)
        grouped.setdefault((stream, layer), []).append(head)
    if not selected:
        raise ValueError('heads selection is empty or no text attention was discovered')
    paths = [specs[stream][layer].path for stream, layer in grouped]
    if len(set(paths)) != len(paths):
        raise ValueError('Do not address the same module through multiple stream aliases')
    kwargs = {**inputs, 'use_cache': False, 'return_dict': True}
    if 'logits_to_keep' in inspect.signature(type(model).forward).parameters:
        kwargs['logits_to_keep'] = 1
    checks = {'baseline_logit_max_abs': 0.0, 'probe_logit_max_abs': 0.0,
              'unpatched_logit_max_abs': 0.0, 'heads': selected, 'probes': 0,
              'modules': {stream: [{'path': s.path, 'heads': s.heads, 'head_dim': s.dim,
                                   'packed': None} for s in group]
                          for stream, group in specs.items()}}
    patterns = {}
    with torch.no_grad():
        original = model(**kwargs).logits.detach().clone()
        wrapped = NNsight(model)
        for (stream, layer), chosen in grouped.items():
            spec = specs[stream][layer]
            attn = _envoy(wrapped, spec.path)
            value, output = getattr(attn, spec.value), getattr(attn, spec.output)
            # Save the genuine result first, for restoration before the output projection.
            with wrapped.trace(**kwargs):
                saved_z = output.input.detach().clone().save()
                saved_logits = wrapped.output.logits.detach().clone().save()
            baseline_z = saved_z
            packed = baseline_z.ndim == 2
            if baseline_z.ndim not in (2, 3) or baseline_z.shape[-1] != spec.heads * spec.dim:
                raise ValueError(f'Unsupported merged-head output layout at {spec.path}')
            batch, length = (1, baseline_z.shape[0]) if packed else baseline_z.shape[:2]
            checks['modules'][stream][layer].update(packed=packed, tokens=length, batch=batch)
            checks['baseline_logit_max_abs'] = max(checks['baseline_logit_max_abs'],
                                                   (saved_logits-original).abs().max().item())
            pieces = []
            for offset in range(0, length, spec.dim):
                width = min(spec.dim, length-offset)
                basis = torch.zeros(length, spec.dim, device=baseline_z.device, dtype=baseline_z.dtype)
                basis[offset:offset+width, :width] = torch.eye(width, device=basis.device, dtype=basis.dtype)
                fake_v = basis[:, None, :].expand(length, spec.kv_heads, spec.dim).reshape(length, -1)
                if not packed:
                    fake_v = fake_v[None].expand(batch, -1, -1).contiguous()
                with wrapped.trace(**kwargs):
                    if spec.fused:
                        # Preserve the original Q and K portions of a fused projection.
                        value.output = torch.cat([value.output[..., :2*spec.heads*spec.dim], fake_v], dim=-1)
                    else:
                        value.output = fake_v
                    # Slice heads before saving, so unrequested patterns are never transferred.
                    block = output.input.reshape(batch, length, spec.heads, spec.dim).permute(
                        0, 2, 1, 3)[:, chosen, :, :width].detach().clone().save()
                    output.input = baseline_z
                    probe_logits = wrapped.output.logits.detach().clone().save()
                pieces.append(block.cpu())
                checks['probe_logit_max_abs'] = max(checks['probe_logit_max_abs'],
                                                   (probe_logits-original).abs().max().item())
                checks['probes'] += 1
            weights = torch.cat(pieces, dim=-1)
            for index, head in enumerate(chosen):
                patterns[(stream, layer, head)] = weights[:, index]
        untouched = model(**kwargs).logits
        checks['unpatched_logit_max_abs'] = (untouched-original).abs().max().item()
    return {key: patterns[key] for key in selected}, checks


def reconstruct_attention(model, inputs, layers=None, *, heads=None, attention_modules=None):
    """Research compatibility adapter: return stacked text layers and checks."""
    _validate_inputs(model, inputs)
    if attention_modules is not None and layers is not None:
        raise ValueError('Choose layers or attention_modules, not both')
    explicit = {'text': list(attention_modules)} if attention_modules is not None else None
    specs = _discover(model, explicit).get('text', [])
    layers = list(range(len(specs))) if layers is None else list(layers)
    if not layers or any(type(i) is not int or not 0 <= i < len(specs) for i in layers) or len(set(layers)) != len(layers):
        raise ValueError('layers must contain distinct integer text indices in range')
    if heads is None:
        chosen = list(range(specs[layers[0]].heads))
    else:
        chosen = [heads] if type(heads) is int else list(heads)
    if not chosen or any(type(h) is not int or not 0 <= h < specs[i].heads for i in layers for h in chosen) or len(set(chosen)) != len(chosen):
        raise ValueError('heads must contain distinct integer head indices in range')
    patterns, checks = extract_attention(model, inputs,
        heads=[(i, h) for i in layers for h in chosen], attention_modules=explicit)
    weights = torch.stack([torch.stack([patterns[('text', i, h)] for h in chosen], dim=1)
                           for i in layers], dim=1)
    checks.update(layers=layers, heads=chosen, modules=[specs[i].path for i in layers])
    return weights, checks
