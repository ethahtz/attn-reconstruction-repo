"""Measure attention-output reconstruction errors for real and counterfactual V."""
import inspect

import torch
from nnsight import NNsight
from faithful_attention_extraction import extract_attention
from faithful_attention_extraction._core import _discover, _envoy

HELD_OUT_TRIALS = 3
VALIDATION_SEED = 20261013


def errors(predicted, actual):
    predicted, actual = predicted.double().cpu(), actual.double().cpu()
    delta = predicted - actual
    return {'max_abs': delta.abs().max().item(),
            'rmse': delta.square().mean().sqrt().item(),
            'relative_rmse': (delta.norm() / actual.norm().clamp_min(1e-30)).item()}



def forward_kwargs(model):
    """Use the model's supported uncached forward interface for every probe."""
    kwargs = {'use_cache': False, 'return_dict': True}
    if 'logits_to_keep' in inspect.signature(type(model).forward).parameters:
        kwargs['logits_to_keep'] = 1
    return kwargs



def validation_heads(specs, include_vision=False):
    """Fixed early/middle/final decoder layers, with three query heads per layer."""
    text = specs['text']
    layers = [0, (len(text)-1)//2, len(text)-1]
    heads = [('text', layer, head) for layer in layers
             for head in [0, text[layer].heads//2, text[layer].heads-1]]
    if include_vision:
        heads.append(('vision', 0, 0))
    return list(dict.fromkeys(heads))



def validate(model, inputs, heads, seed=VALIDATION_SEED, eager_patterns=None):
    """Compare recovered A with native weights, actual AV, and independent AV trials.

    Every random-value trace restores the genuine pre-O output. Thus downstream
    layers see identical inputs. Comparisons use float64 without renormalizing A.
    Private adapter discovery only identifies projection paths and head layouts.
    """
    specs = _discover(model)
    patterns, preservation = extract_attention(model, inputs, heads=heads)
    kwargs = {**inputs, **forward_kwargs(model)}
    report = {'extraction': preservation, 'restored_logit_max_abs': 0., 'heads': []}
    raw = {'patterns': patterns, 'layers': {}}
    wrapped = NNsight(model)
    native = None
    if model.config._attn_implementation == 'eager':
        with torch.no_grad():
            native = model(**kwargs, output_attentions=True).attentions
    grouped = {}
    for stream, layer, head in patterns:
        grouped.setdefault((stream, layer), []).append(head)
    with torch.no_grad():
        original_logits = model(**kwargs).logits.detach().clone()
        for (stream, layer), chosen in grouped.items():
            spec = specs[stream][layer]
            attn = _envoy(wrapped, spec.path)
            value, output = getattr(attn, spec.value), getattr(attn, spec.output)
            with wrapped.trace(**kwargs):
                if spec.fused:
                    projection = value.output.detach().clone().save()
                else:
                    q = attn.q_proj.output.detach().clone().save()
                    k = attn.k_proj.output.detach().clone().save()
                    projection = value.output.detach().clone().save()
                z = output.input.detach().clone().save()
                logits = wrapped.output.logits.detach().clone().save()
            qk = (projection[..., :2*spec.heads*spec.dim] if spec.fused
                  else torch.cat([q, k], dim=-1))
            actual_v = projection[..., -spec.kv_heads*spec.dim:]
            packed = z.ndim == 2
            B, T = (1, z.shape[0]) if packed else z.shape[:2]

            def value_heads(v):
                return v.reshape(B, T, spec.kv_heads, spec.dim).transpose(1, 2).repeat_interleave(
                    spec.heads // spec.kv_heads, dim=1).cpu()

            def output_heads(activation):
                return activation.reshape(B, T, spec.heads, spec.dim).transpose(1, 2).cpu()

            report['restored_logit_max_abs'] = max(report['restored_logit_max_abs'],
                (logits-original_logits).abs().max().item())
            v_heads, z_heads = value_heads(actual_v), output_heads(z)
            rows = []
            for head in chosen:
                matrix = patterns[stream, layer, head]
                row = {'stream': stream, 'layer': layer, 'head': head,
                       'kv_head': head // (spec.heads // spec.kv_heads),
                       'tokens': T, 'head_dim': spec.dim,
                       'original_v': errors(matrix.double() @ v_heads[:, head].double(), z_heads[:, head]),
                       'held_out': []}
                if native is not None and stream == 'text':
                    row['native_eager'] = errors(matrix, native[layer][:, head])
                eager_matrix = matrix if eager_patterns is None else eager_patterns[stream, layer, head]
                row['extracted_eager_pattern'] = errors(matrix, eager_matrix)
                row['original_v_eager'] = errors(eager_matrix.double() @ v_heads[:, head].double(), z_heads[:, head])
                rows.append(row)
            saved_trials = []
            # Values are generated on CPU so seeds are comparable across kernels/devices.
            for trial in range(HELD_OUT_TRIALS):
                trial_seed = seed + 1000*layer + trial + (100000 if stream == 'vision' else 0)
                generator = torch.Generator().manual_seed(trial_seed)
                fake_v = torch.randn(actual_v.shape, generator=generator).to(actual_v)
                with wrapped.trace(**kwargs):
                    if spec.fused:
                        trial_qk = value.output[..., :2*spec.heads*spec.dim].detach().clone().save()
                        value.output = torch.cat([value.output[..., :2*spec.heads*spec.dim], fake_v], dim=-1)
                    else:
                        trial_q = attn.q_proj.output.detach().clone().save()
                        trial_k = attn.k_proj.output.detach().clone().save()
                        value.output = fake_v
                    trial_z = output.input.detach().clone().save()
                    output.input = z
                    trial_logits = wrapped.output.logits.detach().clone().save()
                if not spec.fused:
                    trial_qk = torch.cat([trial_q, trial_k], dim=-1)
                qk_delta = (trial_qk-qk).abs().max().item()
                report['restored_logit_max_abs'] = max(report['restored_logit_max_abs'],
                    (trial_logits-original_logits).abs().max().item())
                trial_v_heads, trial_z_heads = value_heads(fake_v), output_heads(trial_z)
                for row in rows:
                    head = row['head']
                    matrix = patterns[stream, layer, head].double()
                    v, observed = trial_v_heads[:, head].double(), trial_z_heads[:, head]
                    stats = errors(matrix @ v, observed)
                    eager_matrix = matrix if eager_patterns is None else eager_patterns[stream, layer, head].double()
                    stats.update(seed=trial_seed, qk_max_abs=qk_delta,
                        eager_relative_rmse=errors(eager_matrix @ v, observed)['relative_rmse'],
                        scrambled_relative_rmse=errors(matrix.flip(-1) @ v, observed)['relative_rmse'])
                    row['held_out'].append(stats)
                saved_trials.append({'seed': trial_seed, 'v': fake_v.cpu(), 'z': trial_z.cpu()})
            raw['layers'][stream, layer] = {'path': spec.path, 'v': actual_v.cpu(),
                                          'z': z.cpu(), 'trials': saved_trials}
            report['heads'].extend(rows)
    assert report['restored_logit_max_abs'] == 0, report['restored_logit_max_abs']
    assert all(preservation[k] == 0 for k in
               ['baseline_logit_max_abs', 'probe_logit_max_abs', 'unpatched_logit_max_abs'])
    assert all(t['qk_max_abs'] == 0 for row in report['heads'] for t in row['held_out'])
    return report, raw
