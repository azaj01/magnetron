# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
import json
import os
import glob
import gc

from magnetron import Snapshot, Tensor, dtype, context
from model import Qwen3Model, Qwen3HyperParams
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import torch

def _mag_to_torch_dtype(mag_dtype: dtype.DType) -> torch.dtype:
    return {
        dtype.float16: torch.float16,
        dtype.bfloat16: torch.bfloat16,
        dtype.float32: torch.float32,
        dtype.float8_e4m3fn: torch.float8_e4m3fn
    }[mag_dtype]


def _mag_dtype_from_str(dtype_str: str) -> dtype.DType:
    return {
        'float16': dtype.float16,
        'bfloat16': dtype.bfloat16,
        'float32': dtype.float32,
        'mixed_fp8': dtype.float8_e4m3fn
    }[dtype_str]


def _iter_safetensor_shards(repo_dir: str) -> list[str]:
    index_path = os.path.join(repo_dir, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path, encoding='utf-8') as f:
            index = json.load(f)
        shards = sorted(set(index['weight_map'].values()))
        return [os.path.join(repo_dir, s) for s in shards]
    shards = sorted(glob.glob(os.path.join(repo_dir, 'model-*.safetensors')))
    if shards:
        return shards
    single = os.path.join(repo_dir, 'model.safetensors')
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError('No safetensors weights found in repo snapshot.')

def _quantize(x: Tensor) -> tuple[Tensor, Tensor]:
    qtype = dtype.float8_e4m3fn
    highp = x.cast(dtype.float32)
    amax = float(highp.abs().max().item())
    fp8max = qtype.max
    scale = 1.0 if amax < 1e-12 or not amax==amax else amax/fp8max
    inv_scale = 1.0/scale if scale != 0.0 else 1.0
    q = (highp * inv_scale)
    q = Tensor.where(q > fp8max, fp8max,  Tensor.where(q < -fp8max, -fp8max, q))
    q = q.cast(qtype)
    return q, Tensor([scale], dtype=dtype.float32)

def _convert_model(repo: str, torch_dtype: torch.dtype, mag_dtype: dtype.DType) -> None:
    skip: set[str] = {'cos_cache', 'sin_cache'}

    print(f'Downloading model {repo} from Hugging Face...')

    repo_dir = snapshot_download(repo_id=repo)
    cfg = Qwen3HyperParams()
    fp8w_mode = (mag_dtype == dtype.float8_e4m3fn)
    if fp8w_mode:
        context.set_default_dtype(dtype.bfloat16)
    else:
        context.set_default_dtype(mag_dtype)

    mag_model = Qwen3Model(cfg)
    if not fp8w_mode:
        mag_model = mag_model.cast(mag_dtype)
    sd_mag: dict[str, Tensor] = mag_model.state_dict()
    remaining = dict(sd_mag)

    for k in list(remaining.keys()):
        if k in skip:
            remaining.pop(k)

    def hf_key_for(mag_key: str) -> str:
        if mag_key == 'lm_head.weight' and getattr(cfg, 'tie_word_embeddings', False):
            return 'model.embed_tokens.weight'
        if mag_key.startswith('lm_head.'):
            return mag_key
        return 'model.' + mag_key

    scale_keys: set[str] = {k for k in remaining if k.endswith('.weight_scale')}
    snap_file: str = f'{repo.split("/")[1].lower()}-{mag_dtype.short_name}.mag'
    print(f'Writing snapshot to {snap_file}...')
    with Snapshot.write(snap_file) as snap:
        for shard_path in _iter_safetensor_shards(repo_dir):
            sd_hf: dict[str, Tensor] = load_file(shard_path, device='cpu')
            processed_stack: list[str] = []
            for key, tensor in remaining.items():
                if key in scale_keys:
                    continue
                hf_key: str = hf_key_for(key)
                torch_tensor: torch.Tensor = sd_hf.get(hf_key)
                if torch_tensor is None:
                    continue
                scaled_quant: bool = key.endswith('.weight') and (f'{key}_scale' in scale_keys)
                if scaled_quant:
                    mag_tensor: Tensor = Tensor(torch_tensor.to(torch_dtype).to('cpu').contiguous(), dtype=mag_dtype)
                    quantized, scale = _quantize(mag_tensor)
                    print(f'Quantized {hf_key} -> {key}, Shape={tuple(torch_tensor.shape)}, Scale={scale.item()}')
                    snap.put_tensor(key, quantized)
                    scale_key: str = f'{key}_scale'
                    snap.put_tensor(scale_key, scale)
                    processed_stack.append(scale_key)
                else:
                    mag_tensor: Tensor = Tensor(torch_tensor.to(torch_dtype).to('cpu').contiguous(), dtype=mag_dtype)
                    print(f'Converted {hf_key} -> {key}, Shape={tuple(torch_tensor.shape)}')
                    snap.put_tensor(key, mag_tensor)
                processed_stack.append(key)
            for k in processed_stack:
                remaining.pop(k)
            del sd_hf
            gc.collect()
            if not remaining:
                break
        if remaining:
            for key, tensor in remaining.items():
                if key.endswith('.bias'):
                    print(f'Missing HF bias for {key}; writing zeros')
                    tensor.zeros_()
                    snap.put_tensor(key, tensor)
                elif key.endswith('.weight_scale'):
                    raise KeyError(f'Orphan weight_scale entry: {key}')
                else:
                    raise KeyError(f'Missing HF weight for magnetron key: {key}')

        snap.print_info()

    print(f'Converted model saved to {snap_file}')


def _main() -> None:
    args = argparse.ArgumentParser(description='Convert Hugging Face Qwen model to Magnetron file format')
    args.add_argument('--model', type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='HF repo model name')
    args.add_argument('--dtype', type=str, default='mixed_fp8', choices=['mixed_fp8', 'float16', 'bfloat16', 'float32'], help='Data type for Magnetron tensors')
    args = args.parse_args()
    mag_dtype = _mag_dtype_from_str(args.dtype)
    _convert_model(args.model, torch_dtype=_mag_to_torch_dtype(mag_dtype), mag_dtype=mag_dtype)


if __name__ == '__main__':
    _main()
