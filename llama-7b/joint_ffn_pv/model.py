import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from attention import install_attention
from fixed_ffn import FixedFFN
from streaming_ffn import StreamConfig


ROOT = Path(__file__).resolve().parent


def protected_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".mlp." in name:
            continue
        digest.update(name.encode())
        digest.update(
            parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        )
    return digest.hexdigest()


def load(model_path, ffn_mode, attention_mode, seed, samples=512, chunk=32):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).cuda().eval()
    model.requires_grad_(False)
    original_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    before = protected_hash(model)

    ffns = {}
    if ffn_mode == "fixed":
        calibration = json.loads((ROOT / "calibration_ffn.json").read_text())
        for index, layer in enumerate(model.model.layers):
            replacement = FixedFFN(
                layer.mlp,
                StreamConfig(
                    cycles=4096,
                    interval=256,
                    burn_fraction=0.25,
                    weight_bits=8,
                    layer=index,
                    seed=seed,
                ),
                calibration["scales"][str(index)],
                bits=12,
                fixed_mode="streaming",
                diagnostics=True,
            )
            layer.mlp = replacement
            ffns[index] = replacement
    elif ffn_mode != "original":
        raise ValueError(ffn_mode)

    controls = install_attention(
        model, attention_mode, seed=seed, samples=samples, chunk=chunk
    )
    if before != protected_hash(model):
        raise RuntimeError("Protected model parameters changed during installation")
    return model, ffns, controls, before, original_parameter_count
