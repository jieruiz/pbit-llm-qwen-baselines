import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_local_model(model_path: str, device: str = "cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return tokenizer, model


def device_metadata(device: str) -> dict:
    result = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
    if device == "cuda":
        result.update(
            {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    return result


def save_json(payload: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

