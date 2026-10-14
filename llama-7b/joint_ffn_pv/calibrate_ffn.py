"""Calibrate immutable power-of-two FFN activation ranges on WT2 train."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def power_of_two_bound(value):
    return torch.pow(2.0, torch.ceil(torch.log2(value.clamp_min(2.0**-24))))


class Collector:
    def __init__(self, layers, group=64):
        self.group = group
        self.maxima = {
            str(index): {
                "x": torch.zeros(4096 // group, device="cuda"),
                "g": torch.zeros(11008 // group, device="cuda"),
                "u": torch.zeros(11008 // group, device="cuda"),
            }
            for index in range(len(layers))
        }
        self.handles = []
        for index, layer in enumerate(layers):
            self.handles.append(
                layer.mlp.register_forward_pre_hook(self._pre_hook(str(index)))
            )
            self.handles.append(
                layer.mlp.gate_proj.register_forward_hook(self._hook(str(index), "g"))
            )
            self.handles.append(
                layer.mlp.up_proj.register_forward_hook(self._hook(str(index), "u"))
            )

    def update(self, layer, signal, value):
        width = value.shape[-1]
        grouped = value.detach().float().reshape(-1, width // self.group, self.group)
        observed = grouped.abs().amax(dim=(0, 2))
        self.maxima[layer][signal] = torch.maximum(
            self.maxima[layer][signal], observed
        )

    def _pre_hook(self, layer):
        def hook(module, inputs):
            del module
            self.update(layer, "x", inputs[0])

        return hook

    def _hook(self, layer, signal):
        def hook(module, inputs, output):
            del module, inputs
            self.update(layer, signal, output)

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    def scales(self):
        return {
            layer: {
                signal: power_of_two_bound(value).cpu().tolist()
                for signal, value in signals.items()
            }
            for layer, signals in self.maxima.items()
        }

    def observed(self):
        return {
            layer: {signal: value.cpu().tolist() for signal, value in signals.items()}
            for layer, signals in self.maxima.items()
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--context", type=int, default=2048)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    all_ids = tokenizer(
        args.train.read_text(encoding="utf-8"),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    ids = all_ids[:, : args.tokens].contiguous()
    if ids.shape[1] != args.tokens:
        raise RuntimeError(f"Requested {args.tokens} tokens, found {ids.shape[1]}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).cuda().eval()
    model.requires_grad_(False)
    collector = Collector(model.model.layers)
    with torch.inference_mode():
        for begin in range(0, args.tokens, args.context):
            end = min(begin + args.context, args.tokens)
            model.model(ids[:, begin:end].cuda(), use_cache=False, return_dict=True)
            print(json.dumps({"calibrated_tokens": end}), flush=True)
    collector.close()

    payload = {
        "source": "first 8192 tokens of WT2 train; no weight updates",
        "source_tokens": int(all_ids.shape[1]),
        "calibration_tokens": int(ids.shape[1]),
        "context": args.context,
        "ids_sha256": tensor_sha256(ids),
        "group": 64,
        "policy": "per-layer per-group observed max rounded up to power of two",
        "scales": collector.scales(),
        "observed_maxima": collector.observed(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
