import argparse

import torch

from common import device_metadata, load_local_model, save_json, set_deterministic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_deterministic(args.seed)
    tokenizer, model = load_local_model(args.model, args.device)
    prompt = "Qwen is a language model developed by"
    encoded = tokenizer(prompt, return_tensors="pt").to(args.device)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        logits = model(**encoded, use_cache=True).logits

    total_parameters = sum(p.numel() for p in model.parameters())
    embedding_parameters = model.get_input_embeddings().weight.numel()
    output_weight = model.get_output_embeddings().weight
    tied_embeddings = (
        model.get_input_embeddings().weight.data_ptr() == output_weight.data_ptr()
    )
    result = {
        "test": "model_integrity",
        "status": "pass" if torch.isfinite(logits).all().item() else "fail",
        "model_path": args.model,
        "model_type": model.config.model_type,
        "total_parameters": total_parameters,
        "embedding_parameters": embedding_parameters,
        "non_embedding_parameters": total_parameters - embedding_parameters,
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "vocab_size": model.config.vocab_size,
        "tie_word_embeddings": tied_embeddings,
        "input_tokens": encoded.input_ids.shape[-1],
        "logits_shape": list(logits.shape),
        "logits_all_finite": torch.isfinite(logits).all().item(),
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
        "environment": device_metadata(args.device),
    }
    save_json(result, args.output)
    print(result)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

