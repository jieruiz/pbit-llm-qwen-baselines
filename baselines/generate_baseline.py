import argparse

import torch

from common import device_metadata, load_local_model, save_json, set_deterministic


DEFAULT_PROMPTS = [
    "人工智能的发展将会",
    "请用三句话解释什么是概率计算。\n",
    "The main advantage of probabilistic computing is",
    "Question: What is 17 plus 25?\nAnswer:",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    set_deterministic(args.seed)
    tokenizer, model = load_local_model(args.model, args.device)
    records = []
    for prompt in DEFAULT_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation_ids = output_ids[0, inputs.input_ids.shape[-1] :]
        records.append(
            {
                "prompt": prompt,
                "continuation": tokenizer.decode(
                    continuation_ids, skip_special_tokens=True
                ),
                "input_tokens": inputs.input_ids.shape[-1],
                "new_tokens": continuation_ids.numel(),
            }
        )

    result = {
        "test": "deterministic_greedy_generation",
        "model_path": args.model,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "environment": device_metadata(args.device),
        "records": records,
    }
    save_json(result, args.output)
    for record in records:
        print(f"PROMPT: {record['prompt']}\nOUTPUT: {record['continuation']}\n")


if __name__ == "__main__":
    main()

