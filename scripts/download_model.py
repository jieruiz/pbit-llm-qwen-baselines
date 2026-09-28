import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", default="models/Qwen2.5-0.5B")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output,
    )
    print(f"Model snapshot written to {output}")


if __name__ == "__main__":
    main()

