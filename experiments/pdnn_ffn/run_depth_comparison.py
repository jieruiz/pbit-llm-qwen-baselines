"""Matched-parameter full-path P-DNN depth experiment, inside a GPU allocation."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SHAPES = {2: (4864,), 3: (2192, 2192), 4: (1688, 1688, 1688)}


def parameter_count(hidden):
    dims = (896, *hidden, 896)
    return sum(a * b + b for a, b in zip(dims, dims[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--train-text', required=True)
    parser.add_argument('--test-text', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Refusing to overwrite a nonempty result directory')
    root.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    plan = {
        'scope': 'One decoder FFN (zero-based layer 12); matrix depth varies, parameter count matched within 0.2%.',
        'shapes': {str(d): [896, *h, 896] for d, h in SHAPES.items()},
        'parameters': {str(d): parameter_count(h) for d, h in SHAPES.items()},
        'coding': 'binary', 'input_temperature': 0.125, 'hidden_temperature': 0.5,
        'training_seeds': [0, 1, 2], 'mean_steps': 2000, 'sample_steps': 2000,
        'training_paths': 4, 'primary_evaluation': '4 paths, inference seed 0, three independently trained checkpoints per depth',
        'secondary_evaluation': 'training seed 0: paths 0,1,8,16 at inference seed 0, and paths 4 at inference seeds 1,2',
        'job_id': os.environ.get('SLURM_JOB_ID'),
    }
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    events = root / 'events.jsonl'

    def run(command, label):
        print('START', label, flush=True)
        start = time.perf_counter()
        subprocess.run([sys.executable, *map(str, command)], check=True)
        elapsed = time.perf_counter() - start
        with events.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'label': label, 'seconds': elapsed, 'complete': True}) + '\n')
        print('DONE', label, elapsed, flush=True)

    common_train = [script_dir/'train_full_path_distillation.py', '--model', args.model,
                    '--train-text', args.train_text, '--layer', '12', '--coding', 'binary',
                    '--input-temperature', '0.125', '--hidden-temperature', '0.5',
                    '--mean-steps', '2000', '--sample-steps', '2000', '--train-samples', '4',
                    '--sequence-length', '256', '--batch-size', '4', '--validate-every', '200',
                    '--learning-rate', '3e-4', '--sample-learning-rate', '1e-4']

    def evaluate(directory, count, seed):
        run([script_dir/'evaluate_full_path_perplexity.py', '--model', args.model,
             '--checkpoint', directory/'student_sampled.pt', '--text-file', args.test_text,
             '--output', directory/f'ppl_paths{count}_infer{seed}.json', '--sample-count', count,
             '--seed', seed, '--max-length', '2048', '--stride', '1024'],
            f'{directory.name}/{directory.parent.name}/eval_N{count}_seed{seed}')

    # Every full-size depth gets an allocated-GPU smoke run before the sweep.
    for depth, hidden in SHAPES.items():
        directory = root / 'smoke' / f'depth{depth}'
        run([*common_train, '--hidden-sizes', ','.join(map(str, hidden)), '--output-dir', directory,
             '--mean-steps', '2', '--sample-steps', '2', '--sequence-length', '32',
             '--batch-size', '1', '--validation-batches', '1'], f'smoke_depth{depth}')
        run([script_dir/'evaluate_full_path_perplexity.py', '--model', args.model,
             '--checkpoint', directory/'student_sampled.pt', '--text-file', args.test_text,
             '--output', directory/'ppl_smoke.json', '--sample-count', '4',
             '--max-tokens', '128', '--max-length', '128', '--stride', '64'], f'smoke_eval_depth{depth}')

    for seed in (0, 1, 2):
        for depth, hidden in SHAPES.items():
            directory = root / f'depth{depth}' / f'train_seed{seed}'
            run([*common_train, '--hidden-sizes', ','.join(map(str, hidden)), '--output-dir', directory,
                 '--seed', seed], f'train_depth{depth}_seed{seed}')
            evaluate(directory, 4, 0)
    for depth in SHAPES:
        directory = root / f'depth{depth}' / 'train_seed0'
        for count, seed in [(0, 0), (1, 0), (8, 0), (16, 0), (4, 1), (4, 2)]:
            evaluate(directory, count, seed)
    (root / 'RUN_COMPLETE').touch()


if __name__ == '__main__':
    main()
