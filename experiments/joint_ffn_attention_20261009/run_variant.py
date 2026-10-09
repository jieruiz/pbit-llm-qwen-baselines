"""Run one published preset without modifying archived results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant', choices=['pv_only', 'qk_ising', 'qk_softmax_pv'])
    p.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    settings = json.loads((ROOT / 'configs' / (a.variant + '.json')).read_text())
    output = a.output.resolve() if a.output else ROOT / 'outputs' / f'{a.variant}_s{a.seed}.json'
    if output.is_relative_to((ROOT / 'results').resolve()):
        p.error('Use outputs/ or another directory; results/ is the published archive.')
    command = [sys.executable, str(ROOT / 'evaluate.py')]
    for key, value in settings.items():
        command.extend(['--' + key, str(value)])
    command.extend(['--seed', str(a.seed), '--output', str(output)])
    if a.dry_run:
        print(json.dumps({'variant': a.variant, 'gpu': a.gpu, 'command': command}, indent=2))
        return
    if output.exists():
        p.error(f'Output already exists: {output}')
    required = [ROOT / 'models/Qwen2.5-0.5B/config.json', ROOT / 'data/wikitext-2/wiki.test.raw']
    for path in required:
        if not path.is_file():
            p.error(f'Missing asset: {path}; see README.md')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    subprocess.run(command, cwd=ROOT, env=env, check=True)

if __name__ == '__main__':
    main()
