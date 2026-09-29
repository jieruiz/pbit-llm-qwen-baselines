"""Validate all paired evaluations and summarize the coding comparison."""
import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    table = {}
    reference_args = None
    for name in ('bipolar', 'binary_same_temperature', 'binary_half_temperature'):
        directory = args.results / name
        values = {}
        for count, seed in [(n, 0) for n in (0, 1, 4, 8, 16)] + [(4, 1), (4, 2)]:
            path = directory / f'student_sampled_samples{count}_seed{seed}_ppl.json'
            result = json.loads(path.read_text(encoding='utf-8'))
            for key in ('corpus_tokens', 'scored_tokens', 'max_length', 'stride'):
                if result[key] != baseline[key]:
                    raise ValueError(f'{path}: evaluation protocol mismatch: {key}')
            if result['sample_count'] != count or result['seed'] != seed or result['layer'] != 12:
                raise ValueError(f'{path}: condition mismatch')
            expected_coding = 'bipolar' if name == 'bipolar' else 'binary'
            if result['student_config']['coding'] != expected_coding:
                raise ValueError(f'{path}: wrong coding')
            expected_temperatures = (0.125, 0.5) if name == 'binary_half_temperature' else (0.25, 1.0)
            actual_temperatures = tuple(result['student_config'][key] for key in ('input_temperature', 'hidden_temperature'))
            if actual_temperatures != expected_temperatures:
                raise ValueError(f'{path}: wrong temperatures')
            if not math.isfinite(result['perplexity']) or result['perplexity'] <= 0:
                raise ValueError(f'{path}: invalid perplexity')
            values[f'N{count}_seed{seed}'] = result['perplexity']
        seeds = [values[f'N4_seed{seed}'] for seed in (0, 1, 2)]
        training = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        header = json.loads((directory / 'train_metrics.jsonl').read_text(encoding='utf-8').splitlines()[0])
        comparable = {key: value for key, value in header['args'].items()
                      if key not in ('coding', 'input_temperature', 'hidden_temperature', 'output_dir')}
        if reference_args is None:
            reference_args = comparable
        elif comparable != reference_args:
            raise ValueError(f'{name}: training protocol mismatch')
        if training['event'] != 'complete' or training['tokens_processed'] != 4096000:
            raise ValueError(f'{name}: incomplete training')
        table[name] = {
            'training_seed': header['args']['seed'],
            'input_temperature': header['args']['input_temperature'],
            'hidden_temperature': header['args']['hidden_temperature'],
            'ppl': values,
            'N4_mean': statistics.mean(seeds),
            'N4_sample_std': statistics.stdev(seeds),
            'N4_increase_percent_vs_original': (statistics.mean(seeds) / baseline['perplexity'] - 1) * 100,
            'training_seconds': training['elapsed_seconds'],
            'training_peak_allocated_bytes': training['peak_allocated_bytes'],
            'validation': training['final_validation_by_sample_count'],
        }
    report = {
        'baseline_ppl': baseline['perplexity'],
        'conditions': table,
        'scope': 'One trained checkpoint per condition; three inference seeds, not three training replications.',
    }
    output = args.results / 'comparison.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
