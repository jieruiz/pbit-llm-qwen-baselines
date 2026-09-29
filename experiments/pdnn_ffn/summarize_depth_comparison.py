"""Validate the complete matched-budget depth experiment and summarize seeds."""
import argparse
import json
import math
import statistics
from pathlib import Path

from run_depth_comparison import SHAPES, parameter_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    assert (root / 'RUN_COMPLETE').is_file(), 'Sweep did not finish'
    baseline = json.loads(args.baseline.read_text())
    report = {'baseline_ppl': baseline['perplexity'], 'depths': {}}
    common_args = None

    def read_eval(directory, count, seed, hidden):
        path = directory / f'ppl_paths{count}_infer{seed}.json'
        result = json.loads(path.read_text())
        for key in ('corpus_tokens', 'scored_tokens', 'max_length', 'stride'):
            assert result[key] == baseline[key], (path, key)
        assert result['layer'] == 12 and result['sample_count'] == count and result['seed'] == seed, path
        config = result['student_config']
        assert config['coding'] == 'binary' and config['input_temperature'] == 0.125 and config['hidden_temperature'] == 0.5, path
        assert tuple(config['hidden_sizes']) == hidden, path
        assert math.isfinite(result['perplexity']) and result['perplexity'] > 0, path
        return result['perplexity']

    for depth, hidden in SHAPES.items():
        independent, training, validation = [], [], []
        for seed in (0, 1, 2):
            directory = root / f'depth{depth}' / f'train_seed{seed}'
            summary = json.loads((directory / 'summary.json').read_text())
            header = json.loads((directory / 'train_metrics.jsonl').read_text().splitlines()[0])
            assert summary['event'] == 'complete' and summary['tokens_processed'] == 4096000, directory
            assert header['args']['seed'] == seed and tuple(header['args']['hidden_sizes']) == hidden, directory
            assert header['student_parameters'] == parameter_count(hidden), directory
            comparable = {k: v for k, v in header['args'].items() if k not in ('hidden_sizes', 'seed', 'output_dir')}
            if common_args is None:
                common_args = comparable
            assert comparable == common_args, directory
            independent.append(read_eval(directory, 4, 0, hidden))
            training.append(summary['elapsed_seconds'])
            validation.append(summary['final_validation_by_sample_count']['4'])
        seed0 = root / f'depth{depth}' / 'train_seed0'
        scan = {str(n): read_eval(seed0, n, 0, hidden) for n in (0, 1, 4, 8, 16)}
        inference_seeds = [read_eval(seed0, 4, s, hidden) for s in (0, 1, 2)]
        report['depths'][str(depth)] = {
            'shape': [896, *hidden, 896], 'parameters': parameter_count(hidden),
            'parameter_delta_percent': (parameter_count(hidden) / parameter_count(SHAPES[2]) - 1) * 100,
            'ppl_by_training_seed': independent, 'ppl_mean': statistics.mean(independent),
            'ppl_training_std': statistics.stdev(independent),
            'increase_percent_vs_original': (statistics.mean(independent) / baseline['perplexity'] - 1) * 100,
            'training_seconds': training,
            'validation_by_training_seed': validation,
            'seed0_path_scan': scan,
            'seed0_ppl_by_inference_seed': inference_seeds,
            'seed0_ppl_inference_mean': statistics.mean(inference_seeds),
            'seed0_ppl_inference_std': statistics.stdev(inference_seeds),
        }
    reference = report['depths']['2']['ppl_by_training_seed']
    for depth in ('3', '4'):
        differences = [b-a for a, b in zip(reference, report['depths'][depth]['ppl_by_training_seed'])]
        report['depths'][depth]['paired_ppl_difference_vs_depth2'] = differences
        report['depths'][depth]['paired_difference_mean'] = statistics.mean(differences)
    report['scope'] = 'Three independent training seeds per depth. Primary PPL uses four complete paths and inference seed 0. Inference-seed variability is separately measured only for training seed 0.'
    (root / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
