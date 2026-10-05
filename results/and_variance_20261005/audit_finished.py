"""Independent, stdlib-only audit of completed A/B training and all 26 tests.

Run after publication. Preserve raw reports; fail before writing a receipt if
any configuration, evaluation scope, or checkpoint identity is inconsistent.
"""
from pathlib import Path
import hashlib
import json
import math

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[1] / 'pbit-llm-qwen-baselines/results/and_variance_20261005'
REMOTE = '/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005/'
LAYERS = [1, *range(4, 23)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value), value


def main():
    published = read(STAGE / 'PUBLISHED.json')
    manifest = read(ROOT / 'result-manifest.json')
    checkpoints = {row['file']: row for row in manifest['checkpoints']}
    for row in manifest['files']:
        name = row['file'].removeprefix('results/')
        path = ROOT / name
        blob = path.read_bytes()
        assert len(blob) == row['bytes'], name
        assert hashlib.sha256(blob).hexdigest() == row['sha256'], name
    assert read(ROOT / 'execution-status.json')['application_exit_code'] == 0
    assert (ROOT / 'RUN_COMPLETE').is_file()
    assert (ROOT / 'TRAINING_COMPLETE').is_file()
    comparison = read(ROOT / 'comparison.json')
    assert comparison == published['comparison']
    finite(comparison)
    initial = read(ROOT / 'initial/initializer_manifest.json')['checkpoints']
    assert [row['layer'] for row in initial] == LAYERS
    initial_hashes = {row['path']: row['sha256'] for row in initial}
    assert all(row['step'] == 800 for row in initial)
    for row in initial:
        assert checkpoints[row['path'].removeprefix(REMOTE)]['sha256'] == row['sha256']
    scales_path = ROOT / 'reference_scales.json'
    scales_hash = hashlib.sha256(scales_path.read_bytes()).hexdigest()
    scales = read(scales_path)
    assert scales['frozen'] is True and scales['calibration_tokens'] == 32768
    assert scales['batch_seed'] == 719
    assert scales['split'] == 'train excluding final 65536 tokens'
    assert scales['source'] == 'unmodified Qwen FFN mean squared output'
    assert sorted(map(int, scales['powers'])) == LAYERS
    assert all(math.isfinite(v) and v > 0 for v in scales['powers'].values())
    weight = read(ROOT / 'locked_selection.json')['selected_lambda']
    assert weight == .1
    expected = dict(steps=2000, sequence_length=256, micro_batch_size=1,
                    gradient_accumulation=4, validation_tokens=65536,
                    validation_batches=64, validation_every=250, sample_count=4,
                    learning_rate=1e-5, weight_decay=.01, warmup_steps=50,
                    kd_weight=.8, kd_temperature=1., max_gradient_norm=1.,
                    validation_seed=12345)
    first_validation = None
    evaluated = []
    summaries = []
    for seed in (0, 1, 2):
        pair_args = []
        for condition in ('control', 'variance'):
            label = f'{condition}_seed{seed}'
            folder = ROOT / label
            logs = [json.loads(line) for line in
                    (folder / 'train_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            finite(logs)
            start, complete = logs[0], logs[-1]
            assert start['event'] == 'start' and complete['event'] == 'complete'
            assert start['layers'] == LAYERS and start['trainable_parameters'] == 264230400
            args = start['args']
            assert all(args[key] == value for key, value in expected.items()), label
            assert args['seed'] == seed
            assert args['variance_weight'] == (0 if condition == 'control' else weight)
            assert args['initial_checkpoint_sha256'] == initial_hashes
            assert args['reference_scales_sha256'] == scales_hash
            pair_args.append({k: v for k, v in args.items()
                              if k not in ('output_dir', 'variance_weight')})
            validations = [row for row in logs if row['event'] == 'validation']
            assert [row['step'] for row in validations] == list(range(0, 2001, 250))
            if first_validation is None:
                first_validation = validations[0]
            assert validations[0] == first_validation
            assert all(row['scored_tokens'] == 16320 for row in validations)
            best = min(validations, key=lambda row: row['perplexity'])
            summary = read(folder / 'summary.json')
            assert summary == complete and summary['steps'] == 2000
            assert summary['best_step'] == best['step']
            assert summary['best_validation_perplexity'] == best['perplexity']
            assert [row for row in logs if row['event'] == 'train'][-1]['step'] == 2000
            for phase in ('best', 'latest', 'final'):
                for layer in LAYERS:
                    assert f'results/{label}/{phase}_layer{layer}.pt' in checkpoints
            diagnostics = read(folder / 'field_diagnostics.json')
            finite(diagnostics)
            assert diagnostics['windows'] == 16 and diagnostics['sequence_length'] == 256
            assert diagnostics['seed'] == 12345
            assert diagnostics['split'] == 'held-out final 65536 train tokens'
            assert diagnostics['reference_scales_sha256'] == scales_hash
            for field in ('layers', 'common_teacher_inputs'):
                assert sorted(map(int, diagnostics[field])) == LAYERS
            for row in diagnostics['layers'].values():
                assert row['n4_variance_nmse'] >= 0 and row['same_input_bias_nmse'] >= 0
                assert row['output_power_to_reference'] >= 0
                assert 0 <= row['gate_saturation'] <= 1 and 0 <= row['value_saturation'] <= 1
            for row in diagnostics['common_teacher_inputs'].values():
                assert row['n4_variance_nmse'] >= 0 and row['bias_nmse'] >= 0
            assert [row['layer'] for row in diagnostics['checkpoints']] == LAYERS
            for row in diagnostics['checkpoints']:
                source = f'results/{label}/best_layer{row["layer"]}.pt'
                assert row['path'] == REMOTE + source
                assert row['sha256'] == checkpoints[source]['sha256']
            budgets = [(4, 0), (4, 1), (4, 2), (0, 0)]
            if seed == 0:
                budgets.append((16, 0))
            for count, infer_seed in budgets:
                name = f'ppl_n{count}_s{infer_seed}.json'
                evaluation = read(folder / name)
                finite(evaluation)
                fields = dict(sample_count=count, seed=infer_seed, source_tokens=299078,
                              start_token=0, end_token=299078, corpus_tokens=299078,
                              scored_tokens=299077, max_length=2048, stride=1024,
                              windows=292, replacement_count=20,
                              student_parameters_total=264230400)
                assert all(evaluation[k] == v for k, v in fields.items()), (label, name)
                assert evaluation['layers'] == LAYERS
                assert [row['layer'] for row in evaluation['checkpoints']] == LAYERS
                for row in evaluation['checkpoints']:
                    source = f'results/{label}/best_layer{row["layer"]}.pt'
                    assert row['path'] == REMOTE + source
                    assert row['sha256'] == checkpoints[source]['sha256']
                    assert row['bits'] == 4 and row['step'] == best['step']
                    assert row['parameters'] == 13211520
                evaluated.append(f'{label}_n{count}_s{infer_seed}')
            summaries.append({'model': label, 'trained_steps': 2000,
                              'best_step': best['step'], 'evaluations': len(budgets)})
        assert pair_args[0] == pair_args[1], seed
    commands = [json.loads(line) for line in
                (ROOT / 'commands.jsonl').read_text(encoding='utf-8').splitlines()]
    names = [row['name'] for row in commands]
    assert len(evaluated) == 26 and len(set(evaluated)) == 26
    assert all(names.count(name) == 1 for name in evaluated)
    assert max(names.index(row['model']) for row in summaries) < min(names.index(n) for n in evaluated)
    archive_index = read(ROOT / 'best-checkpoints-manifest.json')['checkpoints']
    assert len(archive_index) == 140 and len({row['file'] for row in archive_index}) == 140
    expected_sources = {f'results/{row["model"]}/best_layer{layer}.pt'
                        for row in summaries for layer in LAYERS}
    expected_sources.update(row['path'].removeprefix(REMOTE) for row in initial)
    assert {row['source_file'] for row in archive_index} == expected_sources
    for row in archive_index:
        source = checkpoints[row['source_file']]
        assert row['sha256'] == source['sha256'] and row['bytes'] == source['bytes']
    receipt = {'scope': 'strict raw bytes; paired rules; six20-layer heldout diagnostics; all26 full tests and best weight identities;140 archive index identities',
               'models': summaries, 'evaluations': 26, 'archive_checkpoints': 140,
               'source_commit': '13be8a871cd714e430db00ae36d141abb16f65e2',
               'prespecified_target_met': comparison['prespecified_target_met'],
               'note': 'N0 is conditional-mean propagation; archive member byte verification is recorded separately by save_shared.py'}
    (ROOT / 'completion-audit.json').write_text(json.dumps(receipt, indent=2) + '\n',
                                               encoding='utf-8', newline='\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
