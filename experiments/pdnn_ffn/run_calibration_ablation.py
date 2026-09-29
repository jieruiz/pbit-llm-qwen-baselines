"""Prespecified ten-FFN paired experiment, executed serially in one GPU job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAYERS = (7, 8, 9, 10, 11, 12, 13, 14, 15, 18)


def main():
    p = argparse.ArgumentParser()
    for name in ('model', 'train-text', 'test-text', 'four-layer-checkpoints', 'baseline', 'output-dir'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Refusing to overwrite results')
    root.mkdir(parents=True, exist_ok=True)
    (root/'command_logs').mkdir()
    scripts = Path(__file__).resolve().parent
    plan = dict(layers=LAYERS, seeds=[0, 1, 2], inference_seeds=[0, 1, 2],
                conditions=['fixed', 'calibrated'], steps=2000, sample_count=4,
                learning_rate=1e-5, calibration_lr=3e-4, calibration_weight_decay=0.,
                calibration_scale_bounds=[.25, 4.], calibration_shift_bounds=[-4., 4.],
                target_relative_ppl_improvement=.01, required_training_seed_wins=2,
                validation='64 fixed 256-token windows, RNG 12345, final 65536 train tokens',
                selection='lowest validation PPL every 500 steps; test never selects checkpoint',
                initialization='reconstruct published ten-layer joint model, then fresh optimizer per condition',
                job_id=os.environ.get('SLURM_JOB_ID'))
    for name in ('train', 'test'):
        plan[name+'_sha256'] = hashlib.sha256(Path(getattr(args, name+'_text')).read_bytes()).hexdigest()
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')

    def run(script, arguments, label):
        command = [sys.executable, str(scripts/script), *map(str, arguments)]
        print('START', label, flush=True)
        start = time.perf_counter()
        with (root/'command_logs'/f'{label}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        with (root/'events.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(dict(label=label, command=command, seconds=time.perf_counter()-start,
                                      returncode=result.returncode))+'\n')
        if result.returncode:
            raise RuntimeError(f'{label} failed')
        print('DONE', label, round(time.perf_counter()-start, 2), flush=True)

    def evaluate(checkpoints, output, count=4, seed=0, smoke=False):
        cmd = ['--model', args.model, '--checkpoints', *checkpoints, '--text-file', args.test_text,
               '--output', output, '--sample-count', count, '--seed', seed]
        if smoke: cmd += ['--max-tokens', 512, '--max-length', 256, '--stride', 128]
        run('evaluate_calibrated_perplexity.py', cmd, output.parent.name+'_'+output.stem)

    def train(checkpoints, output, seed=0, calibrated=False, smoke=False):
        cmd = ['--model', args.model, '--train-text', args.train_text, '--checkpoints', *checkpoints,
               '--output-dir', output, '--steps', 2000, '--seed', seed, '--validation-batches', 64,
               '--validation-every', 500, '--log-every', 50, '--validation-seed', 12345]
        if calibrated: cmd += ['--calibration']
        if smoke:
            cmd += ['--steps', 4, '--sequence-length', 32, '--gradient-accumulation', 1,
                    '--validation-batches', 2, '--validation-every', 2, '--warmup-steps', 1, '--log-every', 1]
        run('train_joint_calibration.py', cmd, output.name+'_train')

    four = {layer: Path(args.four_layer_checkpoints)/f'best_layer{layer}.pt' for layer in range(10, 14)}
    baseline = json.loads(Path(args.baseline).read_text())
    assert abs(baseline['perplexity'] - 11.652735047455899) < 1e-5
    assert baseline['scored_tokens'] == 299077
    run('test_calibrated_pdnn.py', ['-v'], 'unit_tests')
    for calibrated in (False, True):
        output = root/('smoke_calibrated' if calibrated else 'smoke_fixed')
        train([four[12]], output, calibrated=calibrated, smoke=True)
        evaluate([output/'final_layer12.pt'], output/'ppl.json', smoke=True)
    # At initialization the paired models must have identical validation outputs.
    def first_validation(path):
        return next(json.loads(line) for line in path.read_text().splitlines()
                    if json.loads(line)['event']=='validation')
    a = first_validation(root/'smoke_fixed'/'train_metrics.jsonl')
    b = first_validation(root/'smoke_calibrated'/'train_metrics.jsonl')
    assert a['perplexity'] == b['perplexity'], 'Initial models are not equivalent'

    initial = dict(four)
    for layer in LAYERS:
        if layer in initial: continue
        output = root/'local'/f'layer{layer}'
        run('train_full_path_distillation.py', ['--model', args.model, '--train-text', args.train_text,
            '--layer', layer, '--input-temperature', .25, '--hidden-temperature', 1.,
            '--output-dir', output, '--seed', 0], f'local_layer{layer}')
        initial[layer] = output/'student_sampled.pt'
    checkpoints = [initial[layer] for layer in LAYERS]
    evaluate(checkpoints, root/'staged_initial_ppl.json')
    staged = json.loads((root/'staged_initial_ppl.json').read_text())['perplexity']
    assert abs(staged - 24.813126) < .002, f'Staged initializer mismatch: {staged}'
    reference = root/'reference_ten'
    run('train_joint_full_path_distillation.py', ['--model', args.model, '--train-text', args.train_text,
        '--checkpoints', *checkpoints, '--output-dir', reference, '--seed', 0], 'reference_ten_train')
    ten = [reference/f'best_layer{layer}.pt' for layer in LAYERS]
    evaluate(ten, reference/'ppl_n4_s0.json')
    reproduced = json.loads((reference/'ppl_n4_s0.json').read_text())['perplexity']
    assert abs(reproduced - 19.223202) < .002, f'Ten-layer reference mismatch: {reproduced}'
    (root/'initializer_manifest.json').write_text(json.dumps([
        dict(layer=layer, file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for layer, path in zip(LAYERS, ten)], indent=2)+'\n', encoding='utf-8')

    for seed in (0, 1, 2):
        for condition in ('fixed', 'calibrated'):
            output = root/f'{condition}_seed{seed}'
            train(ten, output, seed=seed, calibrated=condition=='calibrated')
    # All six training runs finish before inspecting any candidate test score.
    for seed in (0, 1, 2):
        for condition in ('fixed', 'calibrated'):
            output = root/f'{condition}_seed{seed}'
            best = [output/f'best_layer{layer}.pt' for layer in LAYERS]
            run('probe_calibrated_fields.py', ['--model', args.model, '--train-text', args.train_text,
                '--checkpoints', *best, '--output', output/'field_diagnostics.json'], output.name+'_fields')
            for n, s in ((4, 0), (4, 1), (4, 2), (0, 0), (16, 0)):
                evaluate(best, output/f'ppl_n{n}_s{s}.json', count=n, seed=s)
    (root/'RUN_COMPLETE').write_text('complete\n', encoding='utf-8')


if __name__ == '__main__':
    main()
