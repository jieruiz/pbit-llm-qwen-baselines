"""Sequential, one allocated GPU reconstruction of the published AND method."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

LAYERS = (1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--assets', required=True)
    args = parser.parse_args()
    root, assets = Path(args.root).resolve(), Path(args.assets).resolve()
    personal = Path('/home/Weican_Chen').resolve()
    assert root.is_relative_to(personal) and assets.is_relative_to(personal)
    assert torch.cuda.is_available()
    exp = Path(__file__).resolve().parent
    model, train, test = assets/'models/Qwen2.5-0.5B', assets/'data/wikitext-2/wiki.train.raw', assets/'data/wikitext-2/wiki.test.raw'
    expected = {'model': '88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342',
                'train': '6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57',
                'test': '173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08'}
    actual = {'model':sha(model/'model.safetensors'), 'train':sha(train), 'test':sha(test)}
    if actual != expected:
        raise RuntimeError('model/corpus hashes do not match published baseline')
    write(root/'initial/assets.json', {'sha256':actual, 'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__})
    def run(name, script, argv):
        command = [sys.executable, '-u', str(script), *map(str,argv)]
        print('START', name, flush=True)
        begin = time.monotonic()
        with (root/'logs'/f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('DONE', name, round(time.monotonic()-begin,2), flush=True)
        write(root/'initial/progress.json', {'last_completed':name})
    run('variance_math_tests', exp/'test_and_variance_loss.py', ['-v'])
    run('baseline', exp.parents[1]/'baselines/perplexity.py', ['--model',model,'--text-file',test,'--output',root/'initial/baseline.json'])
    baseline = json.loads((root/'initial/baseline.json').read_text())
    assert abs(baseline['perplexity']-11.652735047455899)<.002
    for layer in LAYERS:
        run(f'local_layer{layer}', exp/'train_multithreshold_and.py',
            ['--model',model,'--train-text',train,'--bits',4,'--layer',layer,
             '--output-dir',root/f'initial/local/layer{layer}'])
    paths = [root/f'initial/local/layer{layer}/student_sampled.pt' for layer in LAYERS]
    run('joint_reference', exp/'train_joint_multithreshold_and.py',
        ['--model',model,'--checkpoints',*paths,'--train-text',train,
         '--output-dir',root/'initial/joint','--steps',1000,'--sample-count',4,'--seed',0])
    paths = [root/f'initial/joint/best_layer{layer}.pt' for layer in LAYERS]
    records=[]
    for layer,path in zip(LAYERS, paths):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert payload['student_type']=='multithreshold_and_v1' and payload['student_config']['bits']==4
        assert payload['training_args']['layer']==layer
        records.append({'layer':layer,'path':str(path),'sha256':sha(path),'step':payload['step'], 'bytes':path.stat().st_size})
    write(root/'initial/initializer_manifest.json', {'upstream':'8068bfbb71d892ef8b0850dea391aa612739aa4f',
        'method':'fresh 20 local fits, original joint 1000 steps and selection; all pairs reuse these exact weights',
        'checkpoints':records})
    run('reference_n4_seed0', exp/'evaluate_multi_layer_multithreshold_and.py',
        ['--model',model,'--checkpoints',*paths,'--text-file',test,'--sample-count',4,'--seed',0,
         '--output',root/'initial/reference_n4_seed0.json'])
    (root/'initial/REBUILD_COMPLETE').write_text('complete\n', encoding='utf-8')


if __name__ == '__main__':
    main()
