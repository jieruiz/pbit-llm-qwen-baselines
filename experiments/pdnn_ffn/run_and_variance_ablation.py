"""Frozen sequential pipeline after successful reconstruction (one GPU only)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from rebuild_and_variance_initial import LAYERS, sha, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--assets',required=True)
    args=parser.parse_args()
    root=Path(args.root).resolve();assets=Path(args.assets).resolve()
    assert root.is_relative_to(Path('/home/Weican_Chen'))
    assert assets.is_relative_to(Path('/home/Weican_Chen'))
    assert (root/'initial/REBUILD_COMPLETE').is_file()
    assert torch.cuda.is_available()
    exp=Path(__file__).resolve().parent
    result=root/'results'
    result.mkdir(exist_ok=False)
    logs=root/'formal-logs';logs.mkdir(exist_ok=False)
    model=assets/'models/Qwen2.5-0.5B'
    train=assets/'data/wikitext-2/wiki.train.raw';test=assets/'data/wikitext-2/wiki.test.raw'
    source=json.loads((root/'initial/initializer_manifest.json').read_text())
    for item in source['checkpoints']:
        assert sha(item['path'])==item['sha256']
    initial=[root/f'initial/joint/best_layer{layer}.pt' for layer in LAYERS]
    write(result/'plan.json',{'initialization':source,'training_seeds':[0,1,2],
        'steps':2000,'K':4,'N':4,'main_inference_seeds':[0,1,2],
        'lambda_screen':[0,.01,.1,1],'formal_test_before_training_complete':False,
        'target_relative_ppl_gain':.02,'required_paired_wins':2,
        'diagnostics':'mean-field each model; N16 only seed0 A/B (fixed diagnostic subset)',
        'scope':'one initialization, three paired continuation seeds, one corpus'})
    def run(name,script,argv):
        command=[sys.executable,'-u',str(exp/script),*map(str,argv)]
        print('START',name,flush=True)
        write(result/'progress.json',{'stage':name,'state':'running'})
        started=time.monotonic()
        with (logs/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',name,round(time.monotonic()-started,2),flush=True)
        with (result/'commands.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps({'name':name,'command':command,'seconds':time.monotonic()-started})+'\n')
    scales=result/'reference_scales.json'
    run('reference_scales','and_variance_tools.py',['calibrate','--model',model,'--train-text',train,'--output',scales])
    common=['--model',model,'--checkpoints',*initial,'--train-text',train,'--reference-scales',scales,
            '--sample-count',4,'--validation-seed',12345]
    # B's worst screened coefficient is used to catch memory/gradient failures
    # before any expensive formal training. Both controls get the same short test.
    for label,weight in [('control',0),('variance',1)]:
        run(f'smoke_{label}','train_joint_and_variance.py',[*common,'--steps',2,
            '--sequence-length',256,'--gradient-accumulation',4,'--validation-batches',2,
            '--validation-every',2,'--variance-weight',weight,'--output-dir',result/f'smoke_{label}'])
        summary=json.loads((result/f'smoke_{label}/summary.json').read_text())
        assert summary['event']=='complete' and summary['steps']==2
    run('pilot','and_variance_tools.py',['pilot','--model',model,'--train-text',train,
        '--checkpoint',initial[LAYERS.index(12)],'--scales',scales,'--output',result/'pilot'])
    selection=json.loads((result/'pilot/selection.json').read_text())
    weight=selection['selected_lambda']
    assert weight in (.01,.1,1.)
    write(result/'locked_selection.json',selection)
    for seed in (0,1,2):
        for condition,coefficient in [('control',0),('variance',weight)]:
            label=f'{condition}_seed{seed}'
            run(label,'train_joint_and_variance.py',[*common,'--steps',2000,
                '--validation-batches',64,'--validation-every',250,'--seed',seed,
                '--variance-weight',coefficient,'--output-dir',result/label])
    (result/'TRAINING_COMPLETE').write_text('six formal models complete\n',encoding='utf-8')
    for seed in (0,1,2):
        for condition in ('control','variance'):
            label=f'{condition}_seed{seed}'
            folder=result/label
            paths=[folder/f'best_layer{layer}.pt' for layer in LAYERS]
            run(f'{label}_probe','and_variance_tools.py',['probe','--model',model,'--train-text',train,
                '--checkpoints',*paths,'--scales',scales,'--output',folder/'field_diagnostics.json'])
            pairs=[(4,0),(4,1),(4,2),(0,0)]
            if seed==0:pairs.append((16,0))
            for count,infer_seed in pairs:
                run(f'{label}_n{count}_s{infer_seed}','evaluate_multi_layer_multithreshold_and.py',
                    ['--model',model,'--checkpoints',*paths,'--text-file',test,'--sample-count',count,
                     '--seed',infer_seed,'--output',folder/f'ppl_n{count}_s{infer_seed}.json'])
    (result/'RUN_COMPLETE').write_text('complete\n',encoding='utf-8')
    write(result/'progress.json',{'stage':'complete','full_test_evaluations':26})


if __name__=='__main__':
    main()
