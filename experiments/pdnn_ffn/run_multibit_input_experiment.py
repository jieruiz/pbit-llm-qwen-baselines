"""Eight controlled layer12 input encoders; calibration never reads test data."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path('results/input_multibit_layer12_20260930')
CONFIGS=[('sigmoid_k1','sigmoid',1),('continuous','continuous',1),
         *[(f'{mode}_k{k}',mode,k) for mode in ('stochastic','deterministic') for k in (1,2,4)]]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def run(script,args,gpu,log):
    command=[sys.executable,f'experiments/pdnn_ffn/{script}',*map(str,args)]
    log.parent.mkdir(parents=True,exist_ok=True)
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
    with log.open('w') as stream:
        subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return command


def main():
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    usage={int(row.split(',')[0]):int(row.split(',')[1]) for row in memory.strip().splitlines()}
    gpus=list(range(8))
    if any(usage[gpu]>500 for gpu in gpus):
        raise RuntimeError(f'An intended GPU is occupied: {usage}')
    ROOT.mkdir(parents=True,exist_ok=False)
    (ROOT/'moments').mkdir()
    protocol={
        'model':'Qwen2.5-0.5B Base','layer':12,'hidden_width':4864,'architecture':'serial','coding':'0/1',
        'configs':[{'name':name,'input_encoding':mode,'input_bits':bits,'gpu':gpu} for gpu,(name,mode,bits) in zip(gpus,CONFIGS)],
        'mean_steps':2000,'sample_steps':6000,'train_samples':4,'seed':0,'batch_size':4,'sequence_length':256,
        'learning_rates':[3e-4,1e-4],'loss':'normalized output MSE + 0.05*(1-cosine)',
        'checkpoint_selection':'student_sampled.pt; verify phase step=6000, never choose on test PPL',
        'evaluation_pairs':[[0,0],[4,0],[4,1],[4,2],[16,0]],
        'calibration':'frozen per-channel abs 99.9th percentile, 32768 sampled train tokens, excludes final65536 validation tokens',
        'projection_audit':'teacher Wg and Wu are frozen for analytical input-only moment diagnostics',
        'distillation':'student matrices trainable, shared initial matrix draws and training windows; no independent bitplane weights',
        'controls':'Fresh sigmoid baseline; clipped continuous-input control; deterministic input quantization with stochastic hidden p-bits',
        'hardware_scope':'Adjacent stochastic rounding coordinates bitplanes; not a demonstrated independent-sigmoid hardware encoder.',
        'cost':'Equal steps/tokens/path count, NOT equal bitplane operations; K-plane input requires K binary-input matrix evaluations per path.',
    }
    write(ROOT/'protocol.json',protocol)
    write(ROOT/'status.json',{'status':'calibrating'})
    try:
        run('calibrate_multibit_input.py',['--output-dir',ROOT],0,ROOT/'logs'/'calibration.log')
        write(ROOT/'status.json',{'status':'training_and_evaluating'})
        def one(gpu,name,mode,bits):
            training=ROOT/'training'/name
            commands=[]
            try:
                commands.append(run('train_full_path_distillation.py',[
                    '--model','models/Qwen2.5-0.5B','--train-text','data/wikitext-2/wiki.train.raw','--output-dir',training,
                    '--layer',12,'--hidden-sizes',4864,'--coding','binary','--temperature-only',
                    '--input-temperature',.125,'--hidden-temperature',.5,'--input-encoding',mode,'--input-bits',bits,
                    '--input-calibration',ROOT/'calibration.json','--mean-steps',2000,'--sample-steps',6000,
                    '--train-samples',4,'--validation-sample-count',4,'--isolate-validation-rng','--seed',0],
                    gpu,ROOT/'logs'/f'{name}_train.log'))
                checkpoint=training/'student_sampled.pt'
                payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert payload['phase']=='full_path_sample_aware' and payload['step']==6000
                assert payload['student_config']['input_encoding']==mode and payload['student_config']['input_bits']==bits
                sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                header=json.loads((training/'train_metrics.jsonl').read_text().splitlines()[0])
                record={'checkpoint':str(checkpoint),'sha256':sha,'phase':payload['phase'],'phase_step':payload['step'],
                        'student_parameters':header['student_parameters'],'config':payload['student_config']}
                del payload
                write(ROOT/'checkpoints'/f'{name}.json',record)
                write(ROOT/'progress'/f'{name}.json',{'status':'evaluating'})
                for count,seed in protocol['evaluation_pairs']:
                    label=f'samples{count}_seed{seed}'
                    commands.append(run('evaluate_full_path_perplexity.py',[
                        '--model','models/Qwen2.5-0.5B','--checkpoint',checkpoint,'--text-file','data/wikitext-2/wiki.test.raw',
                        '--output',ROOT/'evaluation'/name/f'{label}.json','--sample-count',count,'--seed',seed],
                        gpu,ROOT/'logs'/f'{name}_{label}.log'))
                commands.append(run('measure_path_moments.py',[
                    '--model','models/Qwen2.5-0.5B','--checkpoint',checkpoint,'--train-text','data/wikitext-2/wiki.train.raw',
                    '--output',ROOT/'moments'/f'{name}.json'],gpu,ROOT/'logs'/f'{name}_moments.log'))
                write(ROOT/'commands'/f'{name}.json',commands)
                write(ROOT/'progress'/f'{name}.json',{'status':'complete'})
            except Exception as error:
                write(ROOT/'progress'/f'{name}.json',{'status':'failed','error':repr(error)})
                raise
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs=[pool.submit(one,gpu,*config) for gpu,config in zip(gpus,CONFIGS)]
            for job in jobs:
                job.result()
        sources={path:hashlib.sha256(Path('experiments/pdnn_ffn',path).read_bytes()).hexdigest() for path in (
            'full_path_pdnn_ffn.py','multibit_input_ffn.py','train_full_path_distillation.py','calibrate_multibit_input.py',
            'run_multibit_input_experiment.py','evaluate_full_path_perplexity.py','measure_path_moments.py')}
        write(ROOT/'source_sha256.json',sources)
        write(ROOT/'status.json',{'status':'complete'})
    except Exception as error:
        write(ROOT/'status.json',{'status':'failed','error':repr(error)})
        raise


if __name__=='__main__':
    main()
