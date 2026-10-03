"""Validate and summarize all input-encoding experiment outputs."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path('results/input_multibit_layer12_20260930')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    protocol=read(ROOT/'protocol.json')
    assert read(ROOT/'status.json')['status']=='complete'
    rows=[]
    for config in protocol['configs']:
        name=config['name']
        records={}
        checkpoint=read(ROOT/'checkpoints'/f'{name}.json')
        assert checkpoint['phase_step']==6000
        for count,seed in protocol['evaluation_pairs']:
            r=read(ROOT/'evaluation'/name/f'samples{count}_seed{seed}.json')
            assert r['scored_tokens']==299077
            assert r['checkpoint'].endswith(checkpoint['checkpoint'])
            records[(count,seed)]=r
        n4=[records[(4,seed)]['perplexity'] for seed in range(3)]
        summary=read(ROOT/'training'/name/'summary.json')
        moments=read(ROOT/'moments'/f'{name}.json')['normalized']
        rows.append({
            'name':name,'input_encoding':config['input_encoding'],'input_bits':config['input_bits'],
            'student_parameters':checkpoint['student_parameters'],
            'mean_field_ppl':records[(0,0)]['perplexity'],'n4_ppl_mean':statistics.mean(n4),
            'n4_ppl_population_sd':statistics.pstdev(n4),'n4_ppl_by_seed':n4,
            'n16_ppl':records[(16,0)]['perplexity'],
            'bias_nmse':moments['corrected_bias_nmse'],'single_path_variance_nmse':moments['single_path_variance_nmse'],
            'predicted_n4_nmse':moments['predicted_n4_nmse'],'empirical_n4_nmse':moments['empirical_n4_nmse'],
            'training_seconds':summary['elapsed_seconds'],'peak_training_gib':summary['peak_allocated_bytes']/2**30,
            'peak_evaluation_gib':max(r['peak_allocated_bytes'] for r in records.values())/2**30,
            'checkpoint_sha256':checkpoint['sha256'],
        })
    (ROOT/'comparison.json').write_text(json.dumps({'rows':rows,'notes':[
        'All eight models freshly trained with same 2000+6000 schedule and one training seed.',
        'Deterministic denotes input quantization only; hidden layer remains stochastic for N>0.',
        'Continuous-input control clips to the identical calibrated ranges; it is not original Qwen.',
        'Mean-field is a deterministic surrogate, not generally the exact infinite-sample output.',
        'Adjacent stochastic rounding is an ideal coordinated encoder, not independent sigmoid p-bits.',
    ]},indent=2)+'\n',encoding='utf-8')
    keys=[k for k in rows[0] if k!='n4_ppl_by_seed']
    with (ROOT/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
