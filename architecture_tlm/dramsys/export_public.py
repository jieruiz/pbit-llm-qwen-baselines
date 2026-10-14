"""Copy aggregate results to a public directory without internal run metadata.

No source files are removed. Public export omits raw requests/logs and retains
only synthetic-model aggregate counts, timing/energy and numerical references.
"""
import argparse,copy,json,csv
from pathlib import Path

def sanitize(value):
    omitted={'slurm_job_id','slurm_array_task_id','address_regions','events'}
    if isinstance(value,dict):
        return {key:sanitize(v) for key,v in value.items() if key not in omitted}
    if isinstance(value,list):return [sanitize(v) for v in value]
    if isinstance(value,str) and (value.startswith('/home/') or ':\\Users\\' in value):
        return '[internal runtime path omitted]'
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();source=Path(a.source).resolve();dest=Path(a.destination).resolve()
    if source==dest or source in dest.parents:raise ValueError('use a distinct export directory')
    dest.mkdir(parents=True,exist_ok=True)
    for path in source.glob('*.json'):
        value=sanitize(json.loads(path.read_text(encoding='utf-8')))
        if isinstance(value,dict):
            value['publication']=dict(raw_artifacts_omitted=True,
                private_artifacts='raw request traces, build/runtime logs and scheduler identities retained locally',
                scope='synthetic public Qwen shape model; no weights, prompts, KV contents or private dataset')
        if isinstance(value,dict) and 'scenario_name' in value:
            name=value['scenario_name'];ext=value['config']['external_memory']
            ext['bridge_executable']='build/pbit_dram_bridge'
            ext['configuration']=f"architecture_tlm/dramsys/ddr4_2ch_{name.rsplit('_',1)[-1]}.json"
            ext['output_prefix']=f'architecture_tlm/results/dramsys/{name}'
        (dest/path.name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    for name in ('summary.csv','RESULTS_ZH.md'):
        path=source/name
        if path.exists():(dest/name).write_bytes(path.read_bytes())
    print(json.dumps(dict(exported=str(dest),raw_artifacts_omitted=True)))

if __name__=='__main__':main()
