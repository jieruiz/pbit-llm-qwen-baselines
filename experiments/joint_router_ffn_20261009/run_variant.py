"""Run a published block-screening preset without overwriting archived results."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['screen_exact','screen_sampled'])
    p.add_argument('--context',type=int,choices=[2048,8192],default=8192)
    p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    config=json.loads((ROOT/'configs'/f'{a.mode}_{a.context}.json').read_text())
    output=a.output.resolve() if a.output else ROOT/'outputs'/f'{a.mode}_c{a.context}_s{a.seed}.json'
    if output.is_relative_to((ROOT/'results').resolve()):p.error('results/ is the published archive; use outputs/ or another directory')
    cmd=[sys.executable,str(ROOT/'evaluate.py')]
    for key,value in config.items():cmd.extend(['--'+key,str(value)])
    cmd.extend(['--seed',str(a.seed),'--output',str(output)])
    if a.dry_run:
        print(json.dumps(dict(command=cmd,gpu=a.gpu),indent=2));return
    if output.exists():p.error(f'Output exists: {output}')
    for path in [ROOT/'models/Qwen2.5-0.5B/config.json',ROOT/'data/wikitext-2/wiki.test.raw']:
        if not path.is_file():p.error(f'Missing asset: {path}; see README.md')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
