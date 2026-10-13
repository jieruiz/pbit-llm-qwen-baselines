"""Download one pinned official checkpoint; reuse existing offline benchmarks."""
import hashlib, importlib.metadata, json, os, shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3.5-0.8B'
dest = ROOT/'models/Qwen3.5-0.8B'
api = HfApi(endpoint=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
info = api.model_info(MODEL)
print('PINNED_REVISION', info.sha, flush=True)
snapshot_download(MODEL, revision=info.sha, local_dir=dest, max_workers=4)
files = []
for p in sorted(dest.rglob('*')):
    if not p.is_file() or '.cache' in p.parts: continue
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    files.append({'path':str(p.relative_to(dest)), 'bytes':p.stat().st_size, 'sha256':h.hexdigest()})
versions = {k:importlib.metadata.version(k) for k in ['torch','transformers','lm_eval','datasets','peft','huggingface-hub']}
manifest = {'repo_id':MODEL, 'revision':info.sha, 'endpoint':api.endpoint, 'files':files, 'versions':versions,
    'scope':'Official post-trained checkpoint; text-only inference; original weights; no p-bit replacement'}
(ROOT/'deployment_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
os.system(f'{ROOT}/venv/bin/python -m pip freeze > {ROOT}/requirements.lock.txt')
source = ROOT/'sources'; source.mkdir(exist_ok=True)
import transformers, lm_eval
for package, sub in [(transformers,'models/qwen3_5'),(lm_eval,'models/huggingface.py')]:
    p = Path(package.__file__).parent/sub
    target = source/package.__name__/sub
    target.parent.mkdir(parents=True,exist_ok=True)
    if p.is_dir(): shutil.copytree(p,target,dirs_exist_ok=True)
    else: shutil.copy2(p,target)
print('DEPLOYED',json.dumps({'revision':info.sha,'bytes':sum(x['bytes'] for x in files),'versions':versions}),flush=True)
