from pathlib import Path
import hashlib,json,os,sys,zipfile
root=Path(__file__).resolve().parent
assert root==Path('/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

status={'job_id':os.environ.get('SLURM_JOB_ID'),'application_exit_code':int(sys.argv[1]),
        'run_complete':(root/'results/RUN_COMPLETE').is_file()}
(root/'execution-status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8',newline='\n')
files=[];checkpoints=[]
for folder in ('initial','results','formal-logs'):
    for path in sorted((root/folder).rglob('*')):
        if path.is_file():
            (checkpoints if path.suffix=='.pt' else files).append(path)
for name in ('initial-source-manifest.json','formal-source-manifest.json','initial.slurm','formal.slurm','execution-status.json','collect.py'):
    files.append(root/name)
manifest={'files':[{'file':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in files],
          'checkpoints':[{'file':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in checkpoints]}
(root/'result-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
with zipfile.ZipFile(root/'results-review.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for path in [*files,root/'result-manifest.json']:archive.write(path,path.relative_to(root).as_posix())
print('RESULT_ARCHIVE_SHA256',sha(root/'results-review.zip'),flush=True)
(root/'COLLECT_COMPLETE').write_text('complete\n',encoding='utf-8')
