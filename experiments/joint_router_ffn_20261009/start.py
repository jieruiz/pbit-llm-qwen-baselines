import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
assert not (root/'run_pid.json').exists()
with (root/'launch.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'-u',str(root/'run.py')],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=p.pid,directory=str(root))
(root/'run_pid.json').write_text(json.dumps(result)+'\n')
print(json.dumps(result))
