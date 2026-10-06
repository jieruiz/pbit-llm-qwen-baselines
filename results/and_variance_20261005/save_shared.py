from pathlib import Path
import hashlib,json,shutil,zipfile
stage=Path(__file__).resolve().parent
repo=stage.parents[1]/'pbit-llm-qwen-baselines'
shared=Path(r'E:\BaiduSyncdisk\p-bit\pbit-qwen').resolve()
destination=shared/'results/and_variance_20261005'
assert destination.resolve().is_relative_to(shared)
archive=destination/'best-checkpoints.zip'
assert {p.name for p in destination.iterdir()}=={'best-checkpoints.zip'}
status=json.loads((stage/'checkpoint-archive-status.json').read_text(encoding='utf-8'))
def sha_stream(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert archive.stat().st_size==status['bytes']
with archive.open('rb') as stream:assert sha_stream(stream)==status['archive_sha256']
with zipfile.ZipFile(archive) as z:
    index=json.loads(z.read('best-checkpoints-manifest.json'))
    assert len(index['checkpoints'])==140
    for row in index['checkpoints']:
        assert z.getinfo(row['file']).file_size==row['bytes']
        with z.open(row['file']) as stream:assert sha_stream(stream)==row['sha256'],row['file']
shutil.copytree(repo/'results/and_variance_20261005',destination,dirs_exist_ok=True)
for name in ('results-review.zip','run-identity.json','initial-code.zip','formal-code.zip'):
    shutil.copy2(stage/name,destination/name)
record={'verified_checkpoints':140,'storage':'zip members verified individually; not extracted',
        'archive_sha256':status['archive_sha256'],'formal_best':120,'common_initializer':20}
(destination/'local-verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
readme=shared/'README.md'
assert readme.resolve().is_relative_to(shared)
text=readme.read_text(encoding='utf-8')
entry='- 2026-10-05：二十层AND FFN四路径输出方差约束对照，三个配对训练种子。见[报告](results/and_variance_20261005/RESULTS_ZH.md)；包含140个逐成员哈希核验的checkpoint（120个正式best和20个共同初始化），以ZIP保存。\n'
if 'results/and_variance_20261005/' not in text:
    readme.write_text(text.rstrip()+'\n\n'+entry,encoding='utf-8',newline='\n')
print(json.dumps({'directory':str(destination),**record},ensure_ascii=False,indent=2))
