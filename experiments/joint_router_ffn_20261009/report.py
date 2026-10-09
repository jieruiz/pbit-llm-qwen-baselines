import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
checks=[]
def check(name,ok):
    checks.append(dict(check=name,passed=bool(ok)))
    assert ok,name
check('run complete',read(R/'status.json')['stage']=='complete')
for file in ('source_sha256.json','calibration_sha256.json'):
    for name,h in read(R/file).items():check(name+' hash',hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h)
rows=[read(p) for p in (R/'test').glob('*.json')];check('26 tests',len(rows)==26)
unit=(R/'logs/units.log').read_text();check('unit tests','Ran 4 tests' in unit and '\nOK' in unit)
groups={};tokens={}
for d in rows:
    a=d['args'];key=(a['context'],a['ffn'],a['mode']);groups.setdefault(key,[]).append(d)
    n=math.ceil(a['examples']/a['batch']);calls=n*(a['decode']+1)
    check(str(key)+str(a['seed'])+' scope',d['attention_layers']==24 and d['ffn_layers']==(24 if a['ffn']=='fixed' else 0)
        and d['protected_sha256_before']==d['protected_sha256_after'] and d['scales_unchanged'])
    check(str(key)+str(a['seed'])+' calls',all(c['calls']==calls and c['prefill_calls']==n and c['decode_calls']==n*a['decode'] for c in d['attention_stats']))
    check(str(key)+str(a['seed'])+' ppl',abs(math.exp(sum(x['nll'] for x in d['records'])/d['scored_tokens'])-d['perplexity'])<1e-8)
    tokens.setdefault(a['context'],set()).add(tuple((x['offset'],x['ids_sha256']) for x in d['records']))
check('matched tokens',all(len(v)==1 for v in tokens.values()))
summary=[]
for (ctx,ffn,mode),rs in sorted(groups.items()):
    ppls=[d['perplexity'] for d in rs]
    stats=[c for d in rs for c in d['attention_stats']]
    summary.append(dict(context=ctx,ffn=ffn,mode=mode,seeds=sorted(d['args']['seed'] for d in rs),
        ppl_mean=statistics.mean(ppls),ppl_std=statistics.pstdev(ppls),
        selected_fraction=statistics.mean(c['selected_token_fraction'] for c in stats),
        padded_qk_fraction=statistics.mean(c['padded_qk_fraction'] for c in stats),
        unique_v_fraction=statistics.mean(c['unique_V_fraction'] for c in stats),
        V_events_fraction=statistics.mean(c['V_vector_events_fraction'] for c in stats),
        prefill_seconds=statistics.mean(d['prefill_seconds'] for d in rs),decode_seconds=statistics.mean(d['decode_seconds'] for d in rs)))
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(R/'audit.json').write_text(json.dumps(checks,indent=2)+'\n')
md=['# 当前FFN＋筛块后准确PV / 采样PV（2026-10-09）','',
    '全部24层解码筛选；prefill注意力为原始SDPA，当前固定尺度随机FFN在prefill/decode均启用。Q/K/V/O投影准确。',
    '四个子块均值评分、动态均值/标准差阈值、目标30%；强制首块和最近两块。无退火、无新训练。',
    '两方案均在选中候选内准确Softmax。screen_exact准确加权PV；screen_sampled抽512个候选位置并对V取平均。',
    '同时重跑无筛块的准确/采样PV和原模型，以同范围、同文本对照。不是此前prefill也采样的协议。','',
    '2K为32段×128=4096后缀计分token；8K为16段×128=2048，batch4，随机方案3seed。不是完整语料PPL。','',
    '| 上下文 | FFN | 方法 | PPL均值±seed标准差 | 候选比例 | padded QK比例 | 唯一V比例 |',
    '|---|---|---|---:|---:|---:|---:|']
for r in summary:
    md.append(f"| {r['context']} | {r['ffn']} | {r['mode']} | {r['ppl_mean']:.6f} ± {r['ppl_std']:.6f} | {r['selected_fraction']:.2%} | {r['padded_qk_fraction']:.2%} | {r['unique_v_fraction']:.2%} |")
md+=['','SDPA行未收集候选地址统计，零表示未测，不是无需计算。候选比例与padded比例分别反映逻辑筛选和当前批处理额外填充。',
    'PV采样的512次选择累加不是512次加权MAC；重复地址、归约、摘要维护、采样和调度成本另计。完整KV仍保留。',
    '当前程序先收集候选K，再点积，不先计算完整QK；采样PV直接读取被抽到的V，没有先计算准确PV。',
    '初次decode摘要构建计入耗时；summary.json记录仿真耗时，不能外推硬件速度或能耗。',
    '沿用原模型校准，组合后的阈值泛化/候选覆盖率可能变化，不能将单项历史结果直接相加。',
    f'全部26项测试完成，{len(checks)}条审计通过。']
(R/'RESULTS_ZH.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps(dict(audit=len(checks),summary=summary)),flush=True)
