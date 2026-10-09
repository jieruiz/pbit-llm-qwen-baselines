"""Audit all joint results and report matched suffix metrics, not full-WT2 PPL."""
import hashlib,json,math,random,statistics
from pathlib import Path
from run import name,spec
ROOT=Path(__file__).resolve().parent;R=ROOT/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
checks=[]
def check(label,ok):
    checks.append({'check':label,'passed':bool(ok)})
    if not ok:raise AssertionError(label)
state=read(R/'status.json');check('main complete',state['stage']=='complete')
check('supplement complete',read(R/'supplement_status.json')['stage']=='complete')
for f,h in read(R/'source_sha256.json').items():check('source '+f,hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h)
extra=read(R/'supplement_protocol.json')
check('supplement source',extra['source_sha256']==hashlib.sha256((ROOT/'extra.py').read_bytes()).hexdigest())
for f,h in read(R/'calibration_sha256.json').items():check('calibration '+f,hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h)
check('no test selection supplement',extra['test_used_to_choose'] is False)
cal=read(ROOT/'calibration_projection.json');check('train calibration/validation disjoint',cal['source_tokens']-65536>8192)
unit=(R/'logs/units.log').read_text();check('three unit tests passed','Ran 3 tests' in unit and '\nOK' in unit)
integration=read(ROOT/'integration.json');check('causality and cache integration',integration['causal_verified'] and integration['cached_manual_vs_sdpa_mean']<.06 and integration['cached_manual_vs_sdpa_max']<.8)
V={p.stem:read(p) for p in (R/'validation').glob('*.json')};T={p.stem:read(p) for p in (R/'test').glob('*.json')}
check('counts',len(V)==state['validation_jobs'] and len(T)==state['test_jobs']+4)
for split,rows in [('validation',V),('test',T)]:
    for key,d in rows.items():
        a=d['args'];batches=math.ceil(a['examples']/a['batch']);calls=batches*(a['decode']+1)
        check(split+'/'+key+' static invariants',d['protected_sha256_before']==d['protected_sha256_after'] and d['scales_unchanged'] and d['all_prefill_and_decode_verified'] and not d['training'])
        check(split+'/'+key+' scope',d['ffn_layers']==(0 if a['ffn']=='original' else 24) and d['projection_count']==(48 if a['qcycles'] else 0) and d['attention_layers']==24)
        check(split+'/'+key+' attention calls',all(c['calls']==calls and c['prefill_calls']==batches and c['decode_calls']==batches*a['decode'] for c in d['attention_stats']))
        check(split+'/'+key+' FFN/projection calls',all(x['forward_calls']==calls for x in d['ffn_diagnostics'].values()) and all(x['calls']==calls for x in d['projection_diagnostics'].values()))
        check(split+'/'+key+' target count',d['scored_tokens']==a['examples']*a['decode'] and len(d['records'])==a['examples'])
        nll=sum(x['nll'] for x in d['records'])/d['scored_tokens']
        check(split+'/'+key+' perplexity',abs(math.exp(nll)-d['perplexity'])<1e-8)
        if a['attn']!='sdpa':
            check(split+'/'+key+' full prefill queries',all(c['head_queries']==a['examples']*(a['context']-1+a['decode'])*14 for c in d['attention_stats']))
        if split=='test':check(key+' test protocol',a['context']==2048 and a['decode']==128 and a['examples']==32 and a['batch']==4 and d['scored_tokens']==4096)
    hashes={tuple((r['offset'],r['ids_sha256']) for r in d['records']) for d in rows.values()}
    check(split+' matched tokens/positions',len(hashes)==1)
jobs=[read(p) for p in (R/'jobs').glob('*.json')];check('all jobs complete',all(j['status']=='complete' and j['returncode']==0 for j in jobs))
selection=read(R/'selection.json');q=selection['qcycles']
fv=V[name(spec('fixed',0,'dense'))]['perplexity']
candidates=[(qc,V[name(spec('fixed',qc,'ising'))]['perplexity']) for qc in (256,1024)]
eligible=[x for x in candidates if x[1]<=fv*1.05]
winner=min(eligible)[0] if eligible else min(candidates,key=lambda x:x[1])[0]
check('validation selection',winner==q and selection['test_used'] is False and selection['ising_within_incremental5pct']==bool(eligible))
groups={}
for d in T.values():
    a=d['args'];key=f"{a['ffn']}_q{a['qcycles']}_{a['attn']}"
    groups.setdefault(key,[]).append(d)
for g,rs in groups.items():rs.sort(key=lambda d:d['args']['seed'])
summary=[]
for g,rs in sorted(groups.items()):
    values=[d['perplexity'] for d in rs]
    summary.append({'group':g,'ppl_mean':statistics.mean(values),'ppl_std':statistics.pstdev(values),'seeds':[d['args']['seed'] for d in rs],
        'mean_nll':statistics.mean([sum(x['nll'] for x in d['records'])/d['scored_tokens'] for d in rs]),
        'prefill_seconds_mean':statistics.mean(d['prefill_seconds'] for d in rs),'decode_seconds_mean':statistics.mean(d['decode_seconds'] for d in rs),
        'unique_V_fraction':statistics.mean(c['unique_V_fraction'] for d in rs for c in d['attention_stats']) if rs[0]['args']['attn']!='sdpa' else None,
        'GQA_union_V_fraction':statistics.mean(c['GQA_union_V_fraction'] for d in rs for c in d['attention_stats']) if rs[0]['args']['attn']!='sdpa' else None,
        'ising_adjacent_repeat':statistics.mean(c['adjacent_repeat'] for d in rs for c in d['attention_stats']) if rs[0]['args']['attn']=='ising' else None})
by={r['group']:r for r in summary}
def paired(a,b):
    aa=groups[a];bb=groups[b];assert len(aa)==len(bb)==3
    av=[statistics.mean([x['records'][i]['nll']/128 for x in aa]) for i in range(32)]
    bv=[statistics.mean([x['records'][i]['nll']/128 for x in bb]) for i in range(32)]
    delta=[x-y for x,y in zip(av,bv)];rng=random.Random(20261009)
    boot=sorted(statistics.mean(rng.choices(delta,k=32)) for _ in range(5000))
    return {'a':a,'b':b,'mean_delta_nll':statistics.mean(delta),'ppl_ratio_change':math.exp(statistics.mean(delta))-1,
        'text_bootstrap95_relative_ppl':[math.exp(boot[125])-1,math.exp(boot[4874])-1],
        'note':'paired32 spans,seed-averaged NLL; descriptive text-bootstrap, not broad capability confidence'}
A=f'fixed_q{q}_ising';B=f'fixed_q{q}_iid';C='fixed_q0_iid';D='fixed_q0_dense';E=f'fixed_q{q}_dense'
pairs=[paired(a,b) for a,b in ((A,D),(B,D),(C,D),(A,B),(A,C),(E,D))]
gen=read(R/'generation.json');check('15 continuations',len(gen)==15)
diagnostics={}
for group,rs in groups.items():
    combined={}
    for signal in ('x','g','u'):
        items=[layer['signals'][signal] for d in rs for layer in d['ffn_diagnostics'].values()]
        if items:
            total=sum(x['count'] for x in items);out=sum(x['out_of_fullscale'] for x in items)
            se=sum(x['squared_error'] for x in items);energy=sum(x['signal_energy'] for x in items)
            combined[signal]={'out_of_fullscale_fraction':out/max(total,1),'quantization_nmse':se/max(energy,1e-30)}
    diagnostics[group]=combined
result={'groups':summary,'selected_qcycles':q,'validation_selection':selection,'paired_comparisons':pairs,
    'ffn_diagnostics_aggregate':diagnostics,'counts':{'validation':len(V),'test':len(T),'generation':len(gen),'audit':len(checks)},
    'full_corpus_ppl':False,'prefill_and_decode_both_replaced':True,'hardware_measured':False}
save(R/'summary.json',result);save(R/'audit.json',checks)
base=by['original_q0_sdpa']['ppl_mean']
labels={
'original_q0_sdpa':'原模型SDPA','original_q0_dense':'原模型手写dense',
'original_q0_ising':'仅Ising＋V采样，原FFN/投影',
'fixed_q0_sdpa':'当前FFN＋原attention',
D:'当前FFN＋手写准确attention',E:'当前FFN＋随机Q/K＋准确Softmax/PV',
'fixed_q0_ising':'当前FFN＋Ising，无随机Q/K',
A:'A：当前FFN＋随机Q/K＋Ising',
B:'B：当前FFN＋随机Q/K＋准确Softmax＋PV采样',
C:'C：当前FFN＋准确Q/K/Softmax＋PV采样'}
order=['original_q0_sdpa','original_q0_dense','original_q0_ising','fixed_q0_sdpa',D,E,'fixed_q0_ising',A,B,C]
md=[f'''# 全24层FFN＋Attention联合替换（2026-10-09）

## 本轮结论

全24层联合替换可以完成推理和续写，但相对原模型的精度损失仍明显。
当前FFN加准确attention为{by[D]['ppl_mean']:.4f}；加入随机Q/K与Ising后为{by[A]['ppl_mean']:.4f}，
较该FFN基线增加{(by[A]['ppl_mean']/by[D]['ppl_mean']-1)*100:.2f}%。验证集通过的5%新增误差预算，在本轮test没有保持。
仅把Ising换回准确Softmax、保留随机Q/K和PV采样，得到{by[B]['ppl_mean']:.4f}，未见明显改善。
恢复准确Q/K与Softmax、只在PV采样则为{by[C]['ppl_mean']:.4f}，较FFN基线增加约{(by[C]['ppl_mean']/by[D]['ppl_mean']-1)*100:.2f}%，是本轮更稳妥的组合。

随机Q/K但Softmax/PV都准确的对照为{by[E]['ppl_mean']:.4f}，已增加{(by[E]['ppl_mean']/by[D]['ppl_mean']-1)*100:.2f}%，
提示当前组合新增损失主要来自随机Q/K投影这一组，而非Ising与IID采样方式的差别。
这包括投影量化、采样及其与FFN的相互作用，不是单独测出的纯采样误差。
Ising与IID的成对差异区间跨0，不能认定其中一个精度更好；仅PV采样相对准确attention的区间也接近并跨0。

建议暂以“当前固定尺度FFN＋准确Q/K/Softmax＋PV采样”为集成基线，Ising保留为研究分支。
FFN+Ising而Q/K准确的单seed为{by['fixed_q0_ising']['ppl_mean']:.4f}，值得后续复测，但尚不能替代三seed结论。
若继续扩展随机Q/K，先在独立验证集扩大流长或减少替换投影，再做新测试；本轮不按test调参。
主要组合仍比原模型高约20%—27% PPL，不能称为无损替换。

## 当前版本与范围

FFN沿用最新固定尺度版：全部24层，W8，12位幅度+符号，group_max离线固定尺度，
输入4096次、输出3072次采样，每256次更新，25%预热；保留理想sigmoid和FP32中间估计。
这不是旧训练AND版本，也不是需要在线最大值的动态尺度高精度版本。
残差、RMSNorm与BF16接口保持原模型路径，没有训练。

随机Q/K指固定q_proj/k_proj投影：W10、9位幅度+符号、固定二次幂尺度，选择T={q}。
QK点积仍确定性，V/O投影不变。Ising为理想强互斥连续时间热浴，
预热16、读取间隔4、512次观测；准确Softmax回退也采样512个类别位置。
Ising直接从score差的sigmoid速率生成驻留轨迹，并按固定时间观测，没有用softmax概率生成其输出。
这不是有限惩罚、有限延迟硬件。时间单位也不能与Q/K采样周期直接相加。

**预填充与解码都实际启用替换，覆盖24层。** KV缓存来自替换后的前缀。
这比以往只在decode替换attention、保持精确prefill的协议更严格。

## 匹配测试结果

测试为WT2 test的32段相同文本，每段context2048、128步teacher forcing，
合计4096计分token。不是完整语料滑窗PPL，不能与11.652735或13.873066直接横比。
本轮原模型对应PPL为{base:.6f}。

| 方案 | seed数 | PPL均值±总体标准差 | 相对原模型 | 相对当前FFN＋准确dense |
|---|---:|---:|---:|---:|
''']
for g in order:
    r=by[g];md.append(f"| {labels[g]} | {len(r['seeds'])} | {r['ppl_mean']:.6f} ± {r['ppl_std']:.6f} | {(r['ppl_mean']/base-1)*100:+.2f}% | {(r['ppl_mean']/by[D]['ppl_mean']-1)*100:+.2f}% |")
md.append(f'''
A/B/C三种主组合和FFN基线均复测三个seed。原FFN的Ising及固定FFN无随机Q/K的Ising是单seed诊断。
固定FFN+SDPA与固定FFN+dense均保留三seed，避免将原attention软件路径差别与FFN随机波动混淆。

验证集在train尾部65536token上8段，context512/decode64，共512计分。
FFN+dense验证PPL={fv:.6f}，联合Ising T256/T1024为{candidates[0][1]:.6f}/{candidates[1][1]:.6f}。
预设规则以FFN+dense新增误差5%为探索预算，优先较短T；本轮选择{q}。
回退B/C都按预定对照执行，没有用test选择流长。5%不是一般能力可接受保证，
总退化需同时看原模型基线；小验证集的轻微反向波动不构成架构更优证据。

## 成对文本差异

以下先对每段的三个seed平均NLL，再对32段做5000次成对bootstrap。
区间只描述本次文本抽样的不确定性，不包含所有模型能力/器件误差，不能当作全面显著性证明。
正数表示左侧PPL更高；区间跨0时，不应声称左/右方案确定更好。

| 对比 | 几何PPL相对变化 | 文本bootstrap95%区间 |
|---|---:|---:|
''')
for p in pairs:
    lo,hi=p['text_bootstrap95_relative_ppl']
    md.append(f"| {labels[p['a']]} 对 {labels[p['b']]} | {p['ppl_ratio_change']*100:+.2f}% | [{lo*100:+.2f}%, {hi*100:+.2f}%] |")
md.append('''
## 逻辑访存和采样统计

每层/seed等权统计，分母仅包括因果可见位置。数值是非零采样计数对应的地址，
不是实测DRAM事务或物理KV缓存容量节省。GQA每7个Q头共享1个KV头，地址并集更大。
软件依旧生成完整QK，并使用dense counts@V模拟选择累加，未实现真实稀疏加速。
''')
md.append('| 方案 | 单头V地址比例 | GQA组并集比例 | Ising相邻样本重复率 |\n|---|---:|---:|---:|')
for g in ['original_q0_ising','fixed_q0_ising',A,B,C]:
    r=by[g];rep='—' if r['ising_adjacent_repeat'] is None else f"{r['ising_adjacent_repeat']:.2%}"
    md.append(f"| {labels[g]} | {r['unique_V_fraction']:.2%} | {r['GQA_union_V_fraction']:.2%} | {rep} |")
md.append('\n## GPU参考实现耗时\n\n各seed均值，单位秒；每次处理相同32段前缀和4096个解码计分token。包含统计和诊断开销，不能换算成p-bit电路速度。\n\n| 方案 | prefill | decode |\n|---|---:|---:|')
for g in ['original_q0_sdpa','fixed_q0_sdpa',D,E,A,B,C]:
    r=by[g]
    md.append(f"| {labels[g]} | {r['prefill_seconds_mean']:.2f} | {r['decode_seconds_mean']:.2f} |")
md.append('''
## 精度解释与局限

先比较原模型与当前FFN，再比较FFN+准确attention与A/B/C。
如果联合退化主要已经存在于FFN-only中，换回准确Softmax不能消除FFN的有限流长噪声。
Q/K随机化、概率采样和FFN会相互影响，PPL增量不应按独立误差简单相加。
即使某采样方案单seed低于准确attention，也可能只是不同随机轨迹，不代表采样优于准确算子。

固定尺度来自旧原模型校准，未针对本组合重新训练或重新校准。
FFN越界与局部量化NMSE汇总保存在summary.json；局部NMSE不是相对原模型的端到端误差。
宽累加、sigmoid、Ising速率和驻留时间均为软件参考；未测有限整数溢出、
tanh驱动/偏置电路、随机相关性、耦合与广播延迟、面积或能耗。
GPU耗时属于仿真，不是候选硬件延迟；4096/256/512也不是可直接相加的同类时钟。

## 验证和审计

''')
md.append(f'''{len(V)}项验证、{len(T)}项测试、15条续写完成。3项采样单元测试和{len(checks)}条源码/协议/结果审计通过。
所有24层的prefill/decode调用、FFN与投影数量、固定尺度、其他参数哈希均核对。
手写dense对原SDPA整段logit平均/最大差为{integration['dense_mean_abs_error']:.6f}/{integration['dense_max_abs_error']:.6f}；
同一cached decode路径为{integration['cached_manual_vs_sdpa_mean']:.6f}/{integration['cached_manual_vs_sdpa_max']:.6f}。
因果前缀不受未来token变更影响的检查通过。BF16整段与逐token路径不是逐位相等：
原模型平均差{integration['original_cache_mean_abs_error']:.6f}，手写图{integration['cache_mean_abs_error']:.6f}。
首次过紧的跨路径检查失败后，加入原模型同路径对照；正式评测前已重新通过所有检查。

源代码与校准SHA256、作业参数、每段NLL和token哈希均保留。
发布目录：experiments/joint_ffn_attention_20261009
此发布包包含三个联合方案、匹配基线和完整审计记录。

## 全部固定提示续写

Base模型，greedy，最多48新token；仅作定性检查，不是通用任务评测。
''')
for g in gen:md.append(f"### {name(g['spec'])} / {g['prompt']}\n\n```text\n{g['continuation']}\n```\n")
text='\n'.join(md).rstrip()+'\n'
while '|\n\n|' in text:text=text.replace('|\n\n|','|\n|')
(R/'RESULTS_ZH.md').write_text(text,encoding='utf-8')
print(json.dumps({'audit':len(checks),'validation':len(V),'test':len(T),'A':by[A],'B':by[B],'C':by[C]},ensure_ascii=False))
