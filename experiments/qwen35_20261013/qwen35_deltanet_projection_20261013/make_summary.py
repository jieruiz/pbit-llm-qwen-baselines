import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
x=json.loads((ROOT/'status_snapshot.json').read_text(encoding='utf-8-sig'))
base=json.loads((ROOT.parent/'qwen35_dynamic_ffn_20261013/status_snapshot.json').read_text(encoding='utf-8-sig'))
joint=[r['perplexity'] for r in base['completed'] if r['args']['variant']=='dynamic_joint']
assert len(joint)==3
baseline={'solo':15.661097804818592,'combined':statistics.mean(joint)}
labels={'out':'输出投影','out_z':'输出＋z投影','out_z_qkv':'输出＋z＋QKV投影'}
lines=['# DeltaNet 固定投影逐步随机化：正式 PPL 结果','',
    'UTC 快照时间：'+x['timestamp_utc']+'；北京时间 UTC+8。仅填已完成的完整WT2结果。', '',
    '两组独立对照：solo保留原始FFN/完整注意力；combined采用已验证的动态随机FFN＋PV512。', '',
    f"原始模型基线：{baseline['solo']:.6f}；动态FFN＋PV基线（三种子均值）：{baseline['combined']:.6f}。",'',
    '| DeltaNet范围 | 仅INT8权重，无采样（solo） | 随机投影solo PPL | 完成种子 | 联合方案PPL | 完成种子 |',
    '|---|---:|---:|---:|---:|---:|']
for scope,label in labels.items():
    control=[r['perplexity'] for r in x['completed'] if r['args']['scope']==scope and r['args']['mode']=='weights' and r['args']['context']==2048]
    values=[]
    for scenario in ['solo','combined']:
        rows=[r for r in x['completed'] if r['args']['scope']==scope and r['args']['mode']=='iid' and r['args']['scenario']==scenario and r['args']['context']==2048]
        ps=[r['perplexity'] for r in rows]
        if ps:
            avg=statistics.mean(ps);text=f'{avg:.6f}'
            if len(ps)>1:text+=f' ± {statistics.pstdev(ps):.6f}'
            text+=f'（较本组基线{(avg/baseline[scenario]-1)*100:+.3f}%）'
            if len(ps)<3:text+='，未齐'
        else:text='运行／排队中'
        values += [text,f'{len(ps)}/3']
    lines.append('| '+' | '.join([label,f'{control[0]:.6f}' if control else '未完成']+values)+' |')
lines+=['','±为三个种子间总体标准差；未齐不作为最终均值。context2048/stride1024，297192预测token恰好计分一次。', '',
    '三个范围均覆盖全部18层DeltaNet：18/36/54个投影模块。固定INT8权重、动态group64概率、每投影4096采样、准确计数压缩；状态核心、alpha/beta、卷积及归一化原代码保留。', '',
    '最大范围的QKV/z共享同一入口计数流；它与z-only阶段的独立入口流随机相关性不同，因此范围增量同时包含这一硬件共享选择。', '',
    '## 8K长度对照', '',
    '| 范围 | PPL | 完成种子数 |', '|---|---:|---:|']
for scope,label in [('baseline','原始模型，8K'),('out_z_qkv','完整投影范围solo，8K')]:
    ps=[r['perplexity'] for r in x['completed'] if r['args']['context']==8192 and r['args']['scope']==scope]
    text=f'{statistics.mean(ps):.6f}' if ps else '运行／排队中'
    if len(ps)>1:text+=f' ± {statistics.pstdev(ps):.6f}'
    lines.append(f'| {label} | {text} | {len(ps)}/{1 if scope=="baseline" else 3} |')
lines+=['','8K采用context8192/stride4096、同一完整语料。8K结果只能与8K原始基线比较，不能用2K基线直接判断变化。它仍是滑窗PPL，每窗重置状态，不能证明无限长序列稳定。', '',
    '任务：逐步阵列1257（solo/combined各一张卡），8K检查1259；只测PPL，不运行五项基础任务。详细说明见README_ZH.md。']
(ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:17]))
