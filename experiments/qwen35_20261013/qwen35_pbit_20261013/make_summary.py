"""Render actual completed result snapshot; never promote partial scores to final."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
stopped=(ROOT/'task_evaluation_stopped.json').exists()
snapshot=json.loads((ROOT/'status_snapshot.json').read_text(encoding='utf-8-sig'))
original=json.loads((ROOT.parent/'qwen35_baseline_20261013/baseline_status.json').read_text(encoding='utf-8-sig'))
base=next(r['perplexity'] for r in original['completed'] if r['task']=='wikitext2')
labels={'ffn_mean':'定点编码＋INT8 权重，不采样','attention_dense_control':'FP32 注意力路径，不采样',
    'ffn_exact':'24 层随机 FFN，准确计数','attention_pv':'6 层完整注意力 PV512 采样',
    'joint_pv':'24 层随机 FFN＋6 层 PV512'}
lines=['# Qwen3.5-0.8B 替换实验：已完成结果','',
    '快照 UTC 时间：'+snapshot['timestamp_utc']+'（北京时间 UTC+8）。', '',
    '以下仅汇总已完成的正式 JSON；模型后训练版、原始 DeltaNet 算子、同一软件环境和数据。', '',
    '| 方案 | 完整 WT2 PPL | 相对原始 PPL 变化 | 完成种子数 |', '|---|---:|---:|---:|',
    f'| 原始模型 | {base:.6f} | — | 1/1 |']
for variant,label in labels.items():
    rows=[r for r in snapshot['completed'] if r['variant']==variant and r['task']=='wikitext2']
    expected=1 if variant in ['ffn_mean','attention_dense_control'] else 3
    if not rows:
        lines.append(f'| {label} | 尚未完成 | — | 0/{expected} |');continue
    ps=[r['perplexity'] for r in rows];avg=statistics.mean(ps)
    text=f'{avg:.6f}'+(f' ± {statistics.pstdev(ps):.6f}' if len(ps)>1 else '')
    if len(ps)<expected:text+='（未齐）'
    lines.append(f'| {label} | {text} | {(avg/base-1)*100:+.3f}% | {len(ps)}/{expected} |')
lines+=['','± 为随机种子间总体标准差，不是语料统计置信区间。上下文 2048，步长 1024，297192 个预测 token 恰好计分一次。', '',
    'FFN 为分阶段 fixed_exact：入口 4096 次、产品/down 另 4096 次。权重 INT8、入口及内部幅度编码 12 位、每 64 通道训练集离线固定尺度；软件用精确计数分布压缩，不展开位流。', '',
    'PV 在每个 query/head 带放回抽 512 次；Q/K、Softmax、位置编码、缓存和注意力输出门控保留。六层完整注意力采样包含前缀预填充及解码。18 层 DeltaNet 不替换。', '',
    '## 五项全量零样本任务', '',
    '| 方案 | HellaSwag acc_norm | PIQA acc | WinoGrande acc | ARC-E acc | ARC-C acc_norm |',
    '|---|---:|---:|---:|---:|---:|']
metric_keys=[('hellaswag','acc_norm,none'),('piqa','acc,none'),('winogrande','acc,none'),('arc_easy','acc,none'),('arc_challenge','acc_norm,none')]
for variant,label in [('original','原始模型')]+[(v,labels[v]) for v in ['ffn_exact','attention_pv','joint_pv']]:
    source=original['completed'] if variant=='original' else [r for r in snapshot['completed'] if r['variant']==variant and r['seed']==0]
    values=[]
    for task,key in metric_keys:
        r=next((r for r in source if r['task']==task),None)
        values.append(f"{r['metrics'][task][key]*100:.2f}%" if r else ('未测（用户停止）' if stopped else '运行／排队中'))
    lines.append('| '+' | '.join([label]+values)+' |')
lines+=['',('用户要求本阶段只使用 PPL，任务阵列 1241 已取消，未完成的基础任务不继续测量。' if stopped else '随机方案任务成绩目前为 seed0；正式文件保留 acc、acc_norm 和标准误等全部输出。未完成处不填小样本成绩。'), '',
    '## 查看最新进度', '',
    '远程目录 `/home/Hongjie_Zeng/qwen35_pbit_20261013/`，PPL 阵列 1236 已完成；基础任务阵列 1241 '+('已取消。' if stopped else '继续运行。'), '',
    '运行只读 collect_status.py 可获取最新完成结果；本报告是上述时间点的快照，不会自行更新。完整方案见 README_ZH.md。']
(ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:19]))
