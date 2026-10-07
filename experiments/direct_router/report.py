"""Aggregate saved results without choosing parameters on test data."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='results/direct_router_20261007');args=ap.parse_args()
    root=Path(args.results);groups=collections.defaultdict(list);all_data=[]
    for phase in ('evaluation','calibrated'):
        for path in sorted((root/phase).glob('*.json')):
            d=json.loads(path.read_text())
            if 'args' not in d:continue
            a=d['args'];key=(phase,a['context'],a['mode'],a['reps'],a['ratio'],a['backend'])
            groups[key].append(d);all_data.append(d)
    assert len(all_data)==66,len(all_data)
    assert len({d['mlp_sha256'] for d in all_data})==1
    assert len({d['text_sha256'] for d in all_data})==1
    assert json.loads((root/'kernel_checks.json').read_text())['status']=='passed'
    rows=[]
    for key,ds in sorted(groups.items()):
        phase,ctx,mode,reps,ratio,backend=key
        ppl=[d['perplexity'] for d in ds]
        row=dict(phase=phase,context=ctx,mode=mode,reps=reps,budget=ratio,backend=backend,seeds=len(ds),ppl_mean=statistics.mean(ppl),ppl_seed_sd=statistics.pstdev(ppl))
        for stat in ('selected_token_fraction','GQA_union_fraction','total_QK_PV_selector_MAC_ratio'):
            row[stat]=statistics.mean(d['sampling_stats'].get(stat,1.) for d in ds)
        # Approximate whole-model linear MACs, including untied computation of tied LM head.
        # Mean attention length is context + (128-1)/2 during this decode protocol.
        length=ctx+63.5
        attention=24*2*14*64*length
        fixed=24*(3*896*4864+896*(896+2*128)+896*896)+896*151936
        row['approx_whole_model_linear_MAC_saving']=attention*(1-row['total_QK_PV_selector_MAC_ratio'])/(fixed+attention)
        rows.append(row)
    (root/'summary.json').write_text(json.dumps({'runs':66,'all_original_FFN_hashes_equal':True,'results':rows},indent=2)+'\n')
    def get(ctx,mode,reps=1,phase='evaluation',ratio=.5):
        return next(r for r in rows if r['context']==ctx and r['mode']==mode and r['reps']==reps and r['phase']==phase and r['budget']==ratio and r['backend']=='sparse')
    lines=['# 直接驱动 p-bit 的块保留概率预测器：实验记录（2026-10-07）','',
        '结论：直接 sigmoid/Bernoulli 筛选在保守校准后可保持较低 PPL，并真实跳过大量 QK/PV 计算；当前 GPU 实现较 SDPA 慢。新预测器尚未证明优于旧的块均值概率筛选。','',
        '## 1. 实现范围','',
        'Qwen2.5-0.5B Base，24 层的解码注意力均启用筛选；FFN 全部保留原实现和权重。Prefill 使用原始 SDPA。每块 64 个 token，缓存 1 个块均值或 4 个子块均值；查询仅与摘要做投影。小预测器直接给出 p-bit 输入，独立采样 0/1 保留掩码。强制保留首块和最近两块。选中后才读取这些块的 K/V，并计算连续 QK、候选内精确 softmax、连续 PV。','',
        '这里的 p-bit 是软件 Bernoulli 模拟。筛选阶段无需块 softmax、Ising 耦合或退火；候选 token 内的 softmax 仍然存在。完整 KV cache 也仍然保留，不能把少读候选块说成已减少 KV 存储容量。','',
        '## 2. 训练和评估','',
        '训练只用 WT2 train 的 32 个 8k 窗口，末尾另留 8 个 8k 窗口验证；每窗抽 1k 至 8k 的 8 个查询。训练每个层/头的预测器，骨干不变。监督目标是完整教师注意力的块质量对应的 Bernoulli 包含概率。单均值线性版 2,688 参数，四摘要小 MLP 64,848 参数。checkpoint 由验证集 BCE 决定。','',
        '第一阶段 54 组实验失败后，增加第二阶段的保守校准：仅利用验证集，逐层逐头提高输入偏置，使该头平均期望保留的教师注意力质量至少达到 95%。这不是每个查询的保证，第二阶段属于探索性后续实验。额外执行 12 组测试。','',
        '2k：32×128=4,096 个评分 token；8k：16×128=2,048 个评分 token。每个随机配置 3 个种子，表中 ± 为种子间总体标准差，不是数据集置信区间。这是 WT2 测试集抽样后缀 PPL，不是全语料 PPL。','',
        '## 3. 主要结果（预算输入 0.5）','',
        '| 方法 | 2k PPL | 2k QK/PV+筛选 MAC 占比 | 8k PPL | 8k QK/PV+筛选 MAC 占比 |',
        '|---|---:|---:|---:|---:|']
    specs=[('原始 Qwen','sdpa',1,'evaluation'),('旧块均值概率筛选','mean',1,'evaluation'),('直接线性预测，未校准','learned',1,'evaluation'),('直接四摘要 MLP，未校准','learned',4,'evaluation'),('线性预测，95% 校准','learned',1,'calibrated'),('四摘要 MLP，95% 校准','learned',4,'calibrated')]
    for title,mode,reps,phase in specs:
        a,b=get(2048,mode,reps,phase),get(8192,mode,reps,phase)
        lines.append(f"| {title} | {a['ppl_mean']:.4f} ± {a['ppl_seed_sd']:.4f} | {a['total_QK_PV_selector_MAC_ratio']:.2%} | {b['ppl_mean']:.4f} ± {b['ppl_seed_sd']:.4f} | {b['total_QK_PV_selector_MAC_ratio']:.2%} |")
    a,b=get(2048,'learned',4,'calibrated'),get(8192,'learned',4,'calibrated')
    lines += ['',f"校准四摘要版实际保留 token 比例为 2k {a['selected_token_fraction']:.2%}、8k {b['selected_token_fraction']:.2%}；计入预测器 MAC 后，局部计算量降低约 {1-a['total_QK_PV_selector_MAC_ratio']:.1%}、{1-b['total_QK_PV_selector_MAC_ratio']:.1%}。",'',
        'MAC 口径仅包含 QK、PV 和预测器线性乘加；不含摘要构建/维护、归约、激活、随机数、访存和调度，也不含保持不变的 Q/K/V/O 投影、FFN 和输出头。', '',
        f"按模型尺寸估算，若把这些不变矩阵和完整词表输出头也算进来，整模型解码线性 MAC 约减少 {a['approx_whole_model_linear_MAC_saving']:.1%}（2k）、{b['approx_whole_model_linear_MAC_saving']:.1%}（8k）。这仍不是完整 FLOPs、速度或能耗测量。",'',
        '## 4. 为什么第一版不理想','',
        '预测器以很少的块摘要估计完整教师的注意力质量。均值摘要丢失了块内极值、细粒度 Key 分布；有限训练查询也不能覆盖全部语境。BCE 小并不保证那些稀少但关键的块不漏选，24 层连续筛选进一步放大误差。', '',
        '验证集 8k、预算 0.5 时，旧均值方法期望保留约 88.07% 教师注意力质量，四摘要直接预测只有约 85.08%，oracle 包含概率可达约 88.39%。保守偏置明显恢复 PPL，说明过于激进的筛选和覆盖不足是本轮重要问题；这不足以证明它们是唯一原因。', '',
        '四摘要校准版 PPL 略好于旧均值版本，但保留了更多 token、用了更多 MAC。当前并未证明精度—计算量前沿更优。下一步更值得尝试输出误差/语言模型损失蒸馏、端到端校准和查询相关的保守回退；不是直接增加 p-bit 采样次数。','',
        '## 5. 实测运行时间','',
        'RTX 5090，预热后串行 CUDA event 计时，真实第 12 层 Q/K/V；以下为四摘要校准版，计时包含少量用于重复同一步的摘要复原操作。单层/单次解码 attention 子路径，不是完整模型每 token 延迟。','',
        '| 上下文 / batch | Qwen repeat_kv + SDPA | 仅稀疏核 | 筛选器 + 稀疏核 |','|---|---:|---:|---:|']
    bench=json.loads((root/'benchmark_calibrated.json').read_text())
    for r in bench['results']:
        if r['reps']==4:
            lines.append(f"| {r['context']} / {r['batch']} | {r['dense_SDPA_ms']:.3f} ms | {r['sparse_kernel_ms']:.3f} ms | {r['selector_plus_sparse_with_reset_ms']:.3f} ms |")
    lines += ['', '当前筛选器有大量小 PyTorch 算子调用，稀疏核又按头逐块循环。虽然实际跳过了未选 K/V 的载入及 QK/PV 算术，这些调度和串行访问开销使总耗时增加。进一步工作应融合筛选算子、优化并行稀疏核；不能从本实验推断 p-bit 硬件速度或能耗。','',
        '## 6. 全局广播与稠密 Ising 耦合','',
        '如果每条互抑制连接强度相同为 g，令 S=Σb_j，则局部驱动 h_i−gΣ(j≠i)b_j = h_i−gS+g b_i。精确全局求和、广播以及自身项补偿可代数精确替代同权稠密连接。配合同样异步更新顺序和随机数，状态轨迹也一致；使用同样更新动力学时，转移核和稳态分布一致。','',
        '前提不适用于任意不同权重的稠密 J。广播延迟、求和误差或多个节点同时依据旧 S 更新，会改变实际动力学。广播也不消除有限惩罚的多热点误差或混合时间。当前块预测器采用独立 p-bit，无需此耦合网络。','',
        '## 7. 保存与复现','',
        '- 开发分支：`pbit-direct-router`；代码与结果发布到注意力分支 `pbit-attention-av-baseline`。',
        '- 远程目录：`/home/Hongjie_Zeng/pbit_direct_router_20261007`。',
        '- 源码：`experiments/direct_router/`；完整 66 组结果：本目录 `evaluation/`、`calibrated/`。',
        '- 权重本地备份：`artifacts/direct_router_20261007/`；训练用教师张量留在远程。',
        '- `summary.json`、`data_protocol.json`、`training_r*.json`、`heldout_coverage.json`、`calibration.json`、`kernel_checks.json` 和两份 benchmark JSON 保存协议、训练和数值证据。','']
    (root/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('experiments/direct_router').glob('*.py')}
    for p in Path('artifacts/direct_router_20261007').glob('*.pt'):manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'local_source_and_weights_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'runs':len(all_data),'groups':len(rows),'report':str(root/'RESULTS_ZH.md')}))


if __name__=='__main__':main()
