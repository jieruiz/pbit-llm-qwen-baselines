"""Audit archived QK experiments, then generate comparison JSON/report/plot."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path('results/pbit_qk_20261006')
EXP=Path('experiments/pbit_qk')


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    for name in ('status.json','extension_status.json'):
        assert read(ROOT/name)['stage']=='complete', name
    for name in ('protocol.json','extension_protocol.json'):
        for filename,digest in read(ROOT/name)['source_sha256'].items():
            assert hashlib.sha256((EXP/filename).read_bytes()).hexdigest()==digest, filename
    for name in ('test_sampling.json','test_integration.json'):
        assert read(ROOT/name)['status']=='passed', name
    records=[read(p) for p in sorted((ROOT/'evaluation').glob('*.json'))]
    assert len(records)==70, len(records)
    assert len({r['mlp_sha256'] for r in records})==1
    assert len({r['text_sha256'] for r in records})==1
    old=read(Path('results/pbit_attention_20261003/comparison.json'))
    assert records[0]['mlp_sha256']==old['original_ffn_sha256']
    for r in records:
        a=r['args']; ctx=a['context']
        assert r['attention_layers']==24 and a['decode']==128
        assert r['value_aggregation']=='dense continuous PV; no value sampling'
        assert (a['examples'],a['batch'],r['scored_tokens'])==((32,16,4096) if ctx==2048 else (16,4,2048))
        assert math.isfinite(r['perplexity'])
        assert math.isclose(r['mean_nll'],sum(e['nll_sum'] for e in r['examples'])/r['scored_tokens'],rel_tol=1e-10)
        assert math.isclose(math.exp(r['mean_nll']),r['perplexity'],rel_tol=1e-10)
        for filename,digest in r['source_sha256'].items():
            assert hashlib.sha256((EXP/filename).read_bytes()).hexdigest()==digest,filename
    rows=[]; baselines={}
    metrics=['unique_key_feature_fraction_per_head','unique_key_feature_fraction_GQA_group_union',
             'mean_active_bit_fraction_per_round','signed_add_events_relative_to_dense_mac_terms',
             'score_relative_l2_global','attention_KL_exact_to_sampled','attention_total_variation',
             'attention_argmax_agreement']
    for ctx in (2048,8192):
        group=[r for r in records if r['args']['context']==ctx]
        positions=[[e['first_target_token'] for e in r['examples']] for r in group]
        assert all(p==positions[0] for p in positions)
        previous=read(Path(f'results/pbit_attention_20261003/evaluation/ctx{ctx}_sdpa_s128_seed0.json'))
        assert positions[0]==[e['first_target_token'] for e in previous['examples']]
        assert records[0]['text_sha256']==previous['text_sha256']
        baseline={r['args']['mode']:r['perplexity'] for r in group if r['args']['mode'] in ('sdpa','dense')}
        for mode in ('sdpa','dense'): assert math.isclose(baseline[mode],old['baselines'][str(ctx)][mode],rel_tol=1e-9)
        baselines[str(ctx)]=baseline
        for mode in ('iid','stratified'):
            for B in ((2,4,8,16) if mode=='iid' else (2,4,8,16,32,64,128)):
                selected=[r for r in group if r['args']['mode']==mode and r['args']['samples']==B]
                assert sorted(r['args']['seed'] for r in selected)==[0,1,2]
                values=[r['perplexity'] for r in selected]
                row={'context':ctx,'mode':mode,'B':B,'ppl_mean':statistics.mean(values),
                     'ppl_std':statistics.pstdev(values),'ppl_values':values}
                row['ppl_increase_percent']=100*(row['ppl_mean']/baseline['sdpa']-1)
                for metric in metrics:
                    value=statistics.mean(r['sampling_stats'][metric] for r in selected)
                    assert math.isfinite(value)
                    row[metric]=value
                rows.append(row)
    result={'evaluations':len(records),'baselines':baselines,'rows':rows,
            'audit':'All evaluated source hashes, FFN/corpus identities, matched targets, and prior dense baselines verified.',
            'protocol':read(ROOT/'protocol.json'),'extension_protocol':read(ROOT/'extension_protocol.json')}
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Query侧p-bit编码：QK注意力实验（2026-10-06）','',
        '## 实现与范围','',
        'Qwen2.5-0.5B Base，24层全部替换decode-time QK。原始FFN、连续PV、Q/K/V投影、RoPE及精确prefill保留；无训练。',
        '在RoPE之后，对每头q取a=max(abs(q))；每维生成b~Bernoulli(abs(q)/a)，保留sign(q)。',
        'B轮读出为q_hat=sign(q)*a*count/B，再计算q_hat @ K^T / sqrt(d)，经过原始Softmax和连续PV。',
        'K与V保持连续。未实现组平均query补偿，也没有叠加随机PV或FFN。全零query编码为零。','',
        '- `iid`：count~Binomial(B,p)，与B轮独立p-bit计数同分布。',
        '- `stratified`：将[0,1)分为B个区间，各区间独立取一点并与p比较；计数精确等价于floor(Bp)+Bernoulli(frac(Bp))。',
        '- 这里压缩的是计数的概率分布，不模拟B轮器件的实际耗时。当前GPU仍执行密集浮点矩阵乘法。','',
        '## 协议与验证','',
        '2k：32个互不重叠WT2 test片段，精确prefix2047后128步缓存teacher-forcing，共4096个计分token。',
        '8k：16段，精确prefix8191后128步，共2048个计分token。当前随机网络产生的KV缓存持续参与后续步。',
        '每个随机配置3个推理seed；±为总体标准差，不是跨文本置信区间。2k和8k只在各自基线内比较。',
        '首批52项：两上下文、两种编码、B=2/4/8/16、三seed，加四个密集基线。',
        '看到小B性能明显下降后，追加分层B=32/64/128三seed共18项；全部共70项。此追加由测试结果驱动，不是新独立测试集。',
        '本轮不是完整WT2滑窗PPL，也未测试长时间自由生成及任务准确率；不能与历史完整WT2的11.652735直接比较。',
        '60000次概率测试验证均值、解析方差、字面采样计数分布、零query和边界；缓存集成测试验证精确prefill、SDPA委托、随机种子复现和FFN对象保留。',
        '70份结果的源码哈希、FFN/文本哈希和计分位置已核对。两个密集基线逐值复现先前PV实验，便于横向比较。','',
        '## PPL与逻辑Key特征访问','']
    for ctx in (2048,8192):
        lines += [f'### 起始上下文{ctx}', '',f"原始SDPA：{baselines[str(ctx)]['sdpa']:.6f}；手写FP32密集参考：{baselines[str(ctx)]['dense']:.6f}。",'',
            '| 编码 | B | PPL均值 ± 标准差 | PPL增加 | 单头Key访问 | GQA组Key访问 | 加减事件/原始MAC项数 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:
            if r['context']!=ctx: continue
            lines.append(f"| {r['mode']} | {r['B']} | {r['ppl_mean']:.6f} ± {r['ppl_std']:.6f} | {r['ppl_increase_percent']:.2f}% | {100*r['unique_key_feature_fraction_per_head']:.3f}% | {100*r['unique_key_feature_fraction_GQA_group_union']:.5f}% | {r['signed_add_events_relative_to_dense_mac_terms']:.3f} |")
        lines += ['']
    lines += ['## 如何解读','',
        '分层采样明显改善精度，但足够准确时GQA组特征并集接近全量，Key读取收益很小。',
        'Qwen每组7个查询头共享Key，每头有64个特征；访问比例按所有B轮和组内7头取并集。',
        '加减事件比例按字面执行B轮符号位乘Key计数，不等于速度比：加法和乘加成本不同，合并计数也会改变实现。',
        '所有访问比例是逻辑特征数，不是实际DRAM事务或实测加速。当前实现仍密集访问Key，并为诊断额外计算精确QK。',
        '分数相对L2、Softmax分布KL/TV在同一当前Q/K状态下比较，详见comparison.json；不是与另一条精确模型轨迹作逐层比较。',
        '线性QK估计无偏不保证Softmax概率无偏；本实验支持分数噪声改变注意力分布，但不定位具体敏感层或唯一因果机制。',
        '本轮未实现论文组共享补偿方案，也未模拟真实p-bit驱动、器件相关性、寻址和能耗。不能将这些结果解释为硬件不可行或其他编码都无效。','',
        '![Quality and logical key access](quality_access.png)','',
        '源码：[experiments/pbit_qk](../../experiments/pbit_qk/README.md)。原始结果在evaluation/，运行命令在commands/。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    for col,ctx in enumerate((2048,8192)):
        for mode in ('iid','stratified'):
            selected=[r for r in rows if r['context']==ctx and r['mode']==mode]
            x=[r['B'] for r in selected]
            axes[0,col].plot(x,[r['ppl_mean']/baselines[str(ctx)]['sdpa'] for r in selected],'o-',label=mode)
            axes[1,col].plot(x,[100*r['unique_key_feature_fraction_GQA_group_union'] for r in selected],'o-',label=mode)
        axes[0,col].axhline(1,color='gray',linestyle='--',label='original SDPA')
        axes[0,col].set(yscale='log',ylabel='PPL / original PPL',title=f'{ctx}-token initial context')
        axes[1,col].set(ylabel='Logical GQA key features read (%)',ylim=(0,103))
        for ax in axes[:,col]:
            ax.set_xscale('log',base=2); ax.set_xlabel('B (query sampling rounds)')
            ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Query-only p-bit attention: quality vs logical key access\nOriginal FFNs and dense PV; 3 inference seeds; no sparse GPU kernel',fontsize=12)
    fig.savefig(ROOT/'quality_access.png',dpi=170)
    plt.close(fig)
    print(json.dumps({'audit':'passed','evaluations':len(records),'baselines':baselines,'rows':rows},indent=2))


if __name__=='__main__': main()
