"""Audit the predeclared factorial experiment and report paired NLL changes."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path('results/pbit_joint_20261006')


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'evaluation').glob('*.json'))]
    assert len(records)==len(protocol['jobs'])==48
    assert json.loads((ROOT/'status.json').read_text())['stage']=='complete'
    assert json.loads((ROOT/'integration.json').read_text())['status']=='passed'
    smoke=json.loads((ROOT/'smoke_all20_n16.json').read_text())
    assert len(smoke['checkpoints'])==20 and math.isfinite(smoke['perplexity'])
    for source,digest in protocol['source_sha256'].items():
        archived=ROOT/'executed_sources'/source
        assert hashlib.sha256(archived.read_bytes()).hexdigest()==digest,source
        assert Path(source).read_bytes().replace(b'\r\n',b'\n')==archived.read_bytes().replace(b'\r\n',b'\n'),source
    expected=json.loads(Path('results/multithreshold_and_twenty_layer_20261002/evaluation/joint/n16_seed0.json').read_text())['checkpoints']
    expected={r['layer']:r['sha256'] for r in expected}
    positions={};hashes={};grouped=defaultdict(list)
    def key(a):return (a['context'],a['ffn'],a['ffn_samples'] if a['ffn']=='and' else 0,a['mode'],a['seed'])
    assert {key(r['args']) for r in records}=={key(dict(ffn_samples=4,**j)) if 'ffn_samples' not in j else key(j) for j in protocol['jobs']}
    for r in records:
        a=r['args'];ctx=a['context']
        assert r['attention_layers']==24 and r['source_tokens']==299078
        assert r['scored_tokens']==(4096 if ctx==2048 else 2048)
        assert a['batch']==(16 if ctx==2048 else 4) and a['ffn_chunk']==256
        assert math.isfinite(r['perplexity'])
        for source,digest in r['source_sha256'].items():
            archived=ROOT/'executed_sources'/source
            assert hashlib.sha256(archived.read_bytes()).hexdigest()==digest,source
            # Preserve execution bytes; normal Git checkouts may normalize legacy CRLF.
            assert Path(source).read_bytes().replace(b'\r\n',b'\n')==archived.read_bytes().replace(b'\r\n',b'\n'),source
        assert (not r['checkpoints']) if a['ffn']=='original' else ({e['layer']:e['sha256'] for e in r['checkpoints']}==expected)
        if a['ffn'] in hashes:assert hashes[a['ffn']]==r['mlp_sha256']
        else:hashes[a['ffn']]=r['mlp_sha256']
        pos=[(e['start_token'],e['first_target_token'],e['last_target_token']) for e in r['examples']]
        if ctx in positions:assert positions[ctx]==pos
        else:positions[ctx]=pos
        grouped[key(a)[:-1]].append(r)
    assert len({r['text_sha256'] for r in records})==1
    rows=[]
    for k,rs in sorted(grouped.items()):
        row=dict(zip(('context','ffn','ffn_samples','attention'),k))
        row.update({'seeds':len(rs),'ppl_mean':statistics.mean(r['perplexity'] for r in rs),
            'ppl_sd':statistics.pstdev(r['perplexity'] for r in rs),
            'mean_nll':statistics.mean(r['mean_nll'] for r in rs),
            'prefill_seconds_mean':statistics.mean(r['prefill_seconds'] for r in rs),
            'decode_seconds_mean':statistics.mean(r['decode_seconds'] for r in rs),
            'peak_allocated_bytes_max':max(r['peak_allocated_bytes'] for r in rs),
            'unique_V_per_head':statistics.mean(r['sampling_stats'].get('unique_value_fraction_per_head',1.) for r in rs),
            'unique_V_GQA_union':statistics.mean(r['sampling_stats'].get('unique_value_fraction_GQA_group_union',1.) for r in rs)})
        rows.append(row)
    index={(r['context'],r['ffn'],r['ffn_samples'],r['attention']):r for r in rows}
    interactions=[]
    for ctx in (2048,8192):
        base=index[ctx,'original',0,'sdpa'];attn=index[ctx,'original',0,'tree']
        for n in (0,4,16):
            ffn=index[ctx,'and',n,'sdpa'];both=index[ctx,'and',n,'tree']
            interactions.append({'context':ctx,'ffn_samples':n,
                'attention_only_delta_nll':attn['mean_nll']-base['mean_nll'],
                'attention_on_ffn_delta_nll':both['mean_nll']-ffn['mean_nll'],
                'factorial_interaction_nll':both['mean_nll']-ffn['mean_nll']-attn['mean_nll']+base['mean_nll'],
                'both_vs_ffn_ppl_percent':100*(both['ppl_mean']/ffn['ppl_mean']-1),
                'both_vs_original_ppl_percent':100*(both['ppl_mean']/base['ppl_mean']-1)})
    paired=[]
    for ctx in (2048,8192):
        for n in (4,16):
            for seed in range(3):
                a=next(r for r in records if key(r['args'])==(ctx,'and',n,'sdpa',seed))
                b=next(r for r in records if key(r['args'])==(ctx,'and',n,'tree',seed))
                paired.append({'context':ctx,'ffn_samples':n,'seed':seed,'delta_nll':b['mean_nll']-a['mean_nll'],
                    'ppl_percent':100*(b['perplexity']/a['perplexity']-1),
                    'example_delta_nll':[(y['nll_sum']-x['nll_sum'])/x['scored_tokens'] for x,y in zip(a['examples'],b['examples'])]})
    result={'rows':rows,'interactions':interactions,'paired':paired}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    audit={'status':'passed','evaluations':len(records),'executed_source_bytes_verified':True,
        'published_source_equivalent_after_line_ending_normalization':True,'checkpoints':expected,
        'FFN_hashes_identical_within_condition':True,'targets_identical_within_context':True,
        'text_sha256':records[0]['text_sha256'],'all_evaluation_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'evaluation').glob('*.json'))}}
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# 20层 AND FFN 与精确 softmax 后 PV 采样的组合评估','',
        '实验日期：2026-10-06。48项固定协议评估全部完成，无新增训练。',
        '采用用户确认的20层K=4 AND联合版历史最佳验证检查点（2026-10-02，第800步），保留原始FFN层0、2、3、23。',
        '不是10月5日另一个训练环境中的方差约束检查点，也不声称在所有已有训练seed中重新选出了全局最优。',
        '注意力采用已发布的二叉Bernoulli树，先精确计算QK/softmax，再抽取512个类别样本聚合V；不是Ising或块筛选。','',
        '## 评估范围','',
        '- FFN在prefill和decode均启用所选配置；注意力仅在decode替换全部24层，prefill注意力保持精确。',
        '- FFN使用展开的0/1单bit和AND读出，最后平均N条路径；N=0为条件均值诊断，不是随机网络精确期望或性能下界。',
        '- FFN与注意力使用分开的随机生成器状态；同一seed的FFN随机流在不同注意力模式间配对。',
        '- 固定WikiText-2后缀：2k为32例/4096个计分token，8k为16例/2048个计分token；不是完整测试集PPL。',
        '- 各上下文目标位置与旧注意力实验一致；batch分别16/4，FFN按256个token分块，所有条件匹配。','',
        '## 主要结果','',
        '| FFN | 注意力 | seed数 | 2k PPL | 8k PPL |','| --- | --- | ---: | ---: | ---: |']
    for f,n,m,label in [('original',0,'sdpa','原始'),('original',0,'tree','原始'),('and',4,'sdpa','AND N4'),('and',4,'tree','AND N4'),('and',16,'sdpa','AND N16'),('and',16,'tree','AND N16'),('and',0,'sdpa','AND mean-field'),('and',0,'tree','AND mean-field')]:
        rr=[index[c,f,n,m] for c in (2048,8192)]
        lines.append(f"| {label} | {'原始SDPA' if m=='sdpa' else 'PV S512'} | {rr[0]['seeds']} | "+' | '.join(f"{r['ppl_mean']:.5f} ± {r['ppl_sd']:.5f}" for r in rr)+' |')
    lines+=['','±是推理seed总体标准差，不是数据集置信区间。原始/mean-field确定性对照只有一次；随机主要配置均为3次。','',
        '## 叠加效应','',
        '| 上下文 | FFN N | 加入PV采样后的PPL增幅 | 相对原始PPL增幅 | NLL交互项 |','| ---: | ---: | ---: | ---: | ---: |']
    for r in interactions:lines.append(f"| {r['context']} | {r['ffn_samples']} | {r['both_vs_ffn_ppl_percent']:.3f}% | {r['both_vs_original_ppl_percent']:.3f}% | {r['factorial_interaction_nll']:+.6f} |")
    lines+=['','NLL交互项=组合−仅FFN−仅注意力＋原始。正值表示在当前样本上的损失增量超过相加，负值表示小于相加；',
        '有限文本与3个推理seed不足以将小差异解释为普遍协同或抵消。逐seed及逐例配对差值见summary.json。','',
        '## 数值参考与访问范围','',
        '| 上下文 | 原始FFN手写dense PPL | AND N16手写dense PPL | 组合N16单头V位置比例 | 组合N16 GQA地址并集 |','| ---: | ---: | ---: | ---: | ---: |']
    for c in (2048,8192):
        r=index[c,'and',16,'tree']
        lines.append(f"| {c} | {index[c,'original',0,'dense']['ppl_mean']:.5f} | {index[c,'and',16,'dense']['ppl_mean']:.5f} | {r['unique_V_per_head']:.2%} | {r['unique_V_GQA_union']:.2%} |")
    lines+=['','手写dense与tree共享FP32注意力计算；原始SDPA使用融合BF16实现，两者可能有微小数值差异。',
        '当前PV仍用密集count/S @ V，表中是逻辑地址统计，不是实际显存读取量或加速。FFN展开读出也没有专用二值内核。',
        '时间记录包含软件模拟，且多GPU并行可能干扰墙钟，不作为公平性能基准。完整KV仍保留；不宣称缓存容量下降。','',
        '## 验证与复现','',
        '检查点SHA256与历史20层最佳检查点一致；所有48项源码、文本和计分位置核验通过。',
        '集成测试验证随机FFN下的SDPA委托逐位一致、注意力不消耗FFN随机状态、前缀在固定FFN流下逐位一致，组合输出有限。',
        '执行源码快照保留原始字节；部分旧FFN源文件为CRLF，审计同时核验Git版本在换行规范化后等价。',
        '权重及语料不上传Git，复现需已有20个检查点。协议与结果保留所有对照，不根据本轮test重新选模型或训练。',
        '入口：[实验说明](../../experiments/pbit_joint/README.md)；[完整汇总](summary.json)；[审计](audit.json)。','',
        '![组合结果](comparison.png)','']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    configs=[('original',0,'sdpa','Base'),('original',0,'tree','PV only'),('and',4,'sdpa','FFN N4'),('and',4,'tree','N4 + PV'),('and',16,'sdpa','FFN N16'),('and',16,'tree','N16 + PV')]
    for ax,c in zip(axes,(2048,8192)):
        vals=[index[c,f,n,m] for f,n,m,_ in configs]
        ax.bar(range(len(vals)),[v['ppl_mean'] for v in vals],yerr=[v['ppl_sd'] for v in vals],capsize=3,color=['#64748b','#0d9488','#94a3b8','#60a5fa','#94a3b8','#2563eb'])
        ax.set_xticks(range(len(vals)),[t[3] for t in configs],rotation=30,ha='right')
        ax.set_title(f'{c//1024}k context / matched suffixes');ax.set_ylabel('Perplexity (lower is better)')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('20 AND FFNs + exact-softmax PV sampling (S=512)')
    fig.savefig(ROOT/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps({'status':'passed','evaluations':len(records),'interactions':interactions}),flush=True)


if __name__=='__main__':main()
