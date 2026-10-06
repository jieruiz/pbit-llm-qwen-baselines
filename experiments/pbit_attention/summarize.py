"""Audit sampled-attention results and write a Chinese report and figure."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path('results/pbit_attention_20261003')
EXP=Path('experiments/pbit_attention')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    for name in ('status.json','repeat_status.json'):
        if read(ROOT/name)['stage']!='complete':
            raise RuntimeError(f'Incomplete: {name}')
    protocol=read(ROOT/'protocol.json')
    for name,digest in protocol['source_sha256'].items():
        if hashlib.sha256((EXP/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Source changed: {name}')
    repeat=read(ROOT/'long_context_repeat_protocol.json')
    assert hashlib.sha256((EXP/'repeat_long_context.py').read_bytes()).hexdigest()==repeat['script_sha256']
    assert read(ROOT/'test_sampling.json')['status']=='passed'
    assert read(ROOT/'test_integration.json')['status']=='passed'
    records=[read(p) for p in sorted((ROOT/'evaluation').glob('*.json'))]
    assert len(records)==34
    assert len({r['mlp_sha256'] for r in records})==1, 'FFNs differ'
    assert len({r['text_sha256'] for r in records})==1, 'test source differs'
    for r in records:
        a=r['args']; context=a['context']
        assert a['decode']==128 and r['attention_layers']==24
        assert (a['examples'],a['batch'],r['scored_tokens'])==((32,16,4096) if context==2048 else (16,4,2048))
        assert math.isclose(r['mean_nll'],sum(e['nll_sum'] for e in r['examples'])/r['scored_tokens'],rel_tol=1e-10)
        for name,digest in r['source_sha256'].items():
            assert hashlib.sha256((EXP/name).read_bytes()).hexdigest()==digest, name
    rows=[]; baselines={}
    for context in (2048,8192):
        group=[r for r in records if r['args']['context']==context]
        target_lists=[[e['first_target_token'] for e in r['examples']] for r in group]
        assert all(t==target_lists[0] for t in target_lists), 'unmatched target positions'
        baselines[str(context)]={r['args']['mode']:r['perplexity'] for r in group if r['args']['mode'] in ('dense','sdpa')}
        for samples in ((32,128,512) if context==2048 else (128,512)):
            for mode in ('tree','independent'):
                selected=[r for r in group if r['args']['mode']==mode and r['args']['samples']==samples]
                assert sorted(r['args']['seed'] for r in selected)==[0,1,2]
                values=[r['perplexity'] for r in selected]
                mean=statistics.mean(values)
                rows.append({'context':context,'mode':mode,'samples':samples,'ppl_mean':mean,'ppl_std':statistics.pstdev(values),
                    'ppl_values':values,'above_original_percent':100*(mean/baselines[str(context)]['sdpa']-1),
                    'above_dense_reference_percent':100*(mean/baselines[str(context)]['dense']-1),
                    'GQA_unique_value_fraction':statistics.mean(r['sampling_stats']['unique_value_fraction_GQA_group_union'] for r in selected),
                    'mean_count':statistics.mean(r['sampling_stats']['mean_selected_events_per_S_rounds'] for r in selected),
                    'count_std':statistics.mean(r['sampling_stats']['selected_events_std'] for r in selected),
                    'reference_decode_seconds_mean':statistics.mean(r['decode_seconds'] for r in selected)})
    comparison={'baselines':baselines,'rows':rows,'evaluation_count':34,'original_ffn_sha256':records[0]['mlp_sha256'],
        'source_audit':'All evaluated source hashes, FFN hashes, corpus hashes and scored positions verified',
        'protocol':protocol,'repeat_protocol':repeat}
    (ROOT/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n',encoding='utf-8')
    lines=['# 原始FFN下的p-bit注意力比较（2026-10-03）','',
        '## 结论与范围','',
        '两种采样都可在保留原始FFN的条件下近似注意力Value聚合；S增大时本轮误差减小。',
        '这次同时替换24层注意力的AV步骤，不训练、不修改FFN，不近似QK或Softmax。',
        '两种采样均为理想概率可编程p-bit的软件仿真；没有真实器件、驱动误差或随机相关性。',
        '本轮是独立类别采样SANTA，不包含论文S²ANTA的分层/系统采样降方差。','',
        '## 两种实现','',
        '1. `tree`：对注意力概率构造补零至2的幂的二叉质量树，每个节点按右子树质量/总质量',
        '   做0/1 Bernoulli选择，直到叶子产生一个Value地址。每轮恰好选择一个，重复S轮。',
        '2. `independent`：每个地址独立b_i~Bernoulli(p_i)，重复S轮；输出sum(b_i V_i)/S。',
        '   一轮可能选中零行或多行。除数固定为S，不除以实际选中数。为了加快仿真，用',
        '   C_i~Binomial(S,p_i)直接生成S轮的计数，概率分布与逐轮独立p-bit严格等价。','',
        '两种均以FP32计数/S乘Value作线性聚合参考，没有实现真正稀疏gather内核；',
        '线性运算允许合并重复地址，但这里的GPU矩阵仍是密集运算，不代表二值硬件执行。',
        '二叉树用rand<p模拟可编程Bernoulli响应；sigmoid物理p-bit仍需概率到驱动的映射。','',
        '## 测试协议','',
        '- Qwen2.5-0.5B Base，BF16模型/缓存，FP32分数、Softmax与采样读出。',
        '- 所有24层原始Qwen2MLP对象保留，运行前后权重SHA256一致；不同运行也完全一致。',
        '- 使用WikiText-2 test（299078 token）中均匀选取的互不重叠片段，无训练及参数选择。',
        '- 2k：32段，每段前2047 token精确prefill，然后128个单token缓存解码查询，计分4096个后续token。',
        '- 8k：16段，前8191 token精确prefill，然后128个解码查询，计分2048个后续token。',
        '- 所谓2k/8k表示第一个随机查询的可见上下文为2048/8192，随后上下文随解码增长。',
        '- 解码使用真实后续token进行teacher forcing，保留此前随机层产生的KV缓存。',
        '- 2k和8k的目标位置不同，只在各自表格内部比较。不是完整WT2滑窗PPL，不能与11.652735直接比较。',
        '- 每个随机配置3个推理seed；±为总体标准差，不是跨文本或训练置信区间。',
        '- 首批25项后，因8k单seed方法排序不稳定，补充两seed及8k原始SDPA，共34项评估。','',
        '## PPL结果','']
    for context in (2048,8192):
        baseline=baselines[str(context)]
        lines += [f'### 起始上下文{context}', '',f"原始SDPA：**{baseline['sdpa']:.6f}**；同精度手写密集参考：**{baseline['dense']:.6f}**。",'',
            '| S | 二叉p-bit选择树PPL | 独立p-bit PPL | 相对原始PPL增加（树/独立） |',
            '| ---: | ---: | ---: | ---: |']
        for n in ((32,128,512) if context==2048 else (128,512)):
            a,b=[next(r for r in rows if r['context']==context and r['samples']==n and r['mode']==m) for m in ('tree','independent')]
            lines.append(f"| {n} | {a['ppl_mean']:.6f} ± {a['ppl_std']:.6f} | {b['ppl_mean']:.6f} ± {b['ppl_std']:.6f} | {a['above_original_percent']:.2f}% / {b['above_original_percent']:.2f}% |")
        lines += ['']
    lines += ['![PPL versus sample budget](quality.png)','',
        '两种方案在S=128/512下表现接近，不能根据几个seed的小差异认定其中一种必然优越。',
        'SANTA保持每轮权重和为1；独立p-bit只保证期望权重和为1，会引入随机幅度波动。',
        '这种差异可解释部分误差，但两者方差没有对任意Value成立的统一大小关系。','',
        '## 逻辑Value访问与成本','',
        'Qwen配置为14个query head、2个KV head，每组共享7个query head。下表统计整个S轮中',
        '非零计数地址在GQA组内的并集，已计入共享KV头；分母是完整Value地址数。',
        '这只是理论可跳过的地址比例，不是实测带宽减少量；缓存行、读取合并和采样器遍历仍有成本。','',
        '| 上下文 | 方法 | S | GQA组内不同Value行占比 | 总选中次数均值±标准差 |',
        '| ---: | --- | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['context']} | {r['mode']} | {r['samples']} | {100*r['GQA_unique_value_fraction']:.3f}% | {r['mean_count']:.3f} ± {r['count_std']:.3f} |")
    lines += ['',
        '随机配置的PyTorch参考解码比原始SDPA慢：树构建/遍历、概率生成及密集读出都有开销。',
        '计时包含统计开销且没有专门预热/重复计时设计，只存作运行记录，不作为硬件速度比较。',
        'Binomial模拟耗时不能当作S轮物理p-bit耗时；独立方案若逐位置逐轮生成，约需L×S个bit，',
        '选择树每个地址约需ceil(log2 L)次条件bit选择，另需构建概率树及寻址。并行度、器件数量',
        '和驱动映射会影响实际成本。尚未证明任一方案加速或节能。','',
        '## 验证与文件','',
        '40000次抽样检验均值和解析方差、非2次幂树、0/1边界、固定类别计数、独立零/多选、',
        'Binomial与字面p-bit模拟、常数Value保持性质；全部通过。',
        '集成检查确认原始FFN对象保留、prefill逐位一致、SDPA委托逐位一致；FP32密集参考与',
        '原始SDPA存在小的有限精度差别，因此每个上下文都保留两种密集基线。','',
        '源码：`experiments/pbit_attention/`；每次运行：`evaluation/`；命令：`commands/`；',
        '原始与追加协议：`protocol.json`、`long_context_repeat_protocol.json`；汇总：`comparison.json`。',
        '执行源码哈希、34次FFN哈希、测试文本与计分位置已核对。','',
        '本轮未把已有随机FFN叠加进来，后续组合必须另测，不能直接把两项单独误差相加。','']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    if '--no-plot' in sys.argv:
        print(json.dumps({'audit':'passed','evaluations':len(records),'baselines':baselines}))
        return
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.3))
    for ax,context in zip(axes,(2048,8192)):
        for mode,color,label in (('tree','#2166ac','Categorical p-bit tree'),('independent','#d6604d','Independent p-bits')):
            rr=[r for r in rows if r['context']==context and r['mode']==mode]
            ax.errorbar([r['samples'] for r in rr],[r['ppl_mean'] for r in rr],yerr=[r['ppl_std'] for r in rr],
                marker='o',capsize=4,color=color,label=label)
        ax.axhline(baselines[str(context)]['sdpa'],linestyle='--',color='#333333',label='Original SDPA')
        ticks=[32,128,512] if context==2048 else [128,512]
        ax.set_xscale('log',base=2); ax.set_xticks(ticks,[str(n) for n in ticks])
        ax.set_title(f'Initial context {context}; original FFNs')
        ax.set_xlabel('S (sampling rounds)'); ax.set_ylabel('Sampled suffix perplexity')
        ax.grid(alpha=.2); ax.legend(fontsize=8,frameon=False)
    fig.tight_layout(); fig.savefig(ROOT/'quality.png',dpi=180); plt.close(fig)
    print(json.dumps({'baselines':baselines,'rows':rows},indent=2))


if __name__=='__main__':
    main()
