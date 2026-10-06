"""Audit saved experiments, then generate Chinese report and scientific plots."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/pbit_softmax_router_20261006')
EXP=Path('experiments/pbit_softmax_router')


def key(a):
    mode=a['mode'];sample=mode in ('iid','ising');router=mode.startswith(('block','oracle'))
    return (a['context'],mode,a['samples'] if sample else 0,a['spacing'] if mode=='ising' else 0,
        a['burn'] if mode=='ising' else 0,a['ratio'] if router else 0,a['rank'] if mode.startswith('block') else 0)


def main():
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'evaluation').glob('*.json'))]
    assert len(records)==140,len(records)
    assert len({r['mlp_sha256'] for r in records})==1
    assert len({r['text_sha256'] for r in records})==1
    assert all(r['attention_layers']==24 for r in records)
    positions={}
    for r in records:
        for name,digest in r['source_sha256'].items():
            assert hashlib.sha256((EXP/name).read_bytes()).hexdigest()==digest,(name,r['args'])
        ctx=r['args']['context']
        pos=[(e['start_token'],e['first_target_token'],e['last_target_token'],e['scored_tokens']) for e in r['examples']]
        if ctx in positions:assert pos==positions[ctx]
        else:positions[ctx]=pos
        assert r['scored_tokens']==(4096 if ctx==2048 else 2048)
    groups=defaultdict(list)
    for r in records:groups[key(r['args'])].append(r)
    aggregate={}
    table=[]
    for k,rs in sorted(groups.items()):
        v=[r['perplexity'] for r in rs]
        item=dict(zip(('context','mode','samples','spacing','burn','ratio','rank'),k))
        item.update({'seeds':len(rs),'ppl_mean':stats.mean(v),'ppl_std_population':stats.pstdev(v),
            'ppl_min':min(v),'ppl_max':max(v),'decode_seconds_mean':stats.mean(r['decode_seconds'] for r in rs)})
        for field in ('unique_value_fraction_per_head','unique_value_fraction_GQA_group_union','teacher_probability_mass_on_selected_positions','mean_attention_TV','adjacent_sample_repeat_fraction'):
            item[field]=stats.mean(r['sampling_stats'].get(field,1. if 'fraction' in field else 0.) for r in rs)
        if k[1].startswith(('block','oracle')):
            ctx,_,_,_,_,_,_=k
            lengths=list(range(ctx,ctx+128));B=rs[0]['args']['block_size']
            padded=sum(math.ceil(L/B)*B for L in lengths);total=sum(lengths)
            block_frac=stats.mean(r['sampling_stats']['selected_block_fraction'] for r in rs)
            item['candidate_token_fraction_per_head']=(block_frac*padded-(padded-total))/total
            # The online counter records nonzero FP32 weights; mandatory last
            # block permits exact candidate coverage reconstruction. Union
            # undercount is bounded by seven times the per-head undercount.
            missed=max(0,item['candidate_token_fraction_per_head']-item['unique_value_fraction_per_head'])
            item['candidate_GQA_union_lower_bound']=item['unique_value_fraction_GQA_group_union']
            item['candidate_GQA_union_upper_bound']=min(1.,item['candidate_GQA_union_lower_bound']+7*missed)
        aggregate[k]=item;table.append(item)
    columns=list(dict.fromkeys(c for row in table for c in row))
    with (ROOT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(table)
    (ROOT/'summary.json').write_text(json.dumps(table,indent=2)+'\n')
    audit={'status':'passed','evaluations':len(records),'unique_configurations':len(groups),
        'source_hashes_verified':True,'ffn_and_text_hashes_identical':True,'target_positions_identical_within_context':True,
        'diagnostics':json.loads((ROOT/'diagnostics.json').read_text())['status'],
        'integration':json.loads((ROOT/'integration.json').read_text()),
        'note':'Inference-seed population SD is not a dataset confidence interval. All selection and burn choices were screened on these suffixes.'}
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    def get(ctx,mode,samples=0,gap=0,burn=0,ratio=0,rank=0):return aggregate[(ctx,mode,samples,gap,burn,ratio,rank)]
    def pair(mode,samples=0,gap=0,burn=0,ratio=0,rank=0):
        rows=[get(c,mode,samples,gap,burn,ratio,rank) for c in (2048,8192)]
        return ' | '.join(f"{r['ppl_mean']:.5f} ± {r['ppl_std_population']:.5f}" for r in rows)
    lines=['# Ising SoftMax与BoltzFormer式筛选：Qwen实验结果（2026-10-06）','',
        '完成140项评估：首轮70项、随机种子复测60项、保守预热确认10项。全部已结束。',
        'Qwen2.5-0.5B Base全部24层仅替换decode注意力；保留原始FFN、投影、RoPE及精确prefill；无训练。',
        '2k/8k分别计分4096/2048个WT2后缀token，位置与历史注意力实验相同。不是完整WT2 PPL；不同上下文不能横向比较。','',
        '## 1. 连接与实现范围','',
        'L类softmax用L-1个0/1 p-bit：全零代表参考类别，单个1代表其余类别。每位逻辑连接度L-2，无向连接(L-1)(L-2)/2。',
        '| 类别数L | p-bit数 | 每位连接度 | 无向连接数 |','| ---: | ---: | ---: | ---: |']
    for L in (64,2048,8192):lines.append(f'| {L} | {L-1} | {L-2} | {(L-1)*(L-2)//2} |')
    lines += ['',
        '这些连接是统一互斥系数，不是需要学习的任意J。可研究共享总激活数反馈，但物理广播、延迟、负载与稳定性仍需设计；不能把逻辑全连接直接当作零成本。',
        '能量采用 E=-sum_i(z_i-z_ref)b_i + lambda*sum_(i<j)b_i*b_j；参考取最大logit。相对场非正，避免差参考引起过慢离开激活态。',
        '**大模型实验是强惩罚极限下的理想异步连续时间热浴动力学，不是有限交叉阵列电路复现。** 从全零出生速率a_i=sigmoid(z_i-z_ref)，从激活i返回全零速率1-a_i。',
        '程序批量生成交替的指数驻留时间并在等时刻观测，保持正确驻留加权与时间相关性；没有偷偷用softmax概率直接生成Ising输出。转移事件计数不能代替等时采样。',
        '时间单位是每个p-bit单位速率时钟的倒数，不能换算成32微秒或直接视作GPU时间。固定目标beta=1预热，不做优化式降温退火。','',
        '## 2. 采样次数与PPL','',
        '下表重点配置采用5个推理seed，±为seed总体标准差，非数据集置信区间。',
        '| 方法 | 样本数 | 间隔 | 预热 | 2k PPL | 8k PPL |','| --- | ---: | ---: | ---: | ---: | ---: |',
        '| 原始SDPA | — | — | — | '+pair('sdpa')+' |',
        '| 手写同精度密集参考 | — | — | — | '+pair('dense')+' |']
    for mode,n,g,b in [('iid',128,0,0),('ising',128,4.,4.),('iid',512,0,0),('ising',512,1.,4.),('ising',512,4.,4.),('ising',512,4.,16.),('iid',1024,0,0),('ising',1024,4.,4.)]:
        lines.append(f'| {mode} | {n} | {g or "独立"} | {b or "—"} | '+pair(mode,n,g,b)+' |')
    lines += ['',
        '**建议起点：预热16、读取间隔4、采样512次。** 这是当前理想动力学与本评测上的推荐，不是物理时钟保证。',
        '512次在两种上下文下已接近基线；1024次一般继续降低采样误差，但小差异受种子和有限文本影响，不能把不同设置的轻微反向波动解释为机制优势。',
        'Ising512/gap4与已有IID512总体接近，支持它作为softmax类别样本发生器；仍需完整QK，主要替换softmax与后续类别采样。',
        '原始软件为数值softmax->IID样本；Ising路径为QK->相对场->互斥动力学->样本->V平均。诊断代码仍额外计算精确softmax，但它不驱动Ising采样器。','',
        '## 3. 平衡与独立性验证','',
        '有限惩罚：6类、16000副本、600轮逐位随机扫描，对lambda=0/4/8/16/24分别与全状态枚举比较，TV均<0.025；有效状态条件分布与softmax在浮点误差内一致。',
        '压缩连续时间模拟：100000副本，与精确生成矩阵指数的多个瞬态分布比较，逐元素误差<0.006；增量块缓存跨边界误差<1e-6。',
        '6类诊断中512次读数的方差等效独立样本数：间隔0.25约67，间隔1约244，间隔4约491；不能把它当作所有Qwen行通用ESS。',
        '额外提取原始Qwen的20条真实分数（2k/8k、5层、各2个文本位置），直接计算理想生成矩阵的瞬态分布：',
        '| 预热时标 | 平均TV | 最大TV |','| ---: | ---: | ---: |']
    real=json.loads((ROOT/'real_logit_transients.json').read_text())['rows']
    for t in ('0','0.5','1','4','8','16'):
        v=[r['exact_transient_TV'][t] for r in real];lines.append(f'| {t} | {stats.mean(v):.8g} | {max(v):.8g} |')
    lines += ['',
        '选择最大logit为全零参考时，这20条分数在lambda24下的无效稳态质量上界最大为'+f"{max(r['lambda24_invalid_mass_upper_bound'] for r in real):.3g}"+'。',
        'lambda32、4100时标内的硬约束/有限惩罚路径分离概率上界最大为'+f"{max(r['lambda32_hard_vs_finite_path_difference_upper_bound_at_t4100'] for r in real):.3g}"+'。',
        '这些上界来自理想相同热浴速率与输入场的数学比较，不覆盖器件漂移、耦合误差、传播延迟；20条分数不是全部模型状态的保证。','',
        '## 4. BoltzFormer式候选筛选','',
        '**这是免训练的Qwen适配，不是图像论文中已训练MLP置信度模型的复现。** 每64个Key组成一块，缓存每块均值/坐标方差；首个decode建立摘要，之后只更新新Key。',
        'block_mean：q与块均值点积，加log块大小估计质量；block_moment额外加入对角高斯方差修正。block_paper用sigmoid置信度及tau=1/(layer+1)。',
        'Bernoulli候选包含率=1-(1-p_block)^K；K=ceil(预算比例*块数)。强制保留首块和最后两块。预算比例不是最终保留比例，重复抽取及强制块都会影响它。',
        '选中块内部仍做精确softmax注意力。oracle使用完整QK的真实块质量，不能节省QK，只是筛选器诊断；oracle_topk为确定性相关性对照，不是端到端质量的严格上界。',
        '| 上下文 | 方法 | 预算比 | seed数 | PPL | 实际候选位置/头 | GQA组地址并集约 |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for c in (2048,8192):
        for mode,r,rank in [('block_mean',.25,64),('block_mean',.5,64),('block_paper',.5,64),('oracle',.25,0),('oracle_topk',.25,0),('block_mean',.25,16)]:
            a=get(c,mode,ratio=r,rank=rank)
            lines.append(f"| {c} | {mode} / d{rank or 64} | {r} | {a['seeds']} | {a['ppl_mean']:.5f} ± {a['ppl_std_population']:.5f} | {a['candidate_token_fraction_per_head']:.2%} | {a['candidate_GQA_union_lower_bound']:.1%} |")
    lines += ['',
        '8k下均值摘要的预算0.5配置实际保留约13.33%位置，三seed PPL约9.560，比原始9.419增加约1.49%；2k下同配置约10.943，增加约5.49%。',
        'GQA组并集在8k约42.6%，不能把单头13.3%解释为整个KV缓存只读13.3%。实际物理访存和QK跳过仍需稀疏实现。',
        '统计细节：在线地址并集按非零FP32权重计算，极少数候选权重下溢为零；summary.csv由选中块数量重建精确单头候选比例，并给出组并集上下界，避免把下溢当成物理跳读。',
        '真实块质量oracle在更少候选下通常更好，说明廉价置信度仍有改进空间；均值/方差和固定16维子采样不足以稳定预测重要块。',
        '照搬sigmoid及逐层降温也未改善当前免训练版本；不能据此否定原文的已训练图像模型。下一步可蒸馏训练块选择器，比较多代表Key与GQA共享预算。','',
        '## 5. 成本、局限与文件','',
        'Ising仍计算完整QK；通过类别样本可把PV转为选择加法，但本次采用密集count/M @ V作数值参考。',
        '筛选器由缓存摘要驱动，理论上可在选块后才计算精确QK；当前程序为诊断仍计算完整QK及密集masked PV，**实测GPU代码未加速，不能宣称已节省实际带宽或能耗**。',
        '原始SDPA单次decode约3秒/6秒；这些参考实现约7-20秒，具体记录见JSON。并行任务会影响墙钟，不能作为公平内核性能基准。',
        '140项源码SHA256、原始FFN哈希、文本哈希、目标位置均核验通过。只测试生成阶段，尚未测试随机prefill、FFN联合替换、真实电路或训练后的选择器。',
        '测试集已用于探索与选择配置；正式报告泛化结论前应冻结配置并补充独立数据。','',
        '- `summary.csv` / `summary.json`：所有配置均值、seed标准差、候选和访问统计。',
        '- `evaluation/`：140份逐次结果、逐样本NLL及哈希。',
        '- `diagnostics.json`：有限惩罚、瞬态、样本相关性、摘要缓存测试。',
        '- `real_logit_transients.json`：真实Qwen分数上的精确瞬态检查。',
        '- `audit.json`：结果完整性核验。',
        '- `protocol.json`、`repeat_protocol.json`、`burn_protocol.json`：三阶段实验清单。','',
        '![Sampling and routing](comparison.png)','']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,c in enumerate((2048,8192)):
        ax=axes[0,col];base=get(c,'sdpa')['ppl_mean']
        ax.axhline(base,color='black',ls='--',label='Original SDPA')
        for mode,g,b,label,color in [('iid',0,0,'IID','#3577b5'),('ising',4.,4.,'Ising, gap 4, burn 4','#bd5a35')]:
            vals=[get(c,mode,n,g,b) for n in (128,512,1024)]
            ax.errorbar([128,512,1024],[v['ppl_mean'] for v in vals],yerr=[v['ppl_std_population'] for v in vals],marker='o',capsize=3,label=label,color=color)
        v=get(c,'ising',512,4.,16.)
        ax.errorbar([512],[v['ppl_mean']],yerr=[v['ppl_std_population']],marker='*',ms=13,capsize=3,color='#348a53',label='Ising, burn 16')
        ax.set_xscale('log',base=2);ax.set_xticks([128,512,1024],[128,512,1024]);ax.set_xlabel('Samples');ax.set_ylabel('Suffix PPL');ax.set_title(f'{c} tokens: sample budget (5 seeds)');ax.legend(fontsize=8)
        ax=axes[1,col];ax.axhline(base,color='black',ls='--',label='Original SDPA')
        for mode,rank,label in [('block_mean',64,'Mean-key proxy'),('block_paper',64,'Sigmoid + cooling'),('oracle',0,'Full-QK oracle')]:
            vals=[get(c,mode,ratio=r,rank=rank) for r in (.125,.25,.5)]
            ax.plot([100*v['candidate_token_fraction_per_head'] for v in vals],[v['ppl_mean'] for v in vals],marker='o',label=label)
        ax.set_xlabel('Candidate positions per head (%)');ax.set_ylabel('Suffix PPL');ax.set_title(f'{c} tokens: block routing (1-3 seeds)');ax.legend(fontsize=8)
    fig.savefig(ROOT/'comparison.png',dpi=170)
    plt.close(fig)
    print(json.dumps(audit))


if __name__=='__main__':main()
