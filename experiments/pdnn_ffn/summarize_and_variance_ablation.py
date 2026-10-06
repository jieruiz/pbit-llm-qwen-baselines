"""Validate paired conditions and report the prespecified N=4 outcome."""
import argparse
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    root=args.results
    assert (root/'RUN_COMPLETE').is_file()
    plan=read(root/'plan.json');locked=read(root/'locked_selection.json')
    rows=[];initial_values=[];init_hashes=[]
    for seed in (0,1,2):
        for condition in ('control','variance'):
            folder=root/f'{condition}_seed{seed}'
            logs=[json.loads(line) for line in (folder/'train_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            params=logs[0]['args']
            assert logs[0]['layers']==[r['layer'] for r in plan['initialization']['checkpoints']]
            assert params['steps']==2000 and params['seed']==seed and params['sample_count']==4
            assert params['validation_seed']==12345 and params['validation_batches']==64
            assert params['variance_weight']==(0 if condition=='control' else locked['selected_lambda'])
            assert params['learning_rate']==1e-5 and params['kd_weight']==.8
            assert params['sequence_length']==256 and params['gradient_accumulation']==4
            init_hashes.append(params['initial_checkpoint_sha256'])
            assert logs[-1]['event']=='complete'
            validations=[r for r in logs if r['event']=='validation']
            assert len(validations)==9
            initial_values.append(validations[0]['perplexity'])
            summary=read(folder/'summary.json')
            best=min(validations,key=lambda r:r['perplexity'])
            assert best['step']==summary['best_step']
            assert best['perplexity']==summary['best_validation_perplexity']
            values=[]
            for infer_seed in (0,1,2):
                evaluation=read(folder/f'ppl_n4_s{infer_seed}.json')
                assert evaluation['scored_tokens']==299077 and evaluation['replacement_count']==20
                assert evaluation['sample_count']==4 and evaluation['seed']==infer_seed
                assert evaluation['max_length']==2048 and evaluation['stride']==1024
                assert all(c['bits']==4 and c['step']==best['step'] for c in evaluation['checkpoints'])
                values.append(evaluation['perplexity'])
            diagnostics=read(folder/'field_diagnostics.json')
            common=diagnostics['common_teacher_inputs']
            actual=diagnostics['layers']
            rows.append({'condition':condition,'training_seed':seed,'ppl_values':values,
                'ppl_mean':statistics.mean(values),'inference_sample_std':statistics.stdev(values),
                'mean_field_ppl':read(folder/'ppl_n0_s0.json')['perplexity'],
                'n16_ppl':read(folder/'ppl_n16_s0.json')['perplexity'] if seed==0 else None,
                'best_step':best['step'],'validation_ppl':best['perplexity'],
                'seconds':summary['elapsed_seconds'],'peak_allocated_bytes':summary['peak_allocated_bytes'],
                'common_input_n4_variance':statistics.mean(r['n4_variance_nmse'] for r in common.values()),
                'common_input_bias':statistics.mean(r['bias_nmse'] for r in common.values()),
                'actual_input_n4_variance':statistics.mean(r['n4_variance_nmse'] for r in actual.values()),
                'actual_input_bias':statistics.mean(r['same_input_bias_nmse'] for r in actual.values()),
                'gate_saturation':statistics.mean(r['gate_saturation'] for r in actual.values()),
                'value_saturation':statistics.mean(r['value_saturation'] for r in actual.values())})
    assert len(set(initial_values))==1
    assert all(h==init_hashes[0] for h in init_hashes)
    aggregates={}
    for condition in ('control','variance'):
        selected=[r for r in rows if r['condition']==condition]
        aggregates[condition]={key:statistics.mean(r[key] for r in selected) for key in
            ('ppl_mean','seconds','common_input_n4_variance','common_input_bias','actual_input_n4_variance','actual_input_bias')}
        aggregates[condition]['training_seed_sample_std']=statistics.stdev(r['ppl_mean'] for r in selected)
        aggregates[condition]['mean_best_validation_ce']=statistics.mean(math.log(r['validation_ppl']) for r in selected)
    paired=[]
    for seed in (0,1,2):
        a=next(r for r in rows if r['condition']=='control' and r['training_seed']==seed)
        b=next(r for r in rows if r['condition']=='variance' and r['training_seed']==seed)
        paired.append({'seed':seed,'delta_ppl':b['ppl_mean']-a['ppl_mean']})
    a,b=aggregates['control'],aggregates['variance']
    gain=1-b['ppl_mean']/a['ppl_mean'];wins=sum(r['delta_ppl']<0 for r in paired)
    valid=b['mean_best_validation_ce']<a['mean_best_validation_ce']
    result={'rows':rows,'aggregates':aggregates,'paired':paired,'relative_ppl_improvement':gain,
        'paired_wins':wins,'validation_improved':valid,'prespecified_target_met':gain>=.02 and wins>=2 and valid,
        'selected_lambda':locked['selected_lambda'],'training_time_ratio':b['seconds']/a['seconds'],
        'common_input_variance_reduction':1-b['common_input_n4_variance']/a['common_input_n4_variance']}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['# 二十层 AND FFN：固定 K=4、N=4 的方差约束对照','',
        f"预定目标：相对等预算KL+CE对照，N=4测试PPL降低至少2%，至少2/3配对训练种子改善，平均最佳验证CE同向。",
        f"结果：**{'达到' if result['prespecified_target_met'] else '未达到'}**。相对改善 {gain*100:.3f}%；配对胜出 {wins}/3；固定lambda={locked['selected_lambda']:g}。",'',
        '| 条件 | 训练seed | best步 | 验证PPL | N=4测试均值 ± 推理样本标准差 | Mean-field | N=16 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        n16=f"{r['n16_ppl']:.6f}" if r['n16_ppl'] is not None else '未测（预定子集）'
        lines.append(f"| {r['condition']} | {r['training_seed']} | {r['best_step']} | {r['validation_ppl']:.6f} | {r['ppl_mean']:.6f} ± {r['inference_sample_std']:.6f} | {r['mean_field_ppl']:.6f} | {n16} |")
    lines+=['','## 三个配对训练种子与成本','',
        '| 条件 | N=4均值 ± 训练样本标准差 | 平均训练秒数 | 同teacher输入Var/4 NMSE | 同teacher输入均值偏差NMSE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for condition in ('control','variance'):
        r=aggregates[condition]
        lines.append(f"| {condition} | {r['ppl_mean']:.6f} ± {r['training_seed_sample_std']:.6f} | {r['seconds']:.2f} | {r['common_input_n4_variance']:.6f} | {r['common_input_bias']:.6f} |")
    lines+=['',f"同teacher输入平均局部Var/4减少 {result['common_input_variance_reduction']*100:.2f}%；训练时间比例 {result['training_time_ratio']:.3f}。",
        '', '## 方法与限制','',
        '- 共同初始化为本账号按上游同方法重训的20个局部模型及1000步联合best；不是师弟checkpoint的字节复现。初始化身份见plan.json、initial/initializer_manifest.json。',
        '- 三个配对训练seed，各2000追加步，KL/CE、原始可训练参数、批次序列、LR和冻结规则相同；B只增加固定尺度解析方差。训练使用因式分解四路径，所有采样test使用实际展开的0/1与AND读出。',
        '- lambda仅用单层保留验证集选择；pilot权重不进入正式初始化。全部六组训练结束后才读取候选test。验证固定64个256窗口和RNG12345，按纯验证PPL选best。',
        '- 固定参考尺度来自原teacher在train去掉尾部后的32768token输出，未用test或当前student幅度更新分母。',
        '- 诊断同时报告完全相同teacher输入及各student实际rollout输入；局部条件方差不等于完整模型输出方差，不用有限PPL差值精确分解偏差与方差。',
        '- N=16仅训练seed0的两组；mean-field每组各一次。三组共用一个初始化，只重复继续训练阶段；推理重复不是独立训练。',
        '- 只有一个语料，未验证下游任务、物理p-bit相关性、低精度驱动和硬件速度/能耗。解析矩未完整覆盖最终BF16舍入。',
        '- 协议：[AND_VARIANCE_PROTOCOL_ZH.md](../../experiments/pdnn_ffn/AND_VARIANCE_PROTOCOL_ZH.md)。后续目标以协议中的条件分支为准，不将建议写成已完成实验。','']
    (root/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
