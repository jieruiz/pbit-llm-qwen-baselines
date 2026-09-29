"""Summarize paired TRAINING seeds separately from inference randomness."""
import argparse
import json
import math
from pathlib import Path
import statistics as stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    args = p.parse_args()
    root = args.root
    rows = []
    for seed in (0, 1, 2):
        for condition in ('fixed', 'calibrated'):
            path = root/f'{condition}_seed{seed}'
            summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
            fields = json.loads((path/'field_diagnostics.json').read_text(encoding='utf-8'))['layers']
            ppl = []
            for inference_seed in (0, 1, 2):
                value = json.loads((path/f'ppl_n4_s{inference_seed}.json').read_text(encoding='utf-8'))
                assert value['scored_tokens'] == 299077
                ppl.append(value['perplexity'])
            assert summary['validation_scored_tokens'] == 16320
            rows.append(dict(condition=condition, training_seed=seed, ppl_values=ppl,
                ppl_mean=stats.mean(ppl), inference_sample_std=stats.stdev(ppl),
                mean_field_ppl=json.loads((path/'ppl_n0_s0.json').read_text())['perplexity'],
                n16_ppl=json.loads((path/'ppl_n16_s0.json').read_text())['perplexity'],
                best_step=summary['best_step'], validation_ppl=summary['best_validation_perplexity'],
                mean_input_saturation=stats.mean(r['input_saturation_fraction'] for r in fields.values()),
                mean_hidden_saturation=stats.mean(r['hidden_saturation_fraction'] for r in fields.values()),
                mean_layer_nmse=stats.mean(r['same_student_input_ffn_nmse'] for r in fields.values()),
                seconds=summary['elapsed_seconds'], peak_allocated_bytes=summary['peak_allocated_bytes']))
    aggregates = {}
    for condition in ('fixed', 'calibrated'):
        selected = [r for r in rows if r['condition']==condition]
        means = [r['ppl_mean'] for r in selected]
        aggregates[condition] = dict(ppl_mean=stats.mean(means), training_seed_sample_std=stats.stdev(means),
            mean_best_validation_ce=stats.mean(math.log(r['validation_ppl']) for r in selected),
            mean_input_saturation=stats.mean(r['mean_input_saturation'] for r in selected),
            mean_hidden_saturation=stats.mean(r['mean_hidden_saturation'] for r in selected),
            mean_layer_nmse=stats.mean(r['mean_layer_nmse'] for r in selected),
            mean_seconds=stats.mean(r['seconds'] for r in selected))
    paired = [dict(training_seed=s, delta_ppl=next(r['ppl_mean'] for r in rows if r['training_seed']==s and r['condition']=='calibrated')
                   - next(r['ppl_mean'] for r in rows if r['training_seed']==s and r['condition']=='fixed')) for s in (0, 1, 2)]
    relative = 1-aggregates['calibrated']['ppl_mean']/aggregates['fixed']['ppl_mean']
    wins = sum(r['delta_ppl']<0 for r in paired)
    validation_improved = aggregates['calibrated']['mean_best_validation_ce'] < aggregates['fixed']['mean_best_validation_ce']
    met = relative >= .01 and wins >= 2 and validation_improved
    result = dict(rows=rows, aggregates=aggregates, paired_differences=paired,
                  relative_ppl_improvement=relative, training_seed_wins=wins,
                  validation_improved=validation_improved, prespecified_target_met=met)
    (root/'comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    lines = ['# 十层 FFN：逐通道 p-bit 编码校准对照结果', '',
        '预定目标：相对固定编码，同预算校准降低至少 1% PPL，至少 2/3 配对训练种子改善，平均最佳验证 CE 同向改善。', '',
        f"目标判定：**{'达到' if met else '未达到'}**。PPL 相对改善 {relative*100:.3f}%，配对训练种子胜出 {wins}/3；验证同向改善：{validation_improved}。", '',
        '| 条件 | 训练 seed | 最佳步数 | 验证 PPL | N=4 PPL 均值 ± 推理样本标准差 | N=16 | Mean-field |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['training_seed']} | {r['best_step']} | {r['validation_ppl']:.6f} | {r['ppl_mean']:.6f} ± {r['inference_sample_std']:.6f} | {r['n16_ppl']:.6f} | {r['mean_field_ppl']:.6f} |")
    lines += ['', '## 训练重复与成本', '', '| 条件 | 三个训练种子均值 ± 样本标准差 | 平均训练秒数 |', '| --- | ---: | ---: |']
    for name, a in aggregates.items():
        lines.append(f"| {name} | {a['ppl_mean']:.6f} ± {a['training_seed_sample_std']:.6f} | {a['mean_seconds']:.2f} |")
    lines += ['', '## 保留验证集上的场与局部误差诊断', '',
              '| 条件 | 入口饱和比例 | 隐藏饱和比例 | 按层等权平均的同输入 FFN NMSE |',
              '| --- | ---: | ---: | ---: |']
    for name, a in aggregates.items():
        lines.append(f"| {name} | {a['mean_input_saturation']:.4%} | {a['mean_hidden_saturation']:.4%} | {a['mean_layer_nmse']:.6f} |")
    lines += ['', '饱和定义为 p-bit 均值绝对值大于 0.99；表中先对十层等权平均，再对三个训练种子平均。局部 NMSE 只解释行为，不能替代完整模型 PPL，也不单独建立因果机制。']
    lines += ['', '三次训练共享同一个十层初始化，考察联合适配随机性；初始化本身没有重复。每个训练模型又做三次 N=4 推理，不能把九次推理当作九次独立训练。', '',
        '## 方法与限制', '',
        '- 相同层集合、初始化、文本批次种子、2000 步预算、原权重学习率和 0.8 KL + 0.2 CE。仅实验组新增入口/隐藏场的逐通道尺度与阈值及对应参数组 LR=3e-4。',
        '- 保持两个矩阵的采样输入严格为 ±1，仅在每个 FFN 最后的连续读出平均。没有连续矩阵补偿。增加 115200 个参数，约十层 P-DNN 参数的 0.132%。',
        '- 入口与隐藏场同时校准，未拆分两者的贡献。隐藏场的仿射尺度/偏置可折叠入前一个连续权重投影，因此该部分属于优化重参数化；新增参数量不等于同等新增表达能力。本轮没有验证折叠后的数值或硬件成本。',
        '- 固定 64×256 验证窗口、随机种子 12345，按验证 PPL 选择 best，另保存 final。测试覆盖 299077 token，2048 窗口、1024 步长；所有六组训练结束后才读取候选 test。',
        '- 原始 BF16、十层基线重建、预检日志与制品哈希随结果保存。field_diagnostics.json 中的 teacher 原 FFN 仅在 student 实际输入上记录局部误差，不进入推理控制。',
        '- N=0 为 mean-field 代理，不是精确无限采样极限。本轮没有进行二十层扩展、下游任务评测、长期持续训练或物理 p-bit 能耗验证。',
        '- 执行协议：[TEN_LAYER_CALIBRATION_ZH.md](../../experiments/pdnn_ffn/TEN_LAYER_CALIBRATION_ZH.md)。模型权重留在个人服务器及指定制品存储，身份见清单。', '']
    (root/'RESULTS_ZH.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))


if __name__ == '__main__':
    main()
