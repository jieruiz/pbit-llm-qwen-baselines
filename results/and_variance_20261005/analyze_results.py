"""Produce a separate interpretation; preserve all original reported bytes."""
from pathlib import Path
import json
import statistics

stage=Path(__file__).resolve().parent
root=stage.parents[1]/'pbit-llm-qwen-baselines/results/and_variance_20261005'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    published=read(stage/'PUBLISHED.json')
    data=read(root/'comparison.json')
    assert data==published['comparison']
    a,b=data['aggregates']['control'],data['aggregates']['variance']
    rows=data['rows']
    diagnostics={(condition,seed):read(root/f'{condition}_seed{seed}/field_diagnostics.json')
                 for seed in (0,1,2) for condition in ('control','variance')}
    layers=sorted(map(int,diagnostics['control',0]['common_teacher_inputs']))
    assert len(layers)==20
    local=[]
    for layer in layers:
        record={'layer':layer}
        for condition in ('control','variance'):
            values=[diagnostics[condition,seed]['common_teacher_inputs'][str(layer)] for seed in (0,1,2)]
            for key in ('bias_nmse','n4_variance_nmse'):
                record[f'{condition}_{key}']=statistics.mean(value[key] for value in values)
        local.append(record)
    empirical=read(root/'empirical_variance.json')
    notes=['# 方差约束实验：解读与下一步','',
        f"预定2%门槛{'达到' if data['prespecified_target_met'] else '未达到'}；相对同预算A的N4 PPL改善{100*data['relative_ppl_improvement']:.3f}%，配对胜出{data['paired_wins']}/3。",
        f"B的三个继续训练seed平均PPL={b['ppl_mean']:.6f}；PPL<13的努力目标{'达到' if b['ppl_mean']<13 else '未达到'}，它不替代预定A/B门槛。",'',
        '## 怎样判断机制','',
        f"完全相同teacher输入的局部Var/4平均减少{100*data['common_input_variance_reduction']:.2f}%；均值偏差NMSE由{a['common_input_bias']:.6f}变为{b['common_input_bias']:.6f}。",
        f"同一输入、同一固定teacher输出尺度下，局部解析均值偏差NMSE+Var/4 NMSE之和，A/B分别为{a['common_input_bias']+a['common_input_n4_variance']:.9f}/{b['common_input_bias']+b['common_input_n4_variance']:.9f}。这同时展示均值与方差的取舍，不是完整网络PPL的误差分解；实际BF16读出还包含量化差异。",
        f"实际student传播输入的局部Var/4 NMSE由{a['actual_input_n4_variance']:.6f}变为{b['actual_input_n4_variance']:.6f}；同入口teacher函数偏差由{a['actual_input_bias']:.6f}变为{b['actual_input_bias']:.6f}。",
        '共同teacher输入便于比较FFN自身变化；实际输入还包含前层产生的分布变化。上述局部量不能精确分解整网PPL，也不能用PPL差值反推出整网输出方差。', '',
        '| 训练seed | A的N4/mean-field PPL | B的N4/mean-field PPL | 配对N4 PPL差 B-A |',
        '| ---: | ---: | ---: | ---: |']
    for seed in (0,1,2):
        first=next(row for row in rows if row['condition']=='control' and row['training_seed']==seed)
        second=next(row for row in rows if row['condition']=='variance' and row['training_seed']==seed)
        notes.append(f"| {seed} | {first['ppl_mean']:.6f}/{first['mean_field_ppl']:.6f} | {second['ppl_mean']:.6f}/{second['mean_field_ppl']:.6f} | {second['ppl_mean']-first['ppl_mean']:+.6f} |")
    notes+=['', 'Mean-field为各FFN使用条件均值的确定性传播，不是完整随机网络的精确期望；它与N4的差距只作诊断。', '',
        '## 相同teacher输入的分层诊断','',
        '以下为三个继续训练seed的平均。层号和数值说明变化位置，不证明某层对PPL的因果贡献。', '',
        '| 层 | A Var/4 NMSE | B Var/4 NMSE | A均值偏差NMSE | B均值偏差NMSE |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for row in local:
        notes.append(f"| {row['layer']} | {row['control_n4_variance_nmse']:.6f} | {row['variance_n4_variance_nmse']:.6f} | {row['control_bias_nmse']:.6f} | {row['variance_bias_nmse']:.6f} |")
    notes+=['', '## 实际二值AND核验与成本','',
        '训练seed0的第12层，在相同128个保留token输入上做128组实际N4采样；该实验不参与任何选择。']
    for label,value in empirical['results'].items():
        notes.append(f"- {label}：实测Var/4 NMSE={value['empirical_n4_variance_nmse']:.6f}，解析值={value['analytical_n4_variance_nmse']:.6f}，实测/解析={value['empirical_over_analytical']:.5f}。")
    notes+=['',f"A/B平均记录运行时间分别{a['seconds']:.1f}/{b['seconds']:.1f}秒，比值{data['training_time_ratio']:.3f}。这包含初始与周期验证、best/latest/final写盘，不能直接当成纯GPU内核开销。",
        '推理仍是K4/N4；没有通过增加推理采样换取质量。浮点投影和FP32读出仍在，未实测物理p-bit速度或能耗。', '',
        '## 后续目标（尚未执行）','']
    if data['prespecified_target_met']:
        notes+=['优先固定K4/N4检验另一语料、下游任务及更长上下文，并保留等预算KL+CE对照。若质量收益跨任务稳定，再测实际驱动、读出精度和物理p-bit相关性；不能由本轮GPU模拟声称硬件加速。']
    elif data['common_input_variance_reduction']>0:
        notes+=['下一轮优先检验“条件均值锚点＋方差约束”：以共同起点的FFN条件均值为固定锚点，在保留验证集重新筛选lambda，检验能否允许更强的降方差而不损害均值。核心对照应是A=KL+CE+beta均值锚点，B=同样A+lambda方差；beta、初始化、可训练参数、K4/N4和总步数两组相同。若要估计锚点单独的收益，另设同预算KL+CE第三组，不能把本轮旧测试结果当作新配对对照。',
                 '先做单层及短联合预检，随后重新进行独立配对A/B；本轮test不用于选择下一轮lambda。若局部方差下降仍不对应PPL改善，再在验证集测层敏感度，用固定层权重替代当前均匀惩罚。不能仅按本表的方差大小断定层重要性。']
    else:
        notes+=['先检查方差项梯度、固定输出尺度、各层实际参数变化和学习率，在短预检中确认惩罚能改变Var/4而保持质量损失有限。确认之后再注册新的同预算A/B；不直接增加推理N。']
    if not data['prespecified_target_met']:
        notes+=['单层筛选的质量项为NMSE+cosine，联合训练为KL+CE，且方差项对20层求平均；两者的数值lambda不具有直接相同的优化强度。下一轮先做短联合保留验证筛选，并分别记录质量项与lambda加权方差项的梯度范数、方向余弦、各层梯度与裁剪前后变化，不只比较损失值大小，也不读取test重新选择lambda。固定x与完整反传两种诊断可区分本层变化与前层输入分布变化；若改变是否detach，应重新注册等规则A/B。']
    notes+=['', '三个seed共享一个从头重训初始化，只重复继续训练；推理seed重复不能充当独立训练样本。本轮单语料和有限训练重复不足以证明普遍收益或精确置信度。','']
    (root/'ANALYSIS_ZH.md').write_text('\n'.join(notes),encoding='utf-8',newline='\n')
    (root/'layer_comparison.json').write_text(json.dumps({'scope':'common teacher inputs; averages over three continuation seeds','layers':local},indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Interpretation generated separately from immutable original reports')

if __name__=='__main__':main()
