import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
F=ROOT/'results/system_decode'


def main():
    rows=json.loads((F/'summary.json').read_text())
    names={'digital_bf16':'数字BF16','digital_int8_ffn':'数字FFN仅权重量化INT8',
           'fixed_sc_exact':'固定权重随机FFN/精确计数','dual_sc_sz':'双随机固定Sz FFN'}
    base={r['variant']:r for r in rows if r['parameter']=='baseline' and r['context']==2048}
    results={v:json.loads((F/(r['name']+'.json')).read_text()) for v,r in base.items()}
    lines=['权重宏抽象与完整Qwen2.5-0.5B decode：时间/能耗限制分析','2026-10-14','',
        '模型覆盖24层、准确稠密attention、标准归一化/RoPE/残差/SiLU、词嵌入和lm_head，batch1，连续2个decode token。',
        '权重存储按宏容量、加载带宽、读写能量、保持功耗和面积收费，不再模拟576个保持DFF。',
        '当前是可运行Python宏级LT事务模型；SystemC/TLM-2.0、AT和精确DRAM时序没有实现。',
        '下面数值为可替换参数下的情景估算；不代表已测ASIC，也不是GPU比较。样本精度、宏成本、共享前端与数字吞吐均需进一步标定。','',
        '一、最重要的发现','',
        '1. 固定权重随机FFN：主要时间由T4096和有限阵列并行度决定，换更快外存改善有限。',
        '2. 双随机固定Sz：独立权重事件若通过均匀随机字+比较器产生，随机源供给主导时间；加计算阵列几乎无效。',
        '3. 数字基线：权重搬运是主要成本，INT8仅存FFN权重已经有明显的搬运收益，可作为随机方案的重要对照。',
        '4. SRAM保持功耗需扫描，但默认小容量下，单独改保持单价不会解除上述时间限制。',
        '5. 长时间等待会放大SRAM漏电、数字空闲和外存背景能量；不能只比较AND与MAC动态能量。','',
        '二、总括表：默认配置','',
        '默认：100MHz；512数字MAC/拍；8个64x64随机阵列；32个已报告随机字源；权重SRAM16MiB、scratch1MiB；外存32GB/s。',
        '每个权重tile只装载后保持，不在4096拍内重新读取全部权重。原始输入随机字先生成，再按输入tile广播/重放，保持跨输出共享关系。',
        '方案 | 2K ms/token | 8K ms/token | 2K 情景mJ/token | 8K 情景mJ/token']
    for v,label in names.items():
        a=base[v];b=next(r for r in rows if r['variant']==v and r['parameter']=='baseline' and r['context']==8192)
        lines.append(f"{label} | {a['mean_TPOT_ms']:.2f} | {b['mean_TPOT_ms']:.2f} | {a['scenario_mJ_per_token']:.2f} | {b['scenario_mJ_per_token']:.2f}")
    lines+=['','表中数字基线也是ASIC参数模型，不是GPU测量。不同方案资源数/面积不同，不能视为严格同面积比赛。',
            '双随机行的核心功耗未定标，默认采用可调代理；巨大时间来自供给需求，能量绝对值尤其依赖等待/背景功耗假设。','',
            '三、2K时间分解：每项包含自己阶段的依赖/等待，不可再加RNG服务时间','',
            '方案 | FFN ms/token | 完整attention与投影 ms/token | lm_head ms/token | 其他 ms/token']
    for v,d in results.items():
        t=d['time_ns'];n=d['config']['decode_tokens']*1e6
        ffn=sum(t.get(k,0) for k in ('FFN_gate_up','FFN_down','FFN_SwiGLU'))/n
        att=sum(t.get(k,0) for k in ('QKV_projection','QK_softmax_PV','output_projection'))/n
        lm=t['lm_head']/n
        total=d['metrics']['mean_TPOT_ms']
        lines.append(f"{names[v]} | {ffn:.2f} | {att:.2f} | {lm:.2f} | {total-ffn-att-lm:.2f}")
    lines+=['','四、2K能量分解：情景mJ/token','',
            '方案 | 外存权重动态 | 随机核心有效采样 | 随机源 | SRAM/scratch动态+保持 | 其他计算及等待/背景']
    for v,d in results.items():
        n=d['config']['decode_tokens'];e=d['energy_J']
        a=e.get('external_weights',0)*1000/n
        b=e.get('SC_logic_active',0)*1000/n
        c=e.get('RNG_sources',0)*1000/n
        k=('weight_sram_access','weight_resident_prepare','weight_sram_leakage','weight_macro_hold','scratch_access','scratch_leakage')
        m=sum(e.get(key,0) for key in k)*1000/n
        rest=sum(e.values())*1000/n-a-b-c-m
        lines.append(f"{names[v]} | {a:.2f} | {b:.2f} | {c:.2f} | {m:.2f} | {rest:.2f}")
    lines+=['','五、最有解释力的参数变化','',
            '方案 | 只改变的参数 | 新值 | 时间相对默认 | 能量相对默认 | 面积预算']
    interesting={('sc_engines',32),('sc_engines',64),('rng.sources',1024),('rng.sources',32768),
                 ('external_memory.bandwidth_GBs',128),('weight_macro.cache_MiB',1024),
                 ('sc_cost.core_multiplier',.1),('sc_profile','g1_case1')}
    for r in rows:
        if (r['parameter'],r['value']) not in interesting:continue
        b=base[r['variant']]
        lines.append(f"{names[r['variant']]} | {r['parameter']} | {r['value']} | {r['mean_TPOT_ms']/b['mean_TPOT_ms']:.3f}x | {r['scenario_mJ_per_token']/b['scenario_mJ_per_token']:.3f}x | {'满足假设预算' if r['area_budget_feasible'] else '超出假设预算'}")
    lines+=['','这些是单因素扫描，不代表最优配置。加阵列/随机源会改变面积、保持功耗和空闲成本。默认常开随机源增大后可能产生大量不用的输出。',
        '默认数字基线的16MiB SRAM无法保留整个模型。4/64/512MiB在这条循环LRU序列中仍可能几乎全重载；1024MiB能出现warm token复用，但按当前面积系数超过100mm2预算。',
        '这不是增大SRAM永远无效：固定常驻重要矩阵、batch复用或不同调度需要另建策略。','',
        '六、随机源账本为什么会成为瓶颈','',
        '固定权重：每token约6.795G原始随机位；输入特征的12位随机字可跨输出tile重放。',
        '双随机：每token313,786,368个FFN乘积×4096次，约1.285万亿权重Bernoulli决策，再按独立均匀字编码约10.36T原始随机位。',
        '32个源的名义总供给为230.4Gbit/s，单看双随机权重所需原始位便约45秒/token。',
        '这里明确是均匀随机字编码架构；原生可调概率p-bit不能直接套用这些位数/功耗。',
        'DS-CIM共享PRNG/互斥重映射也改变相关性与算法，不能在模型里免费减少独立流并沿用现有PPL。',
        '注意：计算服务、RNG服务与memory预取存在重叠，bounds中的服务时间用于诊断，不是可以相加的总时间。','',
        '七、可调SRAM/权重宏参数','',
        'weight_macro.cache_MiB：可驻留容量；load_bandwidth_GBs/load_latency_ns：装载接口。',
        'read_pJ_per_byte/write_pJ_per_byte：SRAM动作能量；leakage_W_per_MiB：驻留保持。',
        'area_mm2_per_MiB：整体宏面积系数；active_tile_hold_W/active_tile_area_mm2：参与计算tile的保持费用。',
        'local_resident_bits_per_tile只记录接口驻留容量，不据此另收费DFF；输出保持结构由宏提供。',
        'resident_read_pJ_per_tile：tile准备动作；替换真实宏参数时要确认它是否已包含加载成本，避免重复。',
        'SRAM读写/面积/时序均是默认情景参数，不能把当前单价当作真实55nm宏。','',
        '八、质量和能耗不能错配','',
        '固定权重随机主模型采用准确SiLU(g)*u后直接Sz编码，未独立补测此固定权重变体的PPL，quality_reference_ppl为null。',
        '双随机Sz可参考已有条件高斯数值PPL：4096次13.95，1024次42.76，256次约12.8万；不把降低T的成本结果解释为质量保持。',
        '真实硬件均匀字相关性、权重编码和累加范围仍需回到Python验证；2K数值PPL也不能当作8K上下文质量实测。',
        '本次不引入注意力采样、Ising或筛选；因此无需稀疏索引trace，保留准确QK/softmax/PV及GQA共享。','',
        '九、模型验证与下一步','',
        f'本次完成{len(rows)}个完整decode情景，11项形状/事务/计费检查通过。每配置保存输入、事件时间、流量、成本、面积、源hash。',
        '已检查数字MAC形状、GQA不重复读KV、SC乘积样本数、输入流跨输出共享、双随机权重需求、warm cache、能量求和、阶段时间不重叠和宏参数单因素效应。',
        '优先补齐真实权重宏及G1矩阵集成成本、LLM实际稀疏活动和stall功耗；同时考察更少采样与原生概率源，修改协议后重新测试质量。',
        '模型为宏级LT，外存采用带宽/延迟模型，未实现bank/row/AT队列，故随机访存微细时序不可由本结果外推。',
        'prefill和TTFT尚未计算；外存PHY/封装、完整CTS、源唤醒、host/整板排除。所有energy_complete_measured=false。',
        '', '文件：configs/system_decode_default.json、configs/cost_provenance.json、system_decode.py、run_system_sweep.py。',
        '数据：results/system_decode/summary.csv、summary.json、sensitivity.json与逐情景JSON。']
    (F/'完整decode宏模型与瓶颈分析.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print(json.dumps(dict(scenarios=len(rows),report=str(F/'完整decode宏模型与瓶颈分析.txt')),ensure_ascii=False))


if __name__=='__main__':main()
