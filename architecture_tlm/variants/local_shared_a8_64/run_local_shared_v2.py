"""Full A12/A8 old vs local-share64 comparisons, separate output directory."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

VARIANT_ROOT=Path(__file__).resolve().parent
TLM_ROOT=VARIANT_ROOT.parents[1]
if str(TLM_ROOT) not in sys.path:
    sys.path.insert(0,str(TLM_ROOT))
from system_decode_base import DecodeModel,load,ROOT
from system_decode_local_shared_v2 import LocalSharedDecode,MatchedBaselineDecode


def configs():
    base,evidence,shape=load()
    base['rng']['architecture']='hierarchical_v1'
    base['rng_hierarchy']=json.loads((VARIANT_ROOT/'config/rng_hierarchy_v1.json').read_text())
    source=json.loads((VARIANT_ROOT/'config/local_shared_v2.json').read_text())
    cases=[]
    for name in ('old_A12','old_A8','new_A8_per_tile','new_A8_continuous','new_A8_startup_only_root',
                 'new_A8_periodic_child','new_A8_fresh_entropy','new_A8_root10ms','new_A8_update4pJ'):
        c=copy.deepcopy(base)
        if name!='old_A12':c['rng']['input_bits_per_decision']=8
        c['local_shared_input']=copy.deepcopy(source)
        if name=='new_A8_continuous':c['local_shared_input']['child_seed_policy']='continuous'
        if name=='new_A8_startup_only_root':c['rng_hierarchy']['seed']['reseed_interval_ns']=0
        if name=='new_A8_periodic_child':c['local_shared_input']['child_seed_policy']='periodic'
        if name=='new_A8_fresh_entropy':c['local_shared_input']['child_seed_policy']='fresh_entropy'
        if name=='new_A8_root10ms':c['rng_hierarchy']['seed']['reseed_interval_ns']=10000000
        if name=='new_A8_update4pJ':c['local_shared_input']['update_pJ']=4
        cases.append((name,c))
    return cases,evidence,shape


def report(output):
    all_runs=[]
    for p in output.glob('*.json'):
        d=json.loads(p.read_text())
        if 'case_name' in d:all_runs.append(d)
    table=['# FFN 本地 PRNG、A8 共享64：TLM v2 对比','',
        '24层 Qwen2.5-0.5B，batch1 decode，context=2048，T4096，8个64×64阵列，100MHz，2个token。',
        '默认使用简化 Link 外存；所有表项使用相同外存后端。面积、输入 LFSR 和多个成本系数尚待工艺标定。',
        '这是宏模型估算，不是实际芯片/GPU功耗或实际LFSR驱动的PPL。','',
        '公平口径：新旧两边均计入相同的最终bit-mask到乘法单元分发能量。历史原始指标在JSON的historical_raw_metrics中保留；此公共项不改变收益的绝对差值。','',
        '| FFN权重方案 | 版本/种子策略 | 时间 ms/token | 能量 mJ/token | 参数化面积 mm² | 根刷新次数/2token | 输入子种子次数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    by={}
    for d in sorted(all_runs,key=lambda d:(d['variant'],d['case_index'])):
        by[(d['variant'],d['case_name'])]=d
        m=d['metrics'];rng=d['rng_hierarchy'];stats=rng['stats']
        table.append(f"| {d['variant']} | {d['case_name']} | {m['mean_TPOT_ms']:.3f} | {m['scenario_mJ_per_token']:.3f} | {m['parameterized_area_mm2']:.4f} | {rng['epochs']} | {int(stats.get('input_child_rekeys',0))} |")
    changes={}
    table+=['','## 主对比：A8保持相同，只改变输入供给与共享前端','']
    for variant in ('fixed_sc_exact','dual_sc_sz'):
        if (variant,'old_A8') not in by or (variant,'new_A8_per_tile') not in by:continue
        a=by[(variant,'old_A8')];b=by[(variant,'new_A8_per_tile')]
        am=a['metrics'];bm=b['metrics']
        delta={k:100*(bm[k]/am[k]-1) for k in ('mean_TPOT_ms','scenario_mJ_per_token','parameterized_area_mm2')}
        changes[variant]=delta
        table.append(f"- {variant}：时间 {delta['mean_TPOT_ms']:+.2f}%，能量 {delta['scenario_mJ_per_token']:+.2f}%，面积 {delta['parameterized_area_mm2']:+.2f}%。")
        old_e=a['energy_J'];new_e=b['energy_J'];n=b['config']['decode_tokens']
        table+=['', '| 分项 mJ/token | 修改前 A8 | 修改后 A8共享 |', '|---|---:|---:|']
        keys=[('SC核＋输入前端',['SC_logic_active','SC_shared_input_frontend']),
              ('本地权重PRNG更新',['RNG_local_updates']),
              ('权重选择/比较',['RNG_local_selection_compare']),
              ('scratch读写',['scratch_access']),
              ('输入PRNG更新/分发/状态',['RNG_shared_updates','RNG_shared_distribution',
                'RNG_input_updates','RNG_input_word_distribution','RNG_input_restore','RNG_input_checkpoint',
                'RNG_input_child_expansion','RNG_input_child_bus']),
              ('共同bit-mask末端分发',['input_mask_distribution']),
              ('外存动态及背景',['external_weights','external_KV','external_background'])]
        for label,ks in keys:
            table.append(f"| {label} | {sum(old_e.get(k,0) for k in ks)/n*1000:.4f} | {sum(new_e.get(k,0) for k in ks)/n*1000:.4f} |")
        table.append('')
    table+=['## 实现与计费边界','',
        '- 输入为8位随机字、64路同字比较。保留一份原有阈值/符号前端，没有另加概率寄存器银行；不同输入仍需64个阈值。',
        '- 删除整段随机字SRAM replay；保留32位工作状态和32位起始状态，按输出波次恢复同一输入序列。恢复不等于刷新熵。',
        '- 默认每个输入tile从已有根状态数字派生32位子种子；根熵默认1秒刷新，只在输入tile边界执行，不打断已开始的重放。',
        '- continuous保留持续推进状态；startup_only_root根仅启动刷新（仍每tile数字派生）；periodic_child每1ms数字重键；fresh_entropy每输入tile获取32位新调理熵。',
        '- root10ms只用于根刷新频率敏感性，根刷新会重播种原权重核；权重数据路径、核数、权重事件和更新算法仍相同。',
        '- 权重侧仍为每列4个160-bit输出Bivium、每阵列256个核、共2048个；拒绝采样及权重事件生成代码继承v1。',
        '- 默认1pJ/输入PRNG更新、0.001mm²含工作/检查点状态、1µW漏电，均为可调占位；update4pJ给出敏感性。广播扇出按每bit接收端计费。',
        '- SC输入前端共享到整个同步阵列簇，采用一份旧13位前端的保守功耗代理；未额外索取8位缩减或真实ICG的收益。',
        '- 面积只明确替换8个旧输入Bivium为1个本地模块；未标定的前端删减面积默认不抵扣。scratch仍为1MiB，因为词表logits峰值仍在。',
        '- 2token结果包含冷启动，不代表充分长稳态。根刷新数受实际运行时长影响，不能固定按每token摊播种。',
        '- PPL仅引用此前IID理想输入共享实验，不证明此LFSR或硬件重放相关性已验证；双随机SC核仍沿用原固定权重核功耗代理。','',
        '代码：`system_decode_local_shared_v2.py`、`rng_local_shared_v2.py`；配置：`config/local_shared_v2.json`。',
        '逐情景JSON保存事件、完整能量账本、源代码hash、种子与熵计数、存储清单。','']
    (output/'RESULTS_ZH.md').write_text('\n'.join(table),encoding='utf-8')
    (output/'comparison.json').write_text(json.dumps(dict(changes_pct=changes,
        metrics={f'{v}/{name}':d['metrics'] for (v,name),d in by.items()}),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(changes,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=VARIANT_ROOT/'results/generated')
    parser.add_argument('--small',action='store_true')
    parser.add_argument('--bridge',type=Path)
    parser.add_argument('--dramsys-config',type=Path,default=ROOT/'dramsys/ddr4_2ch_NoPowerDown.json')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases,evidence,shape=configs()
    if args.small:shape.update(num_hidden_layers=1,vocab_size=1024)
    for variant in ('fixed_sc_exact','dual_sc_sz'):
        baseline=None
        for index,(name,c0) in enumerate(cases):
            c=copy.deepcopy(c0)
            if args.small:c['context_tokens']=128
            if args.bridge:
                c['external_memory'].update(backend='dramsys',bridge_executable=str(args.bridge.resolve()),
                    configuration=str(args.dramsys_config.resolve()),max_pending=32,
                    output_prefix=str((args.output/f'{variant}_{name}_dram').resolve()))
            cls=LocalSharedDecode if name.startswith('new_') else MatchedBaselineDecode
            model=cls(c,variant,evidence,shape)
            try:d=model.run()
            finally:
                if hasattr(model.ext,'close'):model.ext.close()
            d.update(case_name=name,case_index=index,actual_shape=shape,small=args.small)
            d['case_config_sha256']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
            d['source_hashes']['variants/local_shared_a8_64/run_local_shared_v2.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            d['source_hashes']['variants/local_shared_a8_64/config/local_shared_v2.json']=hashlib.sha256((VARIANT_ROOT/'config/local_shared_v2.json').read_bytes()).hexdigest()
            if name=='old_A12':baseline=d
            assert baseline['actions']==d['actions'],'LLM arithmetic changed'
            for key,value in baseline['traffic'].items():
                if not key.startswith('random_replay_'):assert d['traffic'].get(key)==value,key
            if variant=='dual_sc_sz':
                for key in ('local_core_updates_expected','weight_bernoulli_outputs','local_generated_bits_expected'):
                    assert baseline['rng_hierarchy']['stats'][key]==d['rng_hierarchy']['stats'][key],key
            (args.output/f'{variant}_{name}.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
            print(variant,name,json.dumps(d['metrics']),flush=True)
    report(args.output)


if __name__=='__main__':main()
