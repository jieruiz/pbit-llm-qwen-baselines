"""Re-run upstream checks and narrowly probe assumptions without changing source."""
from pathlib import Path
import copy
import importlib.util
import json
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source' / 'architecture_tlm'
if not SOURCE.is_dir():
    SOURCE = ROOT.parents[1] / 'architecture_tlm'
LOGS = ROOT / 'checks'
LOGS.mkdir(exist_ok=True)
checks = []
for relative in ('import_evidence.py', 'checks/test_model.py', 'checks/test_system_decode.py'):
    result = subprocess.run([sys.executable, '-X', 'utf8', str(SOURCE / relative)],
                            cwd=SOURCE, capture_output=True, text=True, encoding='utf-8')
    name = relative.replace('/', '_').replace('.py', '.log')
    (LOGS / name).write_text(result.stdout + result.stderr, encoding='utf-8')
    checks.append({'script': relative, 'exit_code': result.returncode, 'log': f'checks/{name}'})
    if result.returncode:
        print(json.dumps(checks, ensure_ascii=False))
        raise SystemExit(result.returncode)

spec = importlib.util.spec_from_file_location('review_decode', SOURCE / 'system_decode.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cfg, evidence, shape = module.load()
base = []
for length in (2048, 8192):
    for variant in ('digital_bf16', 'digital_int8_ffn', 'fixed_sc_exact', 'dual_sc_sz'):
        configuration = copy.deepcopy(cfg)
        configuration['context_tokens'] = length
        result = module.DecodeModel(configuration, variant, evidence, shape).run()
        stored = json.loads((SOURCE / 'results/system_decode' / f'base_{variant}_{length}.json').read_text(encoding='utf-8'))
        for key, value in result['metrics'].items():
            reference = stored['metrics'][key]
            assert math.isclose(value, reference, rel_tol=1e-12) if isinstance(value, float) else value == reference
        phase_ms = {key: value / cfg['decode_tokens'] / 1e6 for key, value in result['time_ns'].items()}
        attention_ms = phase_ms['QK_softmax_PV']
        total_ms = result['metrics']['mean_TPOT_ms']
        base.append({'variant': variant, 'context_tokens': length,
                     'metrics': result['metrics'], 'phase_ms': phase_ms,
                     'attention_core_time_fraction': attention_ms / total_ms,
                     'speedup_if_attention_core_were_free': total_ms / (total_ms - attention_ms),
                     'traffic': result['traffic'], 'bounds': result['bounds']})

# Upstream attention accounts for local streaming bytes in energy but does not
# reserve scratch bandwidth. Isolate this path from FFN/RNG/replay activity.
bandwidth_probe = []
for bandwidth in (64, 0.064):
    configuration = copy.deepcopy(cfg)
    configuration['scratchpad']['bandwidth_GBs'] = bandwidth
    model = module.DecodeModel(configuration, 'digital_bf16', evidence, shape)
    end = model.attention(0, 8192, 0)
    bandwidth_probe.append({'scratch_bandwidth_GBs': bandwidth, 'end_ns': end,
                            'scratch_KV_tile_bytes': model.traffic['scratch_KV_tile_bytes']})

report = {'upstream_checks': checks, 'base_scenarios_reproduced': len(base),
          'base_scenarios': base, 'attention_scratch_bandwidth_probe': bandwidth_probe,
          'statement': 'Upstream bookkeeping checks and review probes only; no numerical LLM, RTL or hardware validation.'}
(ROOT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'checks': [{'script': c['script'], 'exit_code': c['exit_code']} for c in checks],
                  'base_scenarios_reproduced': len(base), 'scratch_bandwidth_probe': bandwidth_probe}, ensure_ascii=False))
