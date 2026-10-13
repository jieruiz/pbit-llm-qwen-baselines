"""Architecture-level per-decode-token operation counts, not GPU kernel timing/energy."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
D,F,H,DK,DV,L=1024,3584,16,128,128,18
K,V=H*DK,H*DV
projection={'qkv':D*(2*K+V),'z':D*V,'a_and_b':D*H*2,'out':V*D}
state=H*DK*DV
delta_mac=(sum(projection.values())+3*state+(2*K+V)*4)*L
report={'dimensions':{'hidden':D,'ffn_intermediate':F,'heads':H,'key_dim':DK,'value_dim':DV,'delta_layers':L},
    'per_delta_layer_projection_MAC':projection,
    'all18':{'projection_MAC':sum(projection.values())*L,'state_read_update_MAC':3*state*L,
        'state_decay_scalar_MUL':state*L,'conv_MAC':(2*K+V)*4*L,
        'delta_total_MAC':delta_mac,'state_elements':state*L,'state_MiB_FP32':state*L*4/2**20},
    'scope':'One decoded token, batch1, direct recurrent formulation. MAC=one multiplication and addition. Decay MUL separate. Excludes activation/norm/softmax/residual/control and memory-access energy. Prefill chunkwise overhead differs.'}
rows=[]
for context in [2048,8192]:
    ffn=24*3*D*F
    full_projection=6*D*(2*8*256+2*2*256+8*256)
    qkpv=6*2*8*256*context
    total=ffn+delta_mac+full_projection+qkpv
    rows.append({'context':context,'ffn_MAC':ffn,'delta_total_MAC':delta_mac,
        'full_attention_projection_MAC':full_projection,'full_attention_QK_PV_MAC':qkpv,
        'body_MAC':total,'delta_body_MAC_fraction':delta_mac/total,
        'delta_state_core_body_MAC_fraction':3*state*L/total,
        'LM_head_MAC':248320*D,'full_attention_KV_MiB_BF16':context*6*2*2*256*2/2**20})
report['decode_comparisons']=rows
(ROOT/'deltanet_operation_counts.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
