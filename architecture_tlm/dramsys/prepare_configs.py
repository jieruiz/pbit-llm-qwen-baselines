"""Snapshot upstream DDR4 specs; 2 channels, 64B channel/bank interleave."""
import argparse,copy,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('upstream');a=p.parse_args()
root=Path(__file__).resolve().parent
up=Path(a.upstream)/'configs'
mem=json.loads((up/'memspec/MICRON_4Gb_DDR4-1866_8bit_A.json').read_text())['memspec']
mem['memarchitecturespec']['nbrOfChannels']=2
mapping=dict(BYTE_BIT=[0,1,2],COLUMN_BIT=[3,4,5]+list(range(11,18)),
    CHANNEL_BIT=[6],BANK_BIT=[7,8],BANKGROUP_BIT=[9,10],ROW_BIT=list(range(18,33)))
mc=json.loads((up/'mcconfig/fr_fcfs.json').read_text())['mcconfig']
mc.update(Arbiter='Fifo',MaxActiveTransactions=64,RequestBufferSize=8)
sim=json.loads((up/'simconfig/example.json').read_text())['simconfig']
sim.update(DatabaseRecording=False,PowerAnalysis=True,EnableWindowing=False,
           SimulationProgressBar=False,Debug=False,SimulationName='pbit_decode_ddr4')
for mode in ('NoPowerDown','Staggered'):
 c=copy.deepcopy(mc);c['PowerDownPolicy']=mode
 config=dict(simulation=dict(simulationid='pbit_decode_ddr4',addressmapping=mapping,
    mcconfig=c,memspec=mem,simconfig=sim))
 (root/f'ddr4_2ch_{mode}.json').write_text(json.dumps(config,indent=2)+'\n')
metadata=dict(upstream_commit='9f15aaf1fe6f05ff16a2e521a20a96f3bfe29d52',
    dramsys_version='5.6.0',dramutils_version='1.16.0',drampower_version='6.3.1',systemc_version='3.0.2',
    memspec_origin='configs/memspec/MICRON_4Gb_DDR4-1866_8bit_A.json',
    memspec_source_sha256=hashlib.sha256((up/'memspec/MICRON_4Gb_DDR4-1866_8bit_A.json').read_bytes()).hexdigest(),
    channels=2,capacity_bytes=8*1024**3,
    peak_bandwidth_GBs=2*8*2/(mem['memtimingspec']['tCK']*1e9),
    power_warning='Bundled vddq is zero and impedance fields are examples: interface energy is NOT calibrated. Core uses bundled current coefficients; not measured full system.',
    address_mapping='64B channel interleave, 128B bank interleave; fixed complete 33-bit mapping')
(root/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
