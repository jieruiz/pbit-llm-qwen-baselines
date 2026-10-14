"""Verify bundled calibration inputs. No external workspace dependency."""
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def digest(data):
    return hashlib.sha256(data).hexdigest()

def build():
    contract=json.loads((ROOT/'configs/evidence_contract.json').read_text(encoding='utf-8'))
    prov=contract['provenance']
    archive=ROOT/prov['zip']
    assert digest(archive.read_bytes())==prov['zip_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in prov['checked_zip_matches']:
            assert digest(z.read('sc_weight_storage_or4_20261014/'+p['path']))==p['sha256']
    source=ROOT/'evidence/grouped_38015538'
    for filename,key in [('totals.csv','totals_sha256'),('breakdown.csv','breakdown_sha256')]:
        assert digest((source/filename).read_bytes())==prov[key]
    with (source/'totals.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    for key,p in contract['profiles'].items():
        row=next(r for r in rows if int(r['group'])==p['group'] and int(r['batch'])==p['batch'] and int(r['case'])==p['case'])
        assert math.isclose(float(row['total_mW'])/1000,p['total_W'],rel_tol=1e-12)
        assert math.isclose(float(row['energy_pJ_per_frame'])/1000,p['energy_nJ'],rel_tol=1e-12)
    return contract

if __name__=='__main__':
    c=build()
    print(json.dumps(dict(status='PASS',profiles=len(c['profiles']),bundled_evidence=True)))
