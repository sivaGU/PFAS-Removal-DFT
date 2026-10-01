#!/usr/bin/env python3

from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path


def charges(path):
    out=[]; section=None
    for line in Path(path).read_text(errors='replace').splitlines():
        if line.startswith('@<TRIPOS>'):
            section=line.strip(); continue
        if section=='@<TRIPOS>ATOM' and line.strip():
            f=line.split(); out.append(float(f[-1]))
    if not out: raise ValueError(f'No MOL2 charges found in {path}')
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rct-mol2',type=Path,default=Path('pVBTMA12_gaff2_rct.mol2'))
    ap.add_argument('--archived-mol2',type=Path,default=Path('whole12_direct_am1bcc_reference/pVBTMA12_gaff2_am1bcc.mol2'))
    ap.add_argument('--reorder-map',type=Path,default=Path('pVBTMA12_accepted_to_rct_reorder.json'))
    ap.add_argument('--atom-map',type=Path,default=Path('pVBTMA12_rct_atom_mapping.json'))
    ap.add_argument('--json-out',type=Path,default=Path('rct_vs_direct_whole12.json'))
    a=ap.parse_args()
    if not a.archived_mol2.exists():
        raise SystemExit(
            f'Direct whole-12-mer reference not found: {a.archived_mol2}. '
            'Run stage_whole12_reference.py first.'
        )
    qnew=charges(a.rct_mol2); qold_accept=charges(a.archived_mol2)
    reorder=json.loads(a.reorder_map.read_text())['builder_to_accepted_1based']
    amap=json.loads(a.atom_map.read_text())['atoms']
    if len(qnew)!=len(amap) or len(qold_accept)!=len(amap): raise SystemExit('Atom-count mismatch')
    qold=[qold_accept[int(reorder[str(i+1)])-1] for i in range(len(amap))]
    diffs=[x-y for x,y in zip(qnew,qold)]
    def metrics(indices):
        ds=[diffs[i] for i in indices]
        return {
            'n':len(ds),
            'mae_e':statistics.fmean(abs(x) for x in ds),
            'rmse_e':math.sqrt(statistics.fmean(x*x for x in ds)),
            'max_abs_e':max(abs(x) for x in ds),
            'mean_signed_e':statistics.fmean(ds),
        }
    groups={
        'all':list(range(len(amap))),
        'head':[i for i,x in enumerate(amap) if x['repeat_kind']=='head'],
        'internal':[i for i,x in enumerate(amap) if x['repeat_kind']=='internal'],
        'tail':[i for i,x in enumerate(amap) if x['repeat_kind']=='tail'],
        'quaternary_ammonium_N':[i for i,x in enumerate(amap) if x['role']=='Nq'],
    }
    report={
        'purpose':(
            'Diagnostic validation of the RCT charge-transfer approximation against a direct '
            'whole-pVBTMA12 AM1-BCC calculation; the direct reference is not used to derive RCT charges '
            'and is not an independent gold standard for AM1-BCC accuracy.'
        ),
        'rct_mol2':str(a.rct_mol2),
        'direct_whole12_mol2':str(a.archived_mol2),
        'rct_charge_sum':sum(qnew),
        'direct_whole12_charge_sum':sum(qold),
        'metrics':{k:metrics(v) for k,v in groups.items()},
    }
    a.json_out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__': main()
