#!/usr/bin/env python3
from pathlib import Path
HERE=Path(__file__).resolve().parent
FILES=['stage09_common.conf','00_minimize.conf','01_relax_nvt_100ps.conf','02_equil_npt_500ps.conf','03_acceptance_npt_1ns.conf']

def directives(path):
    out=[]
    for lineno,raw in enumerate(path.read_text().splitlines(),1):
        line=raw.split('#',1)[0].strip()
        if not line: continue
        p=line.split(); out.append((p[0].lower(),p[1:],lineno))
    return out

def vals(ds,key): return [(args,line) for k,args,line in ds if k==key.lower()]
def req(c,msg):
    if not c: raise SystemExit('ERROR: '+msg)
def one(ds,key,expected=None):
    f=vals(ds,key); req(len(f)==1,f"expected exactly one '{key}', found {len(f)}")
    args,line=f[0]; req(args,f"'{key}' has no value at line {line}")
    if expected is not None: req(args[0].lower()==expected.lower(),f"'{key}' is {args[0]!r}, expected {expected!r} at line {line}")
    return args[0]

def main():
    d={}
    for fn in FILES:
        p=HERE/fn; req(p.is_file(),f'missing {fn}'); d[fn]=directives(p)
        req(not vals(d[fn],'nonbondedFrequency'),f'legacy nonbondedFrequency spelling in {fn}')
    common=d['stage09_common.conf']; one(common,'nonbondedFreq','1'); one(common,'fullElectFrequency','2'); req(not vals(common,'langevin'),'common file must not set langevin')
    for fn in FILES[1:]: one(d[fn],'ambercoor','$stage09_inpcrd')
    mini=d['00_minimize.conf']; one(mini,'langevin','off'); one(mini,'temperature'); one(mini,'minimize','10000')
    nvt=d['01_relax_nvt_100ps.conf']; one(nvt,'langevin','on'); one(nvt,'temperature','310.15'); one(nvt,'langevinTemp','310.15'); one(nvt,'langevinPiston','off'); one(nvt,'run','50000'); req(not vals(nvt,'binvelocities'),'NVT stage must initialize fresh velocities'); req(not vals(nvt,'extendedSystem'),'first NVT stage must use generated accepted-cell vectors directly')
    for key in ('cellBasisVector1','cellBasisVector2','cellBasisVector3','cellOrigin'): one(nvt,key)
    for fn,prev,steps in [('02_equil_npt_500ps.conf','01_relax_nvt_100ps/relax100ps.vel','250000'),('03_acceptance_npt_1ns.conf','02_equil_npt_500ps/equil500ps.vel','500000')]:
        ds=d[fn]; one(ds,'langevin','on'); one(ds,'langevinTemp','310.15'); one(ds,'langevinPiston','on'); one(ds,'langevinPistonTarget','1.01325'); one(ds,'langevinPistonTemp','310.15'); one(ds,'binvelocities',prev); one(ds,'run',steps); req(not vals(ds,'temperature'),f'{fn} must continue velocities')
    one(d['03_acceptance_npt_1ns.conf'],'DCDfreq','1000')
    print('Stage 09 static configuration validation: PASS')
if __name__=='__main__': main()
