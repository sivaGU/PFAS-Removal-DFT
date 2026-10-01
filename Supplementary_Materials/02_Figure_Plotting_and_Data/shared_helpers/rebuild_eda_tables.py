"""ORCA EDA table extraction"""
import argparse
import csv
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FACTOR = 627.509474
TERMS = ("Pauli Energy", "Electrostatic Energy", "Orbital Energy",
         "Delta Dispersion", "Delta E^0(XC)", "Delta gCP correction",
         "Delta CPCM Dielectric")

def extract(block, label, default=None):
    matches = re.findall(r"^\s*"+re.escape(label)+r"\s*[:=]?\s*([-+]?\d+\.\d+)", block, re.M)
    if not matches:
        if default is not None: return default
        raise ValueError(f"Missing output term {label}")
    return float(matches[0])

def main(archive):
    rows = []
    with zipfile.ZipFile(archive) as z:
        files = sorted(n for n in z.namelist() if '/03_EDA/' in n and n.endswith('.out')
                       and not re.search(r'_frag[12]\.out$', n))
        if len(files) != 14: raise ValueError(f"Expected 14 original EDA outputs, found {len(files)}")
        for file in files:
            output = z.read(file).decode(errors='replace')
            if 'ORCA TERMINATED NORMALLY' not in output: raise ValueError(f"Incomplete: {file}")
            block = output.split('Energy Decomposition Analysis',1)[1].split('NOCV analysis',1)[0]
            if block.count('Delta CPCM Dielectric') != 2: raise ValueError(f"Unexpected CPCM printing: {file}")
            pfas = next((p for p in ('FHEA','PFHxA','PFOA','PFOS') if p in file), None)
            series = ('BTMA_r2SCAN-3c' if '/01_BTMA/' in file and '/r2scan-3c/' in file else
                      'BTMA_wB97X-D3' if '/01_BTMA/' in file else
                      'DVB_BTMA_Octanol' if '/octanol_72.5/' in file else 'DVB_BTMA_Water')
            terms = {term: extract(block,term,0 if term=='Delta gCP correction' else None)*FACTOR
                     for term in TERMS}
            bond = extract(block,'Bond Energy')*FACTOR
            frag = [file[:-4]+f'_frag{i}.out' for i in (1,2)]
            finals = [extract(z.read(p).decode(errors='replace'),'FINAL SINGLE POINT ENERGY') for p in frag]
            final = extract(output,'FINAL SINGLE POINT ENERGY')
            if abs((final-sum(finals))*FACTOR-bond) > .01:
                raise ValueError(f"Fragment reference does not reconstruct Bond Energy: {file}")
            cds = extract(output,'SMD CDS (Gcds)')*FACTOR
            subtotal = sum(terms.values())
            row = {'PFAS':pfas,'series':series,'Bond Energy':bond,
                   **terms,'Sum of listed terms':subtotal,
                   'Source ORCA output':file}
            rows.append((row,cds))
    fields=['PFAS','series','Bond Energy',*TERMS,'Sum of listed terms','Source ORCA output']
    figroot=ROOT/'01_main_text_figures'
    for folder, selected in (('Figure_07_BTMA_energy_decomposition_analysis',{'BTMA_r2SCAN-3c','BTMA_wB97X-D3'}),
                              ('Figure_11_DVB_BTMA_energy_decomposition_analysis',{'BTMA_wB97X-D3','DVB_BTMA_Water'})):
        path=figroot/folder/'input_data/eda_components.csv'
        with path.open('w',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=fields);w.writeheader()
            for pfas in ('FHEA','PFHxA','PFOA','PFOS'):
                for series in sorted(selected):
                    w.writerow(next(r for r,cds in rows if r['PFAS']==pfas and r['series']==series))
        print(path)
    report=ROOT/'eda_source_audit.csv'
    with report.open('w',newline='') as fh:
        auditfields=['PFAS','series','Bond Energy','Sum of listed terms',
                     'Bond minus listed','Complex SMD CDS','Residual minus complex CDS',
                     'Source ORCA output']
        w=csv.DictWriter(fh,fieldnames=auditfields);w.writeheader()
        for row,cds in rows:
            difference=row['Bond Energy']-row['Sum of listed terms']
            w.writerow({'PFAS':row['PFAS'],'series':row['series'],
                        'Bond Energy':row['Bond Energy'],
                        'Sum of listed terms':row['Sum of listed terms'],
                        'Bond minus listed':difference,'Complex SMD CDS':cds,
                        'Residual minus complex CDS':difference-cds,
                        'Source ORCA output':row['Source ORCA output']})
    print(report)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    main(parser.parse_args().archive)
