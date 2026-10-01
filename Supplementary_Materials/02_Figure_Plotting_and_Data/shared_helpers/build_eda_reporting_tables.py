#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import math

ROOT=Path(__file__).resolve().parent.parent
PFAS=('FHEA','PFHxA','PFOA','PFOS')
LABEL={'FHEA':'6:2 FTCA',**{p:p for p in PFAS if p!='FHEA'}}
TERMS=('Pauli Energy','Electrostatic Energy','Orbital Energy','Delta Dispersion',
       'Delta E^0(XC)','Delta gCP correction','Delta CPCM Dielectric')

def load(folder):
    with (ROOT/'01_main_text_figures'/folder/'input_data/eda_components.csv').open() as f:
        return list(csv.DictReader(f))

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    selected={}
    sources=[(load('Figure_07_BTMA_energy_decomposition_analysis'),'water'),
             (load('Figure_11_DVB_BTMA_energy_decomposition_analysis'),'water'),
             (load('Figure_12_PFOA_water_octanol_EDA'),None)]
    for rows,solvent in sources:
        for source in rows:
            pfas=source.get('PFAS','PFOA')
            series=source.get('series','DVB_BTMA_Water')
            model='BTMA⁺' if series.startswith('BTMA_') else 'DVB-BTMA⁺'
            method='r²SCAN-3c' if series=='BTMA_r2SCAN-3c' else 'ωB97X-D3'
            state=source.get('solvent',solvent)
            key=(pfas,model,method,state)
            values={term:float(source.get(term,0)) for term in TERMS}
            if not all(math.isfinite(x) for x in values.values()):raise ValueError(key)
            total=math.fsum(values.values());bond=float(source['Bond Energy'])
            if abs(total-float(source['Sum of listed terms']))>1e-7:
                raise ValueError(f'Listed-term sum mismatch {key}')
            row={'PFAS':LABEL[pfas],'model':model,'method':method,'solvent':state,
                 'Pauli_kcal_mol':values[TERMS[0]],'electrostatic_kcal_mol':values[TERMS[1]],
                 'orbital_kcal_mol':values[TERMS[2]],'dispersion_kcal_mol':values[TERMS[3]],
                 'XC_kcal_mol':values[TERMS[4]],'gCP_kcal_mol':values[TERMS[5]] if method=='r²SCAN-3c' else '',
                 'gCP_separately_printed':method=='r²SCAN-3c','CPCM_kcal_mol':values[TERMS[6]],
                 'Bond_Energy_kcal_mol':bond,'component_sum_kcal_mol':total,
                 'Bond_minus_component_sum_kcal_mol':bond-total,
                 'source_output':source['Source ORCA output']}
            if key in selected:
                previous=selected[key]
                if any(previous[c]!=row[c] for c in row if c!='source_output'):
                    raise ValueError(f'Duplicate plotted EDA record differs: {key}')
            else:selected[key]=row
    if len(selected)!=13:raise ValueError(f'Expected thirteen current EDA records, found {len(selected)}')
    records=sorted(selected.values(),key=lambda r:(list(LABEL.values()).index(r['PFAS']),r['model'],r['method'],r['solvent']))
    output=ROOT/'03_reporting_tables/EDA'


    active_audit=ROOT/'eda_source_audit.csv'
    legacy_audit=output/'provenance/legacy_eda_source_audit_pre_patch.csv'
    if active_audit.exists() and not legacy_audit.exists():
        with active_audit.open() as f:
            previous=list(csv.DictReader(f))
        if any('octanol_72.5' in r.get('Source ORCA output','') for r in previous):
            legacy_audit.parent.mkdir(parents=True,exist_ok=True)
            legacy_audit.write_bytes(active_audit.read_bytes())
    write(active_audit,records)
    write(output/'current_eda_full_precision.csv',records)
    columns=['PFAS','model','method','solvent','Pauli_kcal_mol','electrostatic_kcal_mol',
             'orbital_kcal_mol','dispersion_kcal_mol','XC_kcal_mol','gCP_kcal_mol','CPCM_kcal_mol','Bond_Energy_kcal_mol']
    write(output/'main_eda_table.csv',[{k:r[k] for k in columns} for r in records])
    write(output/'supplementary_S3_eda_table.csv',records)
    text=['# Main EDA table content','',
          'All energies are in kcal/mol. DVB-BTMA⁺ denotes the extended resin fragment.',
          'Calculations use CPCM/SMD with the solvent indicated. A dash denotes gCP not separately reported in the ωB97X-D3 outputs.','']
    numeric=columns[4:]
    for title,subset in [
        ('(a) BTMA⁺ ion pairs in water',[r for r in records if r['model']=='BTMA⁺']),
        ('(b) DVB-BTMA⁺ ion pairs in water at ωB97X-D3',[r for r in records if r['model']=='DVB-BTMA⁺' and r['solvent']=='water']),
        ('(c) DVB-BTMA⁺–PFOA⁻ in water and 1-octanol at ωB97X-D3',[r for r in records if r['model']=='DVB-BTMA⁺' and r['PFAS']=='PFOA'])]:
        text += ['## '+title,'','| PFAS | Method/solvent | Pauli | Electrostatic | Orbital | Dispersion | XC | gCP | CPCM | Bond Energy |',
                 '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in subset:
            tag=r['method'] if r['model']=='BTMA⁺' else r['solvent']
            fields=[r['PFAS'],tag]+['—' if r[k]=='' else f'{float(r[k]):.2f}' for k in numeric]
            text.append('| '+' | '.join(fields)+' |')
        text.append('')
    text += ['Bond Energy is the ORCA endpoint relative to frozen fragments. The component sum and Bond Energy are tabulated separately in Table S3.','']
    (output/'main_eda_table.md').write_text('\n'.join(text))
    (output/'table_summary.json').write_text(json.dumps({'unique_records':13,'main_table_display_rows':14,
        'covers_figures':[7,11,12],'energy_unit':'kcal/mol','preparation_energy_column':False},indent=2)+'\n')
    print(f'Wrote thirteen current EDA records and main-table parts to {output}')

if __name__=='__main__':main()
