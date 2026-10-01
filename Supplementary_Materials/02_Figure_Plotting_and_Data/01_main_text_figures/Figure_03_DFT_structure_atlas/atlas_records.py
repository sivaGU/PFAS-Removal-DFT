"""Figure 3 geometry records"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECIES = [('PFOA', 'PFOA⁻'), ('PFOS', 'PFOS⁻'),
           ('6-2_FTCA', '6:2 FTCA⁻'), ('PFHxA', 'PFHxA⁻')]
GROUPS = [
    ('A', 'BTMA_water', 'r2SCAN-3c'),
    ('B', 'BTMA_water', 'wB97X-D3'),
    ('C', 'DVB-BTMA_water', 'r2SCAN-3c'),
    ('D', 'DVB-BTMA_water', 'wB97X-D3'),
]


def records():
    items = []
    for letter, folder, method in GROUPS:
        base = HERE / 'input_structures' / folder
        for species, label in SPECIES:
            xyz = base / f'{method}_{species}.xyz'
            if not xyz.is_file():
                raise FileNotFoundError(xyz)
            items.append(dict(group=letter, species=label, xyz=xyz, solvent='water'))
    base = HERE / 'input_structures' / 'DVB-BTMA_octanol'
    xyz = base / 'octanol_PFOA_complex.xyz'
    if not xyz.is_file():
        raise FileNotFoundError(xyz)
    items.append(dict(group='E', species='PFOA⁻', xyz=xyz, solvent='1-octanol'))
    return items
