"""Figure 3 geometry records"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPECIES = [('PFOA', 'PFOA⁻'), ('PFOS', 'PFOS⁻'),
           ('6-2_FTCA', '6:2 FTCA⁻'), ('PFHxA', 'PFHxA⁻')]
GROUPS = [
    ('A', 'Figure_04_BTMA_exchange_and_structures', 'r2SCAN-3c'),
    ('B', 'Figure_04_BTMA_exchange_and_structures', 'wB97X-D3'),
    ('C', 'Figure_09_model_size_exchange', 'r2SCAN-3c'),
    ('D', 'Figure_09_model_size_exchange', 'wB97X-D3'),
]


def records():
    items = []
    for letter, folder, method in GROUPS:
        base = ROOT / '01_main_text_figures' / folder / 'input_structures'
        for species, label in SPECIES:
            xyz = base / f'{method}_{species}.xyz'
            if not xyz.is_file():
                raise FileNotFoundError(xyz)
            items.append(dict(group=letter, species=label, xyz=xyz, solvent='water'))
    base = ROOT / '01_main_text_figures/Figure_10_PFOA_water_octanol_exchange/input_data'
    if (base / 'water_PFOA_complex.xyz').read_bytes() != items[12]['xyz'].read_bytes():
        raise ValueError('Figure 10 water PFOA is no longer identical to atlas group D')
    xyz = base / 'octanol_PFOA_complex.xyz'
    if not xyz.is_file():
        raise FileNotFoundError(xyz)
    items.append(dict(group='E', species='PFOA⁻', xyz=xyz, solvent='1-octanol'))
    return items
