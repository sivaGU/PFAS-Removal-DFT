"""Figure 9 molecular assets"""
from pathlib import Path
import importlib.util

HERE=Path(__file__).resolve().parent
RENDER=HERE.parent/'Figure_10_PFOA_water_octanol_exchange/render_figure10_structures.py'
spec=importlib.util.spec_from_file_location('figure10_renderer',RENDER)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
ORDER=('PFOA','PFOS','6-2_FTCA','PFHxA')

def main():
    for method in ('r2SCAN-3c','wB97X-D3'):
        for species in ORDER:
            src=HERE/'input_structures'/f'{method}_{species}.xyz'
            if not src.exists():raise FileNotFoundError(src)
            mod.render_xyz(src,HERE/'structure_renders'/f'{method}_{species}.png',dpi=1200)

if __name__=='__main__':main()
