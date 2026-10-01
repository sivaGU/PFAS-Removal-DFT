"""Figure 11 DVB-BTMA EDA"""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
from shared_helpers.eda_ladders import plot_eda
from shared_helpers.figure_style import EXPORT_DPI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT.parent.parent / "figure_exports")
    parser.add_argument("--dpi", type=int, default=EXPORT_DPI)
    args = parser.parse_args()
    plot_eda(ROOT / "input_data" / "eda_components.csv",
             ("BTMA_wB97X-D3", "DVB_BTMA_Water"), False,
             args.output_dir / "Figure_11_DVB_BTMA_eda_across_pfas.png",
             dpi=args.dpi, legend_labels=('BTMA⁺', 'DVB-BTMA⁺'))
