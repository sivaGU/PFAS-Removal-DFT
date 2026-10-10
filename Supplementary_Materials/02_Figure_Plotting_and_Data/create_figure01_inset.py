"""Export the first two corrected legacy EDA panels for the BioRender workflow inset."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from shared_helpers.eda_ladders import plot_eda
out=Path(__file__).resolve().parent/'figure_handoff';out.mkdir(exist_ok=True)
closed=[]
original_close=plt.close
def capture(fig=None):
 if hasattr(fig,'axes'):closed.append(fig)
plt.close=capture
plot_eda(root/'01_main_text_figures/Figure_07_BTMA_energy_decomposition_analysis/input_data/eda_components.csv',('BTMA_r2SCAN-3c','BTMA_wB97X-D3'),True,out/'Figure_01_EDA_full_corrected.png')
fig=closed[-1];fig.canvas.draw();renderer=fig.canvas.get_renderer()
boxes=[ax.get_tightbbox(renderer) for ax in fig.axes[:2]]+[l.get_window_extent(renderer) for l in fig.legends]
bbox=Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
fig.savefig(out/'Figure_01_EDA_inset_corrected.png',dpi=600,bbox_inches=bbox.padded(.035),facecolor='white')
plt.close=original_close;plt.close(fig)
print(out/'Figure_01_EDA_inset_corrected.png')
