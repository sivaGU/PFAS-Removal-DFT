"""Publish the curated structural rendering used as manuscript Figure 14.

The molecular source structures and MD outputs are archived under
Supplementary_Materials/01_Calculations/05_MD.  The curated rendering is kept
here because camera placement, transparency, and labels are part of the figure
design and are not numerical analysis steps.
"""

from pathlib import Path
import shutil

from PIL import Image


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "input_render" / "Figure_14_MD_structural_snapshots.png"
OUTPUT_DIR = HERE / "outputs"
OUTPUT = OUTPUT_DIR / "Figure_14_MD_structural_snapshots.png"
EXPECTED_SIZE = (7083, 9740)


def make_figure() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing curated structural rendering: {SOURCE}")
    with Image.open(SOURCE) as image:
        if image.size != EXPECTED_SIZE:
            raise ValueError(f"Expected {SOURCE.name} dimensions {EXPECTED_SIZE}, found {image.size}")
        image.verify()

    OUTPUT_DIR.mkdir(exist_ok=True)
    shutil.copyfile(SOURCE, OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    make_figure()
