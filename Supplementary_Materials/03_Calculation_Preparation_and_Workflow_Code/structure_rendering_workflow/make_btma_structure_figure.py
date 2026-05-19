#!/usr/bin/env python3
"""Create a four-panel PyMOL structure figure for BTMA-PFAS optimized complexes."""

from __future__ import annotations

import tempfile
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import render_structure_3d as render


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "final_plots" / "structure_figures"
CONTACT_COLORS = ["#d62728", "#1f77b4", "#2ca02c"]
FINAL_FILES = [
    "BTMA_PFAS_optimized_structures_orthoscopic.png",
    "BTMA_PFAS_optimized_structures_orthoscopic_no_panel_letters.png",
    "BTMA_PFAS_optimized_structures_perspective.png",
    "BTMA_PFAS_optimized_structures_perspective_no_panel_letters.png",
    "BTMA_PFAS_wB97X-D3_structures_orthoscopic.png",
    "BTMA_PFAS_wB97X-D3_structures_orthoscopic_no_panel_letters.png",
    "BTMA_PFAS_wB97X-D3_structures_perspective.png",
    "BTMA_PFAS_wB97X-D3_structures_perspective_no_panel_letters.png",
    "Extended_Monomer_PFAS_r2SCAN-3c_structures_orthoscopic.png",
    "Extended_Monomer_PFAS_r2SCAN-3c_structures_perspective.png",
    "Extended_Monomer_PFAS_wB97X-D3_structures_orthoscopic.png",
    "Extended_Monomer_PFAS_wB97X-D3_structures_perspective.png",
    "Extended_Monomer_PFAS_Octanol_epsilon72p5_structures_orthoscopic.png",
    "Extended_Monomer_PFAS_Octanol_epsilon72p5_structures_perspective.png",
]
GENERATED_INTERMEDIATE_PREFIXES = ("A_BTMA-", "B_BTMA-", "C_BTMA-", "D_BTMA-")


@dataclass(frozen=True)
class PanelSpec:
    letter: str
    title: str
    species: str
    anion_label: str


PANELS = [
    PanelSpec("A", "BTMA-PFOA", "PFOA", "PFOA-"),
    PanelSpec("B", "BTMA-PFOS", "PFOS", "PFOS-"),
    PanelSpec("C", "BTMA-FHEA", "FHEA", "FHEA-"),
    PanelSpec("D", "BTMA-PFHxA", "PFHxA", "PFHxA-"),
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    input_root: Path
    subdir_suffix: str
    xyz_suffix: str | None
    output_stem: str
    title_left: str
    title_superscript: str | None
    title_right: str
    cation_label: str
    xyz_suffix_by_species: dict[str, str] | None = None
    fallback_input_root: Path | None = None
    fallback_subdir_suffix: str | None = None
    fallback_xyz_suffix_by_species: dict[str, str] | None = None
    panel_species: tuple[str, ...] | None = None


DATASETS = {
    "r2scan3c": DatasetSpec(
        key="r2scan3c",
        input_root=ROOT / "01-1_BTMA" / "1_Opt_0.15M" / "1_R4N+X-_0.15M",
        subdir_suffix="_0.15M",
        xyz_suffix=None,
        output_stem="BTMA_PFAS_optimized_structures",
        title_left="PFAS-BTMA Complexes (r",
        title_superscript="2",
        title_right="SCAN-3c)",
        cation_label="BTMA+",
    ),
    "wb97xd3": DatasetSpec(
        key="wb97xd3",
        input_root=ROOT / "01-1_BTMA" / "2-1_Freq_wb" / "1_R4N+X-_wb",
        subdir_suffix="_wb",
        xyz_suffix=None,
        output_stem="BTMA_PFAS_wB97X-D3_structures",
        title_left="PFAS-BTMA Complexes (",
        title_superscript=None,
        title_right="ωB97X-D3)",
        cation_label="BTMA+",
    ),
    "extended-r2scan3c": DatasetSpec(
        key="extended-r2scan3c",
        input_root=ROOT / "01-5_Ext_Monomer" / "1-5-2_ExtFreqGlobMin" / "r2" / "1_R4N+X-_freqGMr2",
        subdir_suffix="_freqGMr2",
        xyz_suffix=None,
        output_stem="Extended_Monomer_PFAS_r2SCAN-3c_structures",
        title_left="PFAS-Extended Monomer Complexes (r",
        title_superscript="2",
        title_right="SCAN-3c)",
        cation_label="Extended\nMonomer",
        xyz_suffix_by_species={
            "FHEA": "_ext",
            "PFHxA": "_ext_r2",
            "PFOA": "_r2",
            "PFOS": "_ext",
        },
        fallback_input_root=ROOT / "01-5_Ext_Monomer" / "1-5-1_ExtOptGlobMin" / "GOAT" / "1_R4N+X-_GOAT",
        fallback_subdir_suffix="_GOAT",
        fallback_xyz_suffix_by_species={
            "FHEA": "_ext",
            "PFHxA": "_ext_r2",
            "PFOA": "_ext_r2",
            "PFOS": "_ext",
        },
    ),
    "extended-wb97xd3": DatasetSpec(
        key="extended-wb97xd3",
        input_root=ROOT / "01-5_Ext_Monomer" / "1-5-2_ExtFreqGlobMin" / "wb" / "1_R4N+X-_freqGMwb",
        subdir_suffix="_freqGMwb",
        xyz_suffix="_wb",
        output_stem="Extended_Monomer_PFAS_wB97X-D3_structures",
        title_left="PFAS-Extended Monomer Complexes (",
        title_superscript=None,
        title_right="ωB97X-D3)",
        cation_label="Extended\nMonomer",
    ),
    "extended-octanol-epsilon72p5": DatasetSpec(
        key="extended-octanol-epsilon72p5",
        input_root=ROOT
        / "01-5_Ext_Monomer"
        / "1-5-3_OtherSolvents"
        / "Octanol_epsilon72.5"
        / "Freq"
        / "wb"
        / "1_R4N+X-",
        subdir_suffix="",
        xyz_suffix="_wb",
        output_stem="Extended_Monomer_PFAS_Octanol_epsilon72p5_structures",
        title_left="PFAS-Extended Monomer Complexes (",
        title_superscript=None,
        title_right="Octanol, ε = 72.5)",
        cation_label="Extended\nMonomer",
        panel_species=("PFOA", "PFOS"),
    ),
}


def union_find(n: int, bonds: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in bonds:
        union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def choose_contacts(structure: render.Structure, count: int = 2) -> list[render.Contact]:
    bonds = render.infer_bonds(structure.elements, structure.coords, scale=1.25)
    components = union_find(len(structure.elements), bonds)
    if len(components) < 2:
        return []

    nitrogen_components = [component for component in components if any(structure.elements[i] == "N" for i in component)]
    cation = nitrogen_components[0] if nitrogen_components else components[0]
    anion = max((component for component in components if component is not cation), key=len)

    candidates: list[tuple[float, int, int]] = []
    for i in cation:
        for j in anion:
            ei, ej = structure.elements[i], structure.elements[j]
            if ei == "H" and ej == "H":
                continue
            dist = float(np.linalg.norm(structure.coords[i] - structure.coords[j]))
            candidates.append((dist, i, j))
    candidates.sort()

    center = structure.coords.mean(axis=0)
    selected: list[render.Contact] = []
    used: set[int] = set()
    for dist, i, j in candidates:
        if dist > 4.0:
            break
        if i in used or j in used:
            continue
        midpoint = structure.coords[[i, j]].mean(axis=0)
        away = midpoint - center
        norm = float(np.linalg.norm(away))
        if norm < 1e-6:
            away = np.array([0.0, 0.0, 1.0])
        else:
            away = away / norm
        rank = len(selected)
        stagger = np.array([0.0, 0.0, 0.9 * ((rank % 2) * 2 - 1)])
        label_position = tuple(midpoint + (1.2 + 0.35 * rank) * away + stagger)
        color = CONTACT_COLORS[rank % len(CONTACT_COLORS)]
        selected.append(render.Contact(i=i, j=j, color=color, label=f"{dist:.1f}", label_position=label_position))
        used.update({i, j})
        if len(selected) == count:
            break
    return selected


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/home/zwa/miniforge3/envs/pymol-render/fonts/DejaVuSans-Bold.ttf" if bold else "/home/zwa/miniforge3/envs/pymol-render/fonts/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def xyz_suffix_for(dataset: DatasetSpec, species: str) -> str:
    if dataset.xyz_suffix_by_species and species in dataset.xyz_suffix_by_species:
        return dataset.xyz_suffix_by_species[species]
    if dataset.xyz_suffix is not None:
        return dataset.xyz_suffix
    return dataset.subdir_suffix


def structure_xyz_path(dataset: DatasetSpec, spec: PanelSpec) -> Path:
    subdir = f"R4N+{spec.species}-{dataset.subdir_suffix}"
    xyz = dataset.input_root / subdir / f"R4N+{spec.species}-{xyz_suffix_for(dataset, spec.species)}.xyz"
    if xyz.exists():
        return xyz

    same_name = dataset.input_root / subdir / f"{subdir}.xyz"
    if same_name.exists():
        return same_name

    if dataset.fallback_input_root and dataset.fallback_subdir_suffix:
        fallback_subdir = f"R4N+{spec.species}-{dataset.fallback_subdir_suffix}"
        if dataset.fallback_xyz_suffix_by_species and spec.species in dataset.fallback_xyz_suffix_by_species:
            fallback_suffix = dataset.fallback_xyz_suffix_by_species[spec.species]
        else:
            fallback_suffix = dataset.fallback_subdir_suffix
        fallback = dataset.fallback_input_root / fallback_subdir / f"R4N+{spec.species}-{fallback_suffix}.xyz"
        if fallback.exists():
            print(f"Using fallback xyz for {dataset.key} {spec.title}: {fallback}")
            return fallback

    raise FileNotFoundError(
        "Missing expected structure file. Tried:\n"
        f"  {xyz}\n"
        f"  {same_name}"
    )


def panels_for(dataset: DatasetSpec) -> list[PanelSpec]:
    if dataset.panel_species is None:
        return PANELS
    species = set(dataset.panel_species)
    return [panel for panel in PANELS if panel.species in species]


def render_panel(
    spec: PanelSpec,
    dataset: DatasetSpec,
    tmpdir: Path,
    projection: str,
) -> tuple[Path, list[render.Contact]]:
    xyz = structure_xyz_path(dataset, spec)

    structure = render.read_structure(xyz, frame=-1)
    contacts = choose_contacts(structure)
    render_contacts = [
        render.Contact(
            i=contact.i,
            j=contact.j,
            color=contact.color,
            label="",
            label_position=contact.label_position,
        )
        for contact in contacts
    ]
    out_path = tmpdir / f"{spec.letter}_{spec.title}_{projection}.png"
    render.draw_structure_pymol(
        input_path=xyz.resolve(),
        structure=structure,
        out_path=out_path.resolve(),
        contacts=render_contacts,
        hide_h=False,
        show_indices=False,
        title=None,
        frame=-1,
        dpi=450,
        width=1800,
        height=1300,
        no_bonds=False,
        sphere_scale=0.20,
        stick_radius=0.12,
        ray=True,
        pymol_projection=projection,
        no_crop=False,
        keep_script=None,
    )
    return out_path, contacts


def trim_white_margin(panel: Image.Image, pad: int = 70) -> Image.Image:
    image = panel.convert("RGBA")
    rgb = np.asarray(image.convert("RGB"))
    non_white = np.any(rgb < 248, axis=2)
    rows, cols = np.where(non_white)
    if rows.size == 0 or cols.size == 0:
        return image
    left = max(int(cols.min()) - pad, 0)
    upper = max(int(rows.min()) - pad, 0)
    right = min(int(cols.max()) + pad + 1, image.width)
    lower = min(int(rows.max()) + pad + 1, image.height)
    return image.crop((left, upper, right, lower))


def paste_fit(canvas: Image.Image, panel: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    max_w, max_h = x1 - x0, y1 - y0
    panel = trim_white_margin(panel)
    scale = min(max_w / panel.width, max_h / panel.height)
    new_size = (int(panel.width * scale), int(panel.height * scale))
    panel = panel.resize(new_size, Image.Resampling.LANCZOS)
    x = x0 + (max_w - panel.width) // 2
    y = y0 + (max_h - panel.height) // 2
    canvas.alpha_composite(panel, (x, y))


def draw_contact_key(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    contacts: list[render.Contact],
    text_font: ImageFont.FreeTypeFont,
) -> None:
    row_h = 96
    box_w = 430
    box_h = 34 + max(len(contacts), 1) * row_h
    dash = 28
    gap = 18
    border_color = "#555555"

    x0, y0 = x - 30, y - 20
    x1, y1 = x0 + box_w, y0 + box_h
    xx = x0
    while xx < x1:
        draw.line((xx, y0, min(xx + dash, x1), y0), fill=border_color, width=5)
        draw.line((xx, y1, min(xx + dash, x1), y1), fill=border_color, width=5)
        xx += dash + gap
    yy = y0
    while yy < y1:
        draw.line((x0, yy, x0, min(yy + dash, y1)), fill=border_color, width=5)
        draw.line((x1, yy, x1, min(yy + dash, y1)), fill=border_color, width=5)
        yy += dash + gap

    for rank, contact in enumerate(contacts):
        yy = y + rank * row_h
        color = contact.color
        draw.line((x, yy + 42, x + 88, yy + 42), fill=color, width=11)
        for tick in range(0, 88, 34):
            draw.rectangle((x + tick + 11, yy + 33, x + tick + 25, yy + 52), fill="white")
        draw.text((x + 118, yy), f"{contact.label} A", fill=color, font=text_font)


def text_width(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    return bbox[2] - bbox[0]


def draw_centered_title(draw: ImageDraw.ImageDraw, width: int, y: int, dataset: DatasetSpec) -> None:
    title_font = font(90, bold=True)
    superscript_font = font(54, bold=True)
    left = dataset.title_left
    superscript = dataset.title_superscript
    right = dataset.title_right
    total_w = (
        text_width(draw, left, title_font)
        + (text_width(draw, superscript, superscript_font) if superscript else 0)
        + text_width(draw, right, title_font)
    )
    x = (width - total_w) // 2
    draw.text((x, y), left, fill="black", font=title_font)
    x += text_width(draw, left, title_font)
    if superscript:
        draw.text((x, y - 18), superscript, fill="black", font=superscript_font)
        x += text_width(draw, superscript, superscript_font)
    draw.text((x, y), right, fill="black", font=title_font)


def make_composite(
    panel_data: list[tuple[Path, list[render.Contact]]],
    out_path: Path,
    dataset: DatasetSpec,
    show_panel_letters: bool = True,
) -> Path:
    cell_w, cell_h = 1740, 1320
    header_h = 265
    margin = 65
    title_h = 140
    line_w = 8
    width = margin * 2 + cell_w * 2 + line_w
    panels = panels_for(dataset)
    n_rows = (len(panels) + 1) // 2
    height = title_h + margin * 2 + (cell_h + header_h) * n_rows + line_w * max(n_rows - 1, 0)
    canvas = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = font(96, bold=True)
    molecule_font = font(88, bold=True)
    cation_font = font(58 if "\n" in dataset.cation_label else 88, bold=True)
    contact_font = font(78, bold=True)
    contact_box_w = 430
    contact_offset = 85

    draw_centered_title(draw, width, 24, dataset)

    for idx, (spec, (path, contacts)) in enumerate(zip(panels, panel_data)):
        row, col = divmod(idx, 2)
        x = margin + col * (cell_w + line_w)
        y = title_h + margin + row * (cell_h + header_h + line_w)
        image_box = (x + 55, y + header_h + 20, x + cell_w - 55, y + header_h + cell_h - 245)
        paste_fit(canvas, Image.open(path), image_box)
        if show_panel_letters:
            draw.text((x + 18, y + 2), spec.letter, fill="black", font=label_font)
        draw.text((x + 135, y + header_h + cell_h - 205), spec.anion_label, fill="black", font=molecule_font)
        if "\n" in dataset.cation_label:
            cation_bbox = draw.multiline_textbbox((0, 0), dataset.cation_label, font=cation_font, spacing=8)
            cation_w = cation_bbox[2] - cation_bbox[0]
            cation_xy = (x + cell_w - 120 - cation_w, y + header_h + cell_h - 250)
        else:
            cation_xy = (x + cell_w - 465, y + header_h + cell_h - 370)
        draw.multiline_text(
            cation_xy,
            dataset.cation_label,
            fill="black",
            font=cation_font,
            spacing=8,
            align="center",
        )
        contact_x = x + cell_w - contact_offset - contact_box_w + 30
        contact_y = y + contact_offset + 20
        draw_contact_key(draw, contact_x, contact_y, contacts, contact_font)

    border_top = title_h + margin
    border = (margin, border_top, width - margin, height - margin)
    divider_x = margin + cell_w
    divider_y = border_top + header_h + cell_h
    draw.rectangle(border, outline="black", width=line_w)
    draw.line((divider_x, border_top, divider_x, height - margin), fill="black", width=line_w)
    if n_rows > 1:
        draw.line((margin, divider_y, width - margin, divider_y), fill="black", width=line_w)

    canvas.convert("RGB").save(out_path, dpi=(600, 600))
    return out_path


def clean_generated_intermediates() -> None:
    for path in OUTDIR.glob("*"):
        if not path.is_file():
            continue
        if path.name in FINAL_FILES:
            continue
        if (
            path.name == "BTMA_PFAS_optimized_structures.png"
            or path.name.startswith("BTMA_PFAS_optimized_structures_")
            or path.name.startswith(GENERATED_INTERMEDIATE_PREFIXES)
        ):
            path.unlink()


def build_projection(dataset: DatasetSpec, projection: str, out_name: str, show_panel_letters: bool = True) -> Path:
    with tempfile.TemporaryDirectory(prefix=f"btma_{projection}_panels_") as tmpdir_raw:
        tmpdir = Path(tmpdir_raw)
        panel_data = []
        for spec in panels_for(dataset):
            panel_path, contacts = render_panel(spec, dataset, tmpdir, projection)
            panel_data.append((panel_path, contacts))
            pairs = ", ".join(f"{contact.i + 1}-{contact.j + 1}" for contact in contacts)
            print(f"{dataset.key} {projection} {spec.letter} {spec.title}: contacts {pairs}")
        return make_composite(panel_data, OUTDIR / out_name, dataset, show_panel_letters=show_panel_letters)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection",
        choices=["orthoscopic", "perspective"],
        default="orthoscopic",
        help="PyMOL projection for the final composite; orthoscopic is PyMOL's orthographic mode",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="r2scan3c",
        help="Structure set to render",
    )
    parser.add_argument(
        "--no-panel-letters",
        action="store_true",
        help="Suppress internal A-D panel letters in the structure composite",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = DATASETS[args.dataset]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    clean_generated_intermediates()
    suffix = "_no_panel_letters" if args.no_panel_letters else ""
    composite = build_projection(
        dataset,
        args.projection,
        f"{dataset.output_stem}_{args.projection}{suffix}.png",
        show_panel_letters=not args.no_panel_letters,
    )
    clean_generated_intermediates()
    print(f"Composite: {composite}")


if __name__ == "__main__":
    main()
