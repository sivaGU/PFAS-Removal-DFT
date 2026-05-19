# MD animation supplements

This directory contains rendered trajectory animations and the scripts used to generate them. The temporary rendered frame directories are intentionally not retained.

## Animations

- `videos/R48_PFOA_associated_MD.mp4`
  - Source topology: `work_amber/05_pfas_exchange/solvated_resin_pfoa_exchange.prmtop`
  - Source trajectory: `work_amber/09_namd_pfoa_exchange/resin_pfoa_exchange_npt_1ns.dcd`
  - Visual content: R48+ resin heavy atoms only, PFOA-, Cl-, and water oxygens within 5 Å of the PFOA fluorinated tail.

- `videos/R48_PFOA_associated_zoom_MD.mp4`
  - Source topology: `work_amber/05_pfas_exchange/solvated_resin_pfoa_exchange.prmtop`
  - Source trajectory: `work_amber/09_namd_pfoa_exchange/resin_pfoa_exchange_npt_1ns.dcd`
  - Visual content: zoomed PFOA- binding-site view with nearby R48+ heavy atoms, Cl-, and water oxygens within 6 Å of the PFOA fluorinated tail.

- `videos/R48_chloride_form_MD.mp4`
  - Source topology: `work_amber/04_leap/solvated_resin_cl.prmtop`
  - Source trajectory: `work_amber/07_namd_resin_only/resin_only_npt_2ns.dcd`
  - Visual content: R48+ resin heavy atoms only and Cl- ions.

## Rendering notes

- R48+ hydrogens are hidden in all animations.
- R48+ carbons are gray and R48+ nitrogens are blue.
- Trajectory frames are aligned to the first frame using R48+ heavy atoms.
- VMD `TachyonInternal` is used to render temporary `.tga` frames.
- FFmpeg is used to encode MP4 files from the temporary frame sequence.
- Default rendering settings: 100 sampled frames, 12 frames/s, 1080-pixel output height.

## Re-rendering

From the repository root:

```bash
bash Supplementary_Materials/05_MD_Animations/scripts/render_md_animations.sh
```

Optional overrides:

```bash
MAX_FRAMES=60 FPS=12 HEIGHT=1080 bash Supplementary_Materials/05_MD_Animations/scripts/render_md_animations.sh
```

If VMD or FFmpeg are installed somewhere else, set `VMD_BIN` or `FFMPEG_BIN` before running the script. During the original render, FFmpeg was provided through a temporary conda environment at `/tmp/md_anim_env`; that environment is not part of the supplementary files.
