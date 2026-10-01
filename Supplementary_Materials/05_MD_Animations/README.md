# pVBTMA12-PFOA Production Movies

`videos/` contains one MP4 for each independent Stage 10 production replica.
Each movie samples the replica's ten consecutive 1 ns DCD chunks in order,
then includes the exact final frame. `frame_manifests/replica_XX.csv` gives
the source chunk, source frame and production time for every movie frame.
All three use the same Stage 09 associated-system topology as the Stage 10
`inputs/pVBTMA12_PFOA_assoc.prmtop` file. The topology checksum matches
between those source locations.

The VMD script in `scripts/` images residues around the polymer under PBC,
fits polymer heavy atoms to the first sampled frame, recenters on the
polymer/PFOA, and uses a fixed camera thereafter. Polymer and PFOA heavy
atoms and nearby sodium/chloride are shown; bulk water is hidden for clarity.
No atom coordinates in the raw DCDs were edited. Frames were sampled every
50 DCD frames (100 ps at the saved 2 ps interval), starting at the first
production frame and ending with the exact 10 ns frame. Videos use 10 frames
per second, 960 x 720 pixels and an on-screen replica/time label.

Run from the supplementary root with VMD, PBCTools and FFmpeg available:

```bash
MODE=test VMD_BIN=/path/to/vmd FFMPEG_BIN=/path/to/ffmpeg \
  bash 05_MD_Animations/scripts/render_md_animations.sh
MODE=final VMD_BIN=/path/to/vmd FFMPEG_BIN=/path/to/ffmpeg \
  bash 05_MD_Animations/scripts/render_md_animations.sh
```

The renderer's temporary TGA frames are written to `_video_render_work` beside
the supplementary root by default. Set `WORK_ROOT` to use another location.
`SHA256SUMS.txt` covers the three MP4s, manifests and source scripts. Raw
production topologies, configurations, logs, XSTs and DCDs are in
`../01_Calculations/05_MD/02_PFOA_Associated_Production`.

