#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ANIM_DIR="$ROOT/Supplementary_Materials/05_MD_Animations"
SCRIPT_DIR="$ANIM_DIR/scripts"
VIDEO_DIR="$ANIM_DIR/videos"
FRAME_ROOT="${TMPDIR:-/tmp}/rz_orca_pfas_md_animation_frames"

VMD_BIN="${VMD_BIN:-$(command -v vmd || true)}"
if [[ -z "$VMD_BIN" && -x /home/zwa/vmd/bin/vmd ]]; then
  VMD_BIN="/home/zwa/vmd/bin/vmd"
fi

FFMPEG_BIN="${FFMPEG_BIN:-$(command -v ffmpeg || true)}"
if [[ -z "$FFMPEG_BIN" && -x /tmp/md_anim_env/bin/ffmpeg ]]; then
  FFMPEG_BIN="/tmp/md_anim_env/bin/ffmpeg"
fi

VMD_LIB="${VMD_LIB:-/home/zwa/vmd/lib/vmd}"

FPS="${FPS:-12}"
MAX_FRAMES="${MAX_FRAMES:-100}"
HEIGHT="${HEIGHT:-1080}"
CRF="${CRF:-20}"

mkdir -p "$VIDEO_DIR" "$FRAME_ROOT"

if [[ ! -x "$VMD_BIN" ]]; then
  echo "ERROR: VMD executable not found at $VMD_BIN" >&2
  exit 1
fi

if [[ ! -x "$FFMPEG_BIN" ]]; then
  echo "ERROR: ffmpeg executable not found at $FFMPEG_BIN" >&2
  echo "Install/provide ffmpeg, or set FFMPEG_BIN to a usable executable." >&2
  exit 1
fi

render_movie() {
  local key="$1"
  local mode="$2"
  local topology="$3"
  local trajectory="$4"
  local title="$5"
  local frames="$FRAME_ROOT/$key"
  local raw_video="$VIDEO_DIR/${key}.raw.mp4"
  local final_video="$VIDEO_DIR/${key}.mp4"

  rm -rf "$frames"
  mkdir -p "$frames"

  echo "Rendering $title"
  echo "  topology:   $topology"
  echo "  trajectory: $trajectory"

  LD_LIBRARY_PATH="$VMD_LIB:${LD_LIBRARY_PATH:-}" \
    "$VMD_BIN" -dispdev text -eofexit \
    -e "$SCRIPT_DIR/vmd_render_animation.tcl" \
    -args "$topology" "$trajectory" "$frames" "$mode" "$MAX_FRAMES"

  "$FFMPEG_BIN" -y \
    -framerate "$FPS" \
    -i "$frames/frame_%04d.tga" \
    -vf "scale=-2:${HEIGHT},format=yuv420p" \
    -c:v libx264 \
    -crf "$CRF" \
    -pix_fmt yuv420p \
    "$raw_video"

  mv "$raw_video" "$final_video"
  rm -rf "$frames"
  echo "  wrote: $final_video"
}

render_movie \
  "R48_PFOA_associated_MD" \
  "pfoa_assoc" \
  "$ROOT/work_amber/05_pfas_exchange/solvated_resin_pfoa_exchange.prmtop" \
  "$ROOT/work_amber/09_namd_pfoa_exchange/resin_pfoa_exchange_npt_1ns.dcd" \
  "PFOA-associated R48+ trajectory"

render_movie \
  "R48_PFOA_associated_zoom_MD" \
  "pfoa_assoc_zoom" \
  "$ROOT/work_amber/05_pfas_exchange/solvated_resin_pfoa_exchange.prmtop" \
  "$ROOT/work_amber/09_namd_pfoa_exchange/resin_pfoa_exchange_npt_1ns.dcd" \
  "zoomed PFOA-associated R48+ trajectory"

render_movie \
  "R48_chloride_form_MD" \
  "r48_cl" \
  "$ROOT/work_amber/04_leap/solvated_resin_cl.prmtop" \
  "$ROOT/work_amber/07_namd_resin_only/resin_only_npt_2ns.dcd" \
  "chloride-form R48+ reference trajectory"

echo "All MD animation renders complete."
