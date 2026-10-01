#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sm_root="$(cd -- "$script_dir/../.." && pwd)"
md_root="$sm_root/01_Calculations/05_MD/02_PFOA_Associated_Production"
topology="$md_root/inputs/pVBTMA12_PFOA_assoc.prmtop"
work_root="${WORK_ROOT:-$sm_root/../_video_render_work}"
vmd_bin="${VMD_BIN:-$(command -v vmd || true)}"
ffmpeg_bin="${FFMPEG_BIN:-$(command -v ffmpeg || true)}"
mode="${MODE:-test}"
stride="${FRAME_STRIDE:-50}"
width="${WIDTH:-960}"
height="${HEIGHT:-720}"
fps="${FPS:-10}"

if [[ "$mode" == test ]]; then
    replicas=(01)
    frame_limit=3
elif [[ "$mode" == final ]]; then
    replicas=(01 02 03)
    frame_limit=100000
else
    printf 'MODE must be test or final\n' >&2
    exit 2
fi

[[ -s "$topology" ]] || { printf 'Missing topology: %s\n' "$topology" >&2; exit 2; }
[[ -x "$vmd_bin" ]] || { printf 'Missing VMD: %s\n' "$vmd_bin" >&2; exit 2; }
[[ -x "$ffmpeg_bin" ]] || { printf 'Missing FFmpeg: %s\n' "$ffmpeg_bin" >&2; exit 2; }

mkdir -p "$work_root" "$sm_root/05_MD_Animations/videos"
for replica in "${replicas[@]}"; do
    replica_dir="$md_root/replica_$replica"
    frame_dir="$work_root/$mode/replica_$replica"
    mkdir -p "$frame_dir"
    "$vmd_bin" -dispdev text -size "$width" "$height" \
        -e "$script_dir/render_pvbtma12_replica.tcl" -args \
        "$topology" "$replica_dir" "$frame_dir" "$replica" \
        "$stride" "$frame_limit" "$width" "$height" \
        > "$frame_dir/vmd_render.log" 2>&1
    grep -q 'COMPLETE: rendered' "$frame_dir/vmd_render.log" || {
        tail -50 "$frame_dir/vmd_render.log" >&2
        exit 2
    }
    encoded_name="replica_${replica}_encoded.mp4"
    (
        cd "$frame_dir"
        "$ffmpeg_bin" -hide_banner -loglevel warning -y -framerate "$fps" \
            -i 'frame_%04d.tga' -vf 'subtitles=frame_labels.srt' \
            -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
            "$encoded_name" > ffmpeg.log 2>&1
    )
    if [[ "$mode" == test ]]; then
        output="$work_root/replica_${replica}_test.mp4"
    else
        output="$sm_root/05_MD_Animations/videos/pVBTMA12_PFOA_replica_${replica}_0-10ns.mp4"
    fi
    mv -- "$frame_dir/$encoded_name" "$output"
    [[ -s "$output" ]] || { printf 'Missing encoded video: %s\n' "$output" >&2; exit 2; }
    printf '%s\n' "$output"
done
