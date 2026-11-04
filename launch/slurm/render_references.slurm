#!/bin/bash
# Videos of references themselves (scripts/policy/visualize_episode.py --reference: the hand and the object put on the
# reference frame by frame, no policy), one per reference, into $OUT/<tag>.mp4. Upload them from a login node with
# scripts/policy/upload_reference_videos.py (compute nodes have no internet).
# On Jean Zay the EGL read-back (pyrender read_color_buf) aborts the process (SIGABRT, no message) after ~50 frames:
# each reference is rendered in CHUNK-frame pieces, one process each (3 attempts), joined with ffmpeg; the references
# run side by side on the GPU.
# Env: REFS (space-separated reference.npz paths)  OUT (default outputs/episodes/references)  CHUNK (default 40)
#SBATCH -J render_refs
#SBATCH -A yfl@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm/%x_%j.out
REPO=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Video2Sim
cd "$REPO"
: "${REFS:?}"
source launch/config.sh
export PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
OUT=${OUT:-outputs/episodes/references}
CHUNK=${CHUNK:-40}
FFMPEG=$("$PY_SIM" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p "$OUT"

render_one() {      # one reference, its chunks one after the other
    local ref=$1 i=$2 tag n a dir ok list
    tag=$(basename "$(dirname "$ref")")_reference
    export QD_OFFLINE_CACHE_FILE_PATH="${JOBSCRATCH:-/tmp/$SLURM_JOB_ID}/quadrants_$i"
    mkdir -p "$QD_OFFLINE_CACHE_FILE_PATH"
    cp -r "$HOME/.cache/quadrants/qdcache/." "$QD_OFFLINE_CACHE_FILE_PATH/" 2>/dev/null || true
    n=$("$PY_SIM" -c "import numpy as np, sys; print(len(np.load(sys.argv[1])['cam_pos']))" "$ref")
    list="$OUT/chunks/$tag/list.txt"; mkdir -p "$OUT/chunks/$tag"; : > "$list"
    for ((a = 0; a < n - 1; a += CHUNK)); do
        dir="$OUT/chunks/$tag/$(printf %05d $a)"; mkdir -p "$dir"; ok=0
        for attempt in 1 2 3; do
            if "$PY_SIM" scripts/policy/visualize_episode.py --reference "$ref" --frames "$a:$((a + CHUNK))" --out "$dir" \
                    > "$dir/log.txt" 2>&1; then ok=1; break; fi
            echo "$tag frames $a: attempt $attempt failed"; grep -m1 -A3 "Fatal" "$dir/log.txt"
        done
        [ $ok = 1 ] || { echo "$tag: frames $a failed 3 times, stopping"; return 1; }
        echo "file '$(realpath "$dir/$tag.mp4")'" >> "$list"
    done
    "$FFMPEG" -y -loglevel error -f concat -safe 0 -i "$list" -c copy "$OUT/$tag.mp4" && echo "$tag: $n frames -> $OUT/$tag.mp4"
}

i=0; pids=()
for ref in $REFS; do
    render_one "$ref" $i 2>&1 | grep --line-buffered -v '^profiling\|ast.Str\|Falling back to convex hull' &
    pids+=($!); i=$((i + 1)); sleep 20
done
for p in "${pids[@]}"; do wait "$p"; done
ls -la "$OUT"
