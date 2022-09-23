#!/bin/bash
#SBATCH --job-name=dtu6_nerf_geom
#SBATCH -p g24
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output="dtu_6_nerf_geom_logs/split16_dtu_illum6_scan63/job-%a-%j.log"
#SBATCH --open-mode=append
#SBATCH --array="0-44"

scene='split16_dtu_illum6_scan63'
# mkdir -p dtu_6_nerf_geom_logs/"$scene"
gpus='0'
proj_root='/home/sanskara/nerfactor'
repo_dir="/home/sanskara/nerfactor"
viewer_prefix=''
data_root="/export/share/projects/svbrdf/data/dtu_6_nerd/nerfactor_dtu_6/input/$scene"

if [[ "$scene" == scan* ]]; then
    # DTU scenes
    imh='256'
else
    imh='512'
fi

lr='5e-4'

trained_nerf="/export/share/projects/svbrdf/data/dtu_6_nerd/nerfactor_dtu_6/output/train/${scene}_nerf/lr$lr"
occu_thres='0.5'

if [[ "$scene" == pinecone* || "$scene" == scan* ]]; then
    # pinecone and DTU scenes
    scene_bbox='-0.3,0.3,-0.3,0.3,-0.3,0.3'
elif [[ "$scene" == vasedeck* ]]; then
    scene_bbox='-0.2,0.2,-0.4,0.4,-0.5,0.5'
else
    # We don't need to bound the synthetic scenes
    scene_bbox='-0.3,0.3,-0.3,0.3,-0.3,0.3'
    # scene_bbox=''
fi

echo ${scene_bbox}

count=$(python 6_get_count_dtu.py "$scene")

idxs=($(seq 0 1 "$count"))
# idxs=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97)
# idxs=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59)

idx="${idxs[$SLURM_ARRAY_TASK_ID]}"
echo "====== Index: $idx ======"

out_root="/export/share/projects/svbrdf/data/dtu_6_nerd/nerfactor_dtu_6/output/train/$scene"
mlp_chunk='1075000' # bump this up until GPU gets OOM for faster computation

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/geometry_from_nerf_run.sh" "$gpus" --data_root="$data_root" --trained_nerf="$trained_nerf" --out_root="$out_root" --imh="$imh" --scene_bbox="$scene_bbox" --occu_thres="$occu_thres" --mlp_chunk="$mlp_chunk" --idx="$idx"
