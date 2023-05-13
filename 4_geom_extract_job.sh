#!/bin/bash
#SBATCH --job-name=wood_bowl_geom_extract
#SBATCH -p g24
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output="all_logs/geom_extract/wood_bowl-%a-%j.log"
#SBATCH --open-mode=append
#SBATCH --array="0-59"

scene='wood_bowl'  # 63
gpus='0'
proj_root='/homes/sanskar/nerfactor'
repo_dir="/homes/sanskar/nerfactor"
viewer_prefix=''
data_root="/export/work/sanskar/nerfactor/input/$scene"

imh='512'

lr='5e-4'

trained_nerf="/export/work/sanskar/nerfactor/output/train/${scene}/lr$lr"
occu_thres='0.5'

# scene_bbox='-0.3,0.3,-0.3,0.3,-0.3,0.3'
scene_bbox=''

echo ${scene_bbox}

count=$(python get_count.py "$scene")

idxs=($(seq 0 1 "$count"))

idx="${idxs[$SLURM_ARRAY_TASK_ID]}"
echo "====== Index: $idx ======"

out_root="/export/work/sanskar/nerfactor/output/train/${scene}_geom"
mlp_chunk='475000' # bump this up until GPU gets OOM for faster computation

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/geometry_from_nerf_run.sh" "$gpus" --data_root="$data_root" --trained_nerf="$trained_nerf" --out_root="$out_root" --imh="$imh" --scene_bbox="$scene_bbox" --occu_thres="$occu_thres" --mlp_chunk="$mlp_chunk" --idx="$idx"

