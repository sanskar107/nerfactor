#!/bin/bash
#SBATCH --job-name=transform
#SBATCH --qos=normal
#SBATCH -c 10
#SBATCH --output="all_logs/transform/array-%j.log"
#SBATCH --open-mode=append
#SBATCH --array="0-3"

scenes=(
air_baloons
chair
hotdog
jugs
)


scene="${scenes[$SLURM_ARRAY_TASK_ID]}"
echo "====== Scene: $scene ======"

proj_root='/homes/sanskar/nerfactor'
repo_dir="/homes/sanskar/nerfactor"

scene_dir="/export/work/sanskar/data/llff_data/$scene"
h='512'
n_vali='9'

outroot="/export/work/sanskar/nerfactor/input/$scene"

REPO_DIR="$repo_dir" "$repo_dir/data_gen/nerf_real/make_dataset_run.sh" --scene_dir="$scene_dir" --h="$h" --n_vali="$n_vali" --outroot="$outroot"
