#!/bin/bash
#SBATCH --job-name=nerf
#SBATCH -p g24
#SBATCH -C ampere
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --output="all_logs/nerf/pingpong-%j.log"
#SBATCH --open-mode=append
#SBATCH --array="0-3"

scenes=(
air_baloons
chair
hotdog
jugs
)

scene="${scenes[$SLURM_ARRAY_TASK_ID]}"
near="0.01"
far="2.0"
echo "====== Scene: $scene , Near: $near, Far: $far  ======"

# gpus='0,1,2,3'
gpus='0'
proj_root='/homes/sanskar/nerfactor'
repo_dir="/homes/sanskar/nerfactor"
viewer_prefix=''
data_root="/export/work/sanskar/nerfactor/input/$scene"

imh='512'

lr='5e-4'

outroot="/export/work/sanskar/nerfactor/output/train/${scene}"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='nerf.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,lr=$lr,outroot=$outroot,viewer_prefix=$viewer_prefix"

# # Optionally, render the test trajectory with the trained NeRF
# ckpt="$outroot/lr$lr/checkpoints/ckpt-20"
# REPO_DIR="$repo_dir" "$repo_dir/nerfactor/nerf_test_run.sh" "$gpus" --ckpt="$ckpt"