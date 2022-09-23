#!/bin/bash
#SBATCH --job-name=transform
#SBATCH -p gpu
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --output="transform_logs/antman-%j.log"
#SBATCH --open-mode=append


scene="antman1"
echo "====== Scene: $scene ======"

proj_root='/home/sanskara/nerfactor'
repo_dir="/home/sanskara/nerfactor"

scene_dir="/export/work/sanskara/nerfactor/llff/$scene"
h='500'
n_vali='7'
outroot="/export/work/sanskara/nerfactor/llff/input_2/$scene"

REPO_DIR="$repo_dir" "$repo_dir/data_gen/nerf_real/make_dataset_run.sh" --scene_dir="$scene_dir" --h="$h" --n_vali="$n_vali" --outroot="$outroot"





## Used at the time of paper submission

# scene="antman1"
# echo "====== Scene: $scene ======"

# proj_root='/home/sanskara/nerfactor'
# repo_dir="/home/sanskara/nerfactor"

# scene_dir="/export/share/projects/svbrdf/data/co3d_nerd/$scene"
# h='512'
# n_vali='7'
# outroot="/export/share/projects/svbrdf/data/co3d_nerd/nerfactor/input/$scene"

# REPO_DIR="$repo_dir" "$repo_dir/data_gen/nerf_real/make_dataset_run.sh" --scene_dir="$scene_dir" --h="$h" --n_vali="$n_vali" --outroot="$outroot"