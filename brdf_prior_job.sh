#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8

gpus='0'
proj_root='/home/sanskara/nerfactor'
repo_dir="/home/sanskara/nerfactor"
data_root="$repo_dir/data/brdf_merl_npz/ims256_envmaph16_spp1"
outroot="$repo_dir/output/train/merl"
viewer_prefix=''
REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='brdf.ini' --config_override="data_root=$data_root,outroot=$outroot,viewer_prefix=$viewer_prefix"


# II. Exploring the Learned Space (validation and testing)
#ckpt="$outroot/lr1e-2/checkpoints/ckpt-50"
#REPO_DIR="$repo_dir" "$repo_dir/nerfactor/explore_brdf_space_run.sh" "$gpus" --ckpt="$ckpt"
