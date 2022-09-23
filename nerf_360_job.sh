#!/bin/bash
#SBATCH --job-name=nerf
#SBATCH -p g24
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --output="nerf_360_logs/vasedeck-%j.log"
#SBATCH --open-mode=append

scene='vasedeck'
gpus='0,1,2,3'
proj_root='/home/sanskara/nerfactor'
repo_dir="/home/sanskara/nerfactor"
viewer_prefix=''
data_root="/export/work/sanskara/svbrdf/data/nerfactor/nerf_360/$scene"

imh='512'

near='0.1'; far='2'  # pinecone and vasedeck

lr='5e-4'

outroot="$repo_dir/output/train/${scene}_nerf"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='nerf.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,lr=$lr,outroot=$outroot,viewer_prefix=$viewer_prefix"

# # Optionally, render the test trajectory with the trained NeRF
# ckpt="$outroot/lr$lr/checkpoints/ckpt-20"
# REPO_DIR="$repo_dir" "$repo_dir/nerfactor/nerf_test_run.sh" "$gpus" --ckpt="$ckpt"
