#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8

proj_root='/home/sanskara/nerfactor'
repo_dir="/home/sanskara/nerfactor"
indir="/export/work/sanskara/nerfactor/brdf/brdfs"
ims='256'
outdir="$proj_root/data/brdf_merl_npz/ims${ims}_envmaph16_spp1"
REPO_DIR="$repo_dir" "$repo_dir"/data_gen/merl/make_dataset_run.sh "$indir" "$ims" "$outdir"
