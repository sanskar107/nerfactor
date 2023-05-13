#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --output="./brdf_logs/brdf_data-%a-%j.log"

proj_root='/homes/sanskar/nerfactor'
repo_dir="/homes/sanskar/nerfactor"
indir="/export/work/sanskar/nerfactor_data/brdf_merl/brdfs"
ims='256'
outdir="$proj_root/data/brdf_merl_npz/ims${ims}_envmaph16_spp1"
REPO_DIR="$repo_dir" "$repo_dir"/data_gen/merl/make_dataset_run.sh "$indir" "$ims" "$outdir"
