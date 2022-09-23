#!/bin/bash
#SBATCH --job-name=transform
#SBATCH -p cpu
#SBATCH --qos=low
#SBATCH -c 10
#SBATCH --output="transform_logs/6_dtu-%j.log"
#SBATCH --open-mode=append
#SBATCH --array="0-14"

scenes=(
split16_dtu_illum6_scan122
split16_dtu_illum6_scan106
split16_dtu_illum6_scan24
split16_dtu_illum6_scan83
split16_dtu_illum6_scan114
split16_dtu_illum6_scan40
split16_dtu_illum6_scan105
split16_dtu_illum6_scan65
split16_dtu_illum6_scan69
split16_dtu_illum6_scan37
split16_dtu_illum6_scan97
split16_dtu_illum6_scan55
split16_dtu_illum6_scan110
split16_dtu_illum6_scan118
split16_dtu_illum6_scan63
)

scene="${scenes[$SLURM_ARRAY_TASK_ID]}"
echo "====== Scene: $scene ======"

proj_root='/home/sanskara/nerfactor'
repo_dir="/home/sanskara/nerfactor"

scene_dir="/export/share/projects/svbrdf/data/dtu_6_nerd/$scene"
h='512'
n_vali='7'

# outroot="/export/work/sanskara/svbrdf/data/nerfactor_dtu/$scene"
outroot="/export/share/projects/svbrdf/data/dtu_6_nerd/nerfactor_dtu_6/input/$scene"

REPO_DIR="$repo_dir" "$repo_dir/data_gen/nerf_real/make_dataset_run.sh" --scene_dir="$scene_dir" --h="$h" --n_vali="$n_vali" --outroot="$outroot"