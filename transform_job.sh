#!/bin/bash
#SBATCH --job-name=transform
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --output="transform_logs/transform-%j.log"
#SBATCH --open-mode=append
#SBATCH --array="0-22"

scenes=(
# split16_dtu_scan122  # old location
split16_dtu_scan106
split16_dtu_scan24
split16_dtu_scan83
split16_dtu_scan114
# split16_bmvs_stone  #all failed
split16_dtu_scan40
split16_dtu_scan105
split16_dtu_scan65
split16_dtu_scan69
split16_bmvs_jade
split16_dtu_scan37
split16_dtu_scan97
split16_dtu_scan55
split16_bmvs_clock
split16_dtu_scan110
split16_bmvs_sculpture
# split16_bmvs_bear  # all failed
split16_dtu_scan118
split16_bmvs_dog
split16_bmvs_man
split16_bmvs_durian
split16_dtu_scan63
)

scene="${scenes[$SLURM_ARRAY_TASK_ID]}"
echo "====== Scene: $scene ======"

proj_root='/home/sanskara/nerfactor'
repo_dir="/home/sanskara/nerfactor"

scene_dir="/export/share/projects/svbrdf/data/dtu_bmvs_nerd/$scene"
h='512'
n_vali='7'
outroot="/export/work/sanskara/svbrdf/data/nerfactor_dtu/$scene"

REPO_DIR="$repo_dir" "$repo_dir/data_gen/nerf_real/make_dataset_run.sh" --scene_dir="$scene_dir" --h="$h" --n_vali="$n_vali" --outroot="$outroot"