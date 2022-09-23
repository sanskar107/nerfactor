#!/bin/bash
#SBATCH --job-name=nerf_
#SBATCH -p gpu
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --output="dtu_nerf_logs/nerf-%a-%j.log"
#SBATCH --open-mode=append
#SBATCH --array="16-22"

scenes=(
split16_dtu_scan122
split16_dtu_scan106
split16_dtu_scan24
split16_dtu_scan83
split16_dtu_scan114
split16_bmvs_stone
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
split16_bmvs_bear
split16_dtu_scan118
split16_bmvs_dog
split16_bmvs_man
split16_bmvs_durian
split16_dtu_scan63
)

scene="${scenes[$SLURM_ARRAY_TASK_ID]}"
echo "====== Scene: $scene ======"

gpus='0,1,2,3'
proj_root='/home/sanskara/nerfactor'
repo_dir="/home/sanskara/nerfactor"
viewer_prefix=''
data_root="/export/work/sanskara/svbrdf/data/nerfactor_dtu/$scene"

if [[ "$scene" == scan* ]]; then
    # DTU scenes
    imh='256'
else
    imh='512'
fi

# near='1.2'; far='11.28'  # backpack
# near='1.2'; far='23.11'    # handbag 396
# near='1.2'; far='31.82'  # handbag 399
# near='1.2'; far='26.46'  # hydrant 106
# near='1.2'; far='33.9'  # hydrant 268
# near='1.2'; far='29.04'  # motorcycle
# near='1.2'; far='44.57'  # plant_253_27235_55344
# near='1.2'; far='14.05'  # plant_372_40884_81286
# near='1.2'; far='37.024'  # toytruck_353_37431_70460
# near='1.2'; far='12.7'  # toytruck_379_44672_89080
# near='1.2'; far='6.08'  # vase_380_44868_89574

near='0.1'; far='2'

lr='5e-4'

outroot="/export/work/sanskara/svbrdf/nerfactor_dtu/output/train/${scene}_nerf"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='nerf.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,lr=$lr,outroot=$outroot,viewer_prefix=$viewer_prefix"

# # Optionally, render the test trajectory with the trained NeRF
# ckpt="$outroot/lr$lr/checkpoints/ckpt-20"
# REPO_DIR="$repo_dir" "$repo_dir/nerfactor/nerf_test_run.sh" "$gpus" --ckpt="$ckpt"
