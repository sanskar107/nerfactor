#!/bin/bash
#SBATCH --job-name=nerf_
#SBATCH -p gpu
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --output="nerf_logs_new/antman-%a-%j.log"
#SBATCH --open-mode=append


scene="antman1"
echo "====== Scene: $scene ======"

gpus='0,1,2,3'
proj_root='/home/sanskara/nerfactor'
repo_dir="/home/sanskara/nerfactor"
viewer_prefix=''
data_root="/export/share/projects/svbrdf/data/co3d_nerd/nerfactor/input/$scene"

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

outroot="/export/share/projects/svbrdf/data/co3d_nerd/nerfactor/output/train/${scene}_nerf"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='nerf.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,lr=$lr,outroot=$outroot,viewer_prefix=$viewer_prefix"

# # Optionally, render the test trajectory with the trained NeRF
# ckpt="$outroot/lr$lr/checkpoints/ckpt-20"
# REPO_DIR="$repo_dir" "$repo_dir/nerfactor/nerf_test_run.sh" "$gpus" --ckpt="$ckpt"
