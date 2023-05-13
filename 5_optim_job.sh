#!/bin/bash
#SBATCH --job-name=nftr_optim
#SBATCH -p g48
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --output="all_logs/optim/relighting-%j.log"
#SBATCH --open-mode=append
#SBATCH --array="0-3"

scenes=(
air_baloons
chair
hotdog
jugs
)

scene="${scenes[$SLURM_ARRAY_TASK_ID]}"
echo "====== Scene: $scene ======"

# gpus='0,1,2,3'
gpus='0'
model='nerfactor'
overwrite='True'
proj_root='/homes/sanskar/nerfactor'
repo_dir="/homes/sanskar/nerfactor"
viewer_prefix=''

# I. Shape Pre-Training
data_root="/export/work/sanskar/nerfactor/input/$scene"

imh='512'

#sadfsad   #SBATCH -C turing

near='0.01'; far='1.5'

use_nerf_alpha='False'

surf_root="/export/work/sanskar/nerfactor/output/train/${scene}_geom"
shape_outdir="/export/work/sanskar/nerfactor/output/train/${scene}_shape"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='shape.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,outroot=$shape_outdir,viewer_prefix=$viewer_prefix,overwrite=$overwrite"

echo "\n\n====== Pretraining done, now running joint optimization ======\n\n"


# II. Joint Optimization (training and validation)
shape_ckpt="$shape_outdir/lr1e-2/checkpoints/ckpt-2"
brdf_ckpt="$repo_dir/output/train/merl/lr1e-2/checkpoints/ckpt-50"

xyz_jitter_std=0.001

# test_envmap_dir="/home/sanskara/16_NeRD/relighting_envmaps"
test_envmap_dir="/export/work/sanskar/data/llff_data/$scene/val_envmaps.npy"
shape_mode='finetune'
# outroot="$proj_root/output_new/train/${scene}_$model"
outroot="/export/work/sanskar/nerfactor/output/train/${scene}_$model"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config="$model.ini" --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,shape_model_ckpt=$shape_ckpt,brdf_model_ckpt=$brdf_ckpt,xyz_jitter_std=$xyz_jitter_std,test_envmap_dir=$test_envmap_dir,shape_mode=$shape_mode,outroot=$outroot,viewer_prefix=$viewer_prefix,overwrite=$overwrite"

echo "\n\n====== optim done, now running view synthesis ======\n\n"

# III. Simultaneous Relighting and View Synthesis (testing)
ckpt="$outroot/lr5e-3/checkpoints/ckpt-2"
color_correct_albedo='false'

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/test_run.sh" "$gpus" --ckpt="$ckpt" --color_correct_albedo="$color_correct_albedo"