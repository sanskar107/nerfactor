#!/bin/bash
#SBATCH --job-name=nftr_optim
#SBATCH -p g24
#SBATCH -C turing
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --output="dtu_6_optim_logs/split16_dtu_illum6_scan106-%j.log"
#SBATCH --open-mode=append

scene='split16_dtu_illum6_scan106'
gpus='0,1,2,3'
model='nerfactor'
overwrite='True'
proj_root='/home/sanskara/nerfactor'
repo_dir="/home/sanskara/nerfactor"
viewer_prefix=''

# I. Shape Pre-Training
# data_root="/export/work/sanskara/svbrdf/data/nerfactor/$scene"
# data_root="/export/work/sanskara/svbrdf/data/nerfactor_dtu/$scene"
data_root="/export/share/projects/svbrdf/data/dtu_6_nerd/nerfactor_dtu_6/input/$scene"

if [[ "$scene" == scan* ]]; then
    # DTU scenes
    imh='256'
else
    imh='512'
fi

#-C turing

near='0.1'; far='2'

use_nerf_alpha='True'

# surf_root="/export/work/sanskara/svbrdf/nerfactor/output_new/train/$scene"
surf_root="/export/share/projects/svbrdf/data/dtu_6_nerd/nerfactor_dtu_6/output/train/$scene"

# shape_outdir="/export/work/sanskara/svbrdf/nerfactor/output_new/train/${scene}_shape"
shape_outdir="/export/share/projects/svbrdf/data/dtu_6_nerd/nerfactor_dtu_6/output/train/${scene}_shape"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='shape.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,outroot=$shape_outdir,viewer_prefix=$viewer_prefix,overwrite=$overwrite"

echo "\n\n====== Pretraining done, now running joint optimization ======\n\n"


# II. Joint Optimization (training and validation)
shape_ckpt="$shape_outdir/lr1e-2/checkpoints/ckpt-2"
brdf_ckpt="$repo_dir/output/train/merl/lr1e-2/checkpoints/ckpt-50"

xyz_jitter_std=0.001

test_envmap_dir="/home/sanskara/16_NeRD/relighting_envmaps"
shape_mode='finetune'
# outroot="$proj_root/output_new/train/${scene}_$model"
outroot="/export/share/projects/svbrdf/data/dtu_6_nerd/nerfactor_dtu_6/output/train/${scene}_$model"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config="$model.ini" --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,shape_model_ckpt=$shape_ckpt,brdf_model_ckpt=$brdf_ckpt,xyz_jitter_std=$xyz_jitter_std,test_envmap_dir=$test_envmap_dir,shape_mode=$shape_mode,outroot=$outroot,viewer_prefix=$viewer_prefix,overwrite=$overwrite"

echo "\n\n====== optim done, now running view synthesis ======\n\n"

# III. Simultaneous Relighting and View Synthesis (testing)
ckpt="$outroot/lr5e-3/checkpoints/ckpt-10"
color_correct_albedo='false'

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/test_run.sh" "$gpus" --ckpt="$ckpt" --color_correct_albedo="$color_correct_albedo"
