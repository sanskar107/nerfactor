import numpy as np

# scenes=["scan37", "scan40", "scan55", "scan63", "scan65", "scan69", "scan83", "scan97"]
scenes=["bear", "clock", "dog", "durian", "jade", "man", "sculpture", "stone"]

path = "/homes/sanskar/data/bmvsdtu/llff_data/"
for scene in scenes:
    poses = np.load(path + scene + "/poses_bounds.npy")
    val = np.load(path + scene + "/val_poses_bounds.npy")
    poses = np.concatenate((poses, val), axis=0)
    b_min = np.min(poses[:, 15])
    b_max = np.max(poses[:, 16])
    scale = 1. / (b_min * 0.75)
    # b_min *= scale
    # b_max *= scale

    print(f"Scene : {scene}, scale : {scale}, min : {b_min}, max : {b_max}")
