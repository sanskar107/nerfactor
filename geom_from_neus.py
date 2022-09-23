import numpy as np
import open3d as o3d
from glob import glob
from tqdm import tqdm
import os
import cv2
import zstandard as zstd 
import msgpack # pip install msgpack
import msgpack_numpy # pip install msgpack-numpy
from PIL import Image, ExifTags
from os.path import join, dirname

msgpack_numpy.patch()


def read_compressed_msgpack(path, decompressor=None):
    if decompressor is None:
        decompressor = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        data = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)
    return data


def read_cam(path, factor=1):
    cam = np.load(path)
    K = cam['K']
    K[0, 0] /= factor
    K[1, 1] /= factor
    K[0:2, 2] /= factor

    T = cam['T']
    R = T[:3, :3]
    t = T[:3, 3]
    return K, R, t.reshape(3,), T

def _warn_degree(angles):
    if (np.abs(angles) > 2 * np.pi).any():
        logger.warning((
            "Some input value falls outside [-2pi, 2pi]. You sure inputs are "
            "in radians"))


def sph2cart(pts_sph, convention='lat-lng'):
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Degrees?
    _warn_degree(pts_sph[:, 1:])

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart


def gen_light_xyz(envmap_h, envmap_w, envmap_radius=5e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas

def norm(x, axis=-1):
    return x / np.expand_dims(np.linalg.norm(x, axis=-1), -1)


def get_lvis(cam, scene, surf, normal):
    lxyz, lareas = gen_light_xyz(16, 32)
    # pcd = o3d.t.geometry.PointCloud(lxyz.reshape(-1, 3))
    # o3d.visualization.draw([pcd])

    lxyz_flat = lxyz.reshape(-1, 3)  # (16*32, 3)
    n_lights = lxyz_flat.shape[0]  # 16*32
    lvis_hit = np.zeros((surf.shape[0], n_lights), dtype=np.float32) # (n_surf_pts, n_lights)

    for i in range(n_lights):
        lxyz_i = lxyz_flat[i]

        # From surface to lights
        surf2l = lxyz_i - surf
        surf2l = norm(surf2l)

        lcos = (surf2l * normal).sum(1)
        front_lit = lcos > 0 # (n_surf_pts,)

        surf_frontlit = surf[front_lit]
        surf2l_frontlit = surf2l[front_lit]

        rays = np.concatenate([surf_frontlit, surf2l_frontlit], axis=1).astype(np.float32)
        ans = {k: v.numpy() for k, v in scene.cast_rays(rays).items()}

        alpha = ans['t_hit'].copy()
        hit = alpha != np.inf
        alpha[hit] = 1.0
        alpha[~hit] = 0.0
        lvis_hit[front_lit, i] = 1 - alpha

    return lvis_hit


def makedirs(path):
    os.makedirs(path, exist_ok=True)


def func(path, root):
    factor = 2
    makedirs(root)
    cams = sorted(glob(os.path.join(path, 'out_cam*.npz')))
    cams = [read_cam(cam, factor) for cam in cams]  # K, R, t, T

    imgs = sorted(glob(os.path.join(path, 'out_im*.png')))
    h, w, _ = cv2.imread(imgs[0]).shape
    h, w = h // factor, w // factor

    mesh_path = os.path.join(path, 'data3d.msgpack.zst')
    data = read_compressed_msgpack(mesh_path)

    mesh = data['mesh']
    pts = data['mesh']['vertices']

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh['vertices'], mesh['triangles'].astype(np.uint32))

    counter = [0, 0]  # train, val
    for i, cam in tqdm(enumerate(cams)):
        if i % 16 == 0:
            outdir = os.path.join(root, 'val_' + str(counter[1]).zfill(3))
            counter[1] += 1
        else:
            outdir = os.path.join(root, 'train_' + str(counter[0]).zfill(3))
            counter[0] += 1
        makedirs(outdir)

        K, R, t, T = cam
        rays = scene.create_rays_pinhole(K, T, w, h)
        ans = {k: v.numpy() for k, v in scene.cast_rays(rays).items()}
        alpha = ans['t_hit'].copy()
        hit = alpha != np.inf
        alpha[hit] = 1.0
        alpha[~hit] = 0.0
        alpha_map = np.clip(alpha, 0.0, 1.0)
        write_alpha(alpha_map.copy(), outdir)
        # cv2.imwrite('alpha.png', alpha.astype(np.uint8) * 255)

        normal = ans['primitive_normals']
        normal[~hit] = [0, -1, 0]
        normal = norm(normal)
        normal_map = np.clip(normal, -1., 1.)
        write_normal(normal_map.copy(), outdir)
        normal[~hit] = [0, 1, 0]

        xyz = rays[hit][:, :3] + rays[hit][:, 3:] * ans['t_hit'][hit].reshape(-1, 1)
        # pcd = o3d.t.geometry.PointCloud(xyz)
        # o3d.visualization.draw([pcd])
        xyz = xyz.numpy()
        xyz_map = np.zeros(normal_map.shape, dtype=np.float32)
        xyz_map[hit] = xyz
        write_xyz(xyz_map.copy(), outdir)

        lvis_hit = get_lvis(cam, scene, xyz, normal[hit])
        n_lights = lvis_hit.shape[1]

        lvis = np.zeros((h, w, n_lights), dtype=np.float32)
        hit_map = hit.reshape(h, w, 1)
        lvis[hit_map[:, :, 0]] = lvis_hit

        for i in range(lvis.shape[2]):
            lvis[:, :, i][~hit] = 0.

        write_lvis(lvis, 12, outdir)

        # exit(0)


def write_alpha(arr, out_dir):
    vis_out = join(out_dir, 'alpha.png')
    write_arr(arr, vis_out)

def write_normal(arr, out_dir):
    # convert to nerfactor convention
    arr[..., 1:] *= -1

    # Dump raw
    raw_out = join(out_dir, 'normal.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, arr)
    # Visualization
    vis_out = join(out_dir, 'normal.png')
    arr = (arr + 1) / 2
    write_arr(arr, vis_out)

def write_xyz(xyz_arr, out_dir):
    arr = xyz_arr
    arr[..., 1:] *= -1
    makedirs(out_dir)
    # Dump raw
    raw_out = join(out_dir, 'xyz.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, arr)
    # Visualization
    vis_out = join(out_dir, 'xyz.png')
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
    write_arr(arr_norm, vis_out, clip=True)

def write_lvis(lvis, fps, out_dir):
    makedirs(out_dir)
    # Dump raw
    raw_out = join(out_dir, 'lvis.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, lvis)
    # Visualize the average across all lights as an image
    vis_out = join(out_dir, 'lvis.png')
    lvis_avg = np.mean(lvis, axis=2)
    write_arr(lvis_avg, vis_out)
    # Visualize light visibility for each light pixel
    vis_out = join(out_dir, 'lvis.mp4')
    frames = []
    # for i in range(lvis.shape[2]): # for each light pixel
    #     frame = xm.img.denormalize_float(lvis[:, :, i])
    #     frame = np.dstack([frame] * 3)
    #     frames.append(frame)
    # xm.vis.video.make_video(frames, outpath=vis_out, fps=fps)




def write_arr(*args, **kwargs):
    return write_float(*args, **kwargs)

def write_float(arr_0to1, outpath, img_dtype='uint8', clip=False):
    r"""Writes a ``float`` array as an image to disk.

    Args:
        arr_0to1 (numpy.ndarray): Array with values roughly :math:`\in [0,1]`.
        outpath (str): Output path.
        img_dtype (str, optional): Image data type. Defaults to ``'uint8'``.
        clip (bool, optional): Whether to clip values to :math:`[0,1]`.
            Defaults to ``False``.

    Writes
        - The resultant image.

    Returns:
        numpy.ndarray: The resultant image array.
    """
    arr_min, arr_max = arr_0to1.min(), arr_0to1.max()
    if clip:
        if arr_max > 1:
            print("Maximum before clipping: %f", arr_max)
        if arr_min < 0:
            print("Minimum before clipping: %f", arr_min)
        arr_0to1 = np.clip(arr_0to1, 0, 1)
    else:
        assert arr_min >= 0 and arr_max <= 1, \
            "Input should be in [0, 1], or allow it to be clipped"

    # Float array to image
    img_arr = (arr_0to1 * np.iinfo(img_dtype).max).astype(img_dtype)

    write_uint(img_arr, outpath)

    return img_arr


def write_uint(arr_uint, outpath):
    if arr_uint.ndim == 3 and arr_uint.shape[2] == 1:
        arr_uint = np.dstack([arr_uint] * 3)

    img = Image.fromarray(arr_uint)

    # Write to disk
    makedirs(dirname(outpath))
    with open(outpath, 'wb') as h:
        img.save(h)

    print("Image written to:\n\t%s", outpath)


def make_video(
        imgs, fps=24, outpath=None, method='matplotlib', dpi=96, bitrate=-1):
    """Writes a list of images into a grayscale or color video.

    Args:
        imgs (list(numpy.ndarray)): Each image should be of type ``uint8`` or
            ``uint16`` and of shape H-by-W (grayscale) or H-by-W-by-3 (RGB).
        fps (int, optional): Frame rate.
        outpath (str, optional): Where to write the video to (a .mp4 file).
            ``None`` means
            ``os.path.join(const.Dir.tmp, 'make_video.mp4')``.
        method (str, optional): Method to use: ``'matplotlib'``, ``'opencv'``,
            ``'video_api'``.
        dpi (int, optional): Dots per inch when using ``matplotlib``.
        bitrate (int, optional): Bit rate in kilobits per second when using
            ``matplotlib``; reasonable values include 7200.

    Writes
        - A video of the images.
    """
    if outpath is None:
        outpath = join(const.Dir.tmp, 'make_video.mp4')
    makedirs(dirname(outpath))

    assert imgs, "Frame list is empty"
    for frame in imgs:
        assert np.issubdtype(frame.dtype, np.unsignedinteger), \
            "Image type must be unsigned integer"

    h, w = imgs[0].shape[:2]
    for frame in imgs[1:]:
        assert frame.shape[:2] == (h, w), \
            "All frames must have the same shape"

    if method == 'matplotlib':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import animation

        w_in, h_in = w / dpi, h / dpi
        fig = plt.figure(figsize=(w_in, h_in))
        Writer = animation.writers['ffmpeg'] # may require you to specify path
        writer = Writer(fps=fps, bitrate=bitrate)

        def img_plt(arr):
            img_plt_ = plt.imshow(arr)
            ax = plt.gca()
            ax.set_position([0, 0, 1, 1])
            ax.set_axis_off()
            return img_plt_

        anim = animation.ArtistAnimation(fig, [(img_plt(x),) for x in imgs])
        anim.save(outpath, writer=writer)
        # If obscure error like "ValueError: Invalid file object: <_io.Buff..."
        # occurs, consider upgrading matplotlib so that it prints out the real,
        # underlying ffmpeg error

        plt.close('all')
    else:
        raise ValueError(method)

    print("Images written as a video to:\n%s", outpath)





if __name__ == '__main__':
    scene = '/export/work/sanskara/nerfactor/antman1/'
    outdir = '/export/work/sanskara/nerfactor/antman_geom_3/'
    func(scene, outdir)
