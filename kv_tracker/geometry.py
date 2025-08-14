import torch
import numpy as np


def get_local_pts3d_exact(mask, depth_map, proj_mat):
    """
    Back-project 2D points to 3D points in camera frame
    K^-1 . D . [u, v, 1]^T

    uv_coords: (N, 2)
    depth_map: (H, W)
    """

    valid_indices = torch.nonzero(mask)
    ray_depth = depth_map[valid_indices[:, 0], valid_indices[:, 1]].unsqueeze(1)

    uv_coords = valid_indices[:, [1, 0]]
    uv_homo = torch.cat([uv_coords, torch.ones_like(ray_depth)], dim=1)

    rays = proj_mat.inverse() @ uv_homo[..., None]
    pts_3d_cam = rays.squeeze() * ray_depth

    return pts_3d_cam

def normalise_pts3d(pts3d):
    """
    Normalises 3D points to the range [-1, 1]
    """
    min_vals = pts3d.min()
    max_vals = pts3d.max()
    return 2 * (pts3d - min_vals) / (max_vals - min_vals) - 1

def umeyama_alignment(x: np.ndarray, y: np.ndarray,
                      with_scale: bool = False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor

    Source:
    https://github.com/MichaelGrupp/evo/blob/master/evo/core/geometry.py
    """
    if x.shape != y.shape:
        raise ValueError("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise ValueError("Degenerate covariance rank, "
                         "Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c
