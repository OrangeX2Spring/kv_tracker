import cv2
import math
import numpy as np


def pi3_resize_image(image_np, target_dim):
    if image_np.ndim == 2:
        image_np = image_np[..., None]

    is_bool = image_np.dtype == bool
    if is_bool:
        image_np = image_np.astype(np.float32)

    PIXEL_LIMIT = target_dim[0] ** 2

    W_orig, H_orig = image_np.shape[1], image_np.shape[0]
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1

    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14

    image_np = cv2.resize(image_np, [TARGET_W, TARGET_H], interpolation=cv2.INTER_LINEAR)

    if is_bool:
        image_np = image_np > 0

    return image_np
