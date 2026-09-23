"""Object-mode Pi3 forward over the patches that touch the object mask only.

Object mode already zeroes every pixel outside the SAM mask; this skips those
patches instead of computing them. Kept patches keep their original encoder
position embedding and decoder RoPE position, frames keep their five register
tokens, and the KV cache stores only kept tokens. With every patch kept this is
the same computation as ``Pi3.forward`` without a token mask, split per frame.
Dropped patches have no prediction: points 0 and confidence 0.
"""
import torch

from pi3.utils.geometry import homogenize_points

PATCH = 14
# Totals since the last reset; the ARCTIC driver reports them per sequence.
stats = dict(calls=0, kept=0, total=0)


def patch_keep(mask):
    """(N, H, W) bool mask at Pi3 input size -> (N, h*w) patches touching it."""
    assert mask.dtype == torch.bool and mask.ndim == 3, (mask.dtype, mask.shape)
    N, H, W = mask.shape
    assert H % PATCH == 0 and W % PATCH == 0, mask.shape
    keep = mask.reshape(N, H // PATCH, PATCH, W // PATCH, PATCH).any(dim=4).any(dim=2)
    keep = keep.reshape(N, -1)
    assert keep.any(dim=1).all(), "every frame needs at least one object patch"
    return keep


def forward_kept(model, imgs, keep, cam_only=False, store_cache=False, use_cache=False):
    B, N, _, H, W = imgs.shape
    h, w = H // PATCH, W // PATCH
    assert B == 1 and H % PATCH == 0 and W % PATCH == 0, imgs.shape
    assert keep.dtype == torch.bool and keep.shape == (N, h * w), (keep.dtype, keep.shape)
    assert not (store_cache and use_cache)
    encoder = model.encoder
    assert not encoder.chunked_blocks
    start = 1 + encoder.num_register_tokens
    special = model.patch_start_idx
    index = [row.nonzero()[:, 0] for row in keep]
    stats["calls"] += 1
    stats["kept"] += int(keep.sum())
    stats["total"] += keep.numel()

    imgs = (imgs - model.image_mean) / model.image_std

    # Encoder: frames are independent, so each runs on its own kept tokens.
    tokens = []
    for n in range(N):
        x = encoder.prepare_tokens_with_masks(imgs[0, n:n + 1])
        x = torch.cat([x[:, :start], x[:, start:][:, index[n]]], dim=1)
        for blk in encoder.blocks:
            x = blk(x)
        tokens.append(encoder.norm(x)[:, start:])

    # Decoder: even blocks attend within a frame, odd blocks across frames and cache.
    grid = model.position_getter(1, h, w, imgs.device)[0] + 1
    zero = torch.zeros(special, 2, device=imgs.device, dtype=grid.dtype)
    pos = [torch.cat([zero, grid[i]])[None] for i in index]
    register = model.register_token.reshape(1, special, -1)
    hidden = [torch.cat([register, t], dim=1) for t in tokens]
    lengths = [x.shape[1] for x in hidden]
    last = []
    for i, blk in enumerate(model.decoder):
        if i % 2 == 0:
            hidden = [blk(x, xpos=p) for x, p in zip(hidden, pos)]
        else:
            x, p = torch.cat(hidden, dim=1), torch.cat(pos, dim=1)
            if store_cache:
                x, k, v = blk(x, xpos=p, ret_kv=True)
                model.cache[i] = {"k": k, "v": v}
            elif use_cache:
                x = blk(x, xpos=p, kv_cache=model.cache[i], ret_kv=False)
            else:
                x = blk(x, xpos=p)
            hidden = list(torch.split(x, lengths, dim=1))
        if i + 1 >= len(model.decoder) - 1:
            last.append(hidden)
    hidden = [torch.cat([a, b], dim=-1) for a, b in zip(*last)]

    poses = []
    for x, p in zip(hidden, pos):
        camera = model.camera_decoder(x, xpos=p)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            poses.append(model.camera_head(camera.float()[:, special:], h, w))
    camera_poses = torch.cat(poses).reshape(1, N, 4, 4)
    if cam_only:
        return dict(camera_poses=camera_poses)

    def dense(decoder, head, dim):
        # Decoders run under the caller's autocast, heads in fp32, as in Pi3.forward.
        # Kept tokens are scattered into the full grid; dropped patches get zero features.
        out = []
        for n, (x, p) in enumerate(zip(hidden, pos)):
            y = decoder(x, xpos=p)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                y = y.float()[:, special:]
                full = y.new_zeros(1, h * w, y.shape[-1])
                full[:, index[n]] = y
                out.append(head([full], (H, W)).reshape(1, H, W, dim))
        return torch.cat(out).reshape(1, N, H, W, dim)

    pixel = keep.reshape(N, h, 1, w, 1).expand(N, h, PATCH, w, PATCH).reshape(1, N, H, W, 1)
    ret = dense(model.point_decoder, model.point_head, 3)
    conf = dense(model.conf_decoder, model.conf_head, 1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        xy, z = ret.split([2, 1], dim=-1)
        z = torch.exp(z)
        local_points = torch.cat([xy * z, z], dim=-1) * pixel
        # Large negative logit: sigmoid gives confidence 0 for dropped patches.
        conf = torch.where(pixel, conf, torch.full_like(conf, -1e4))
        points = torch.einsum("bnij, bnhwj -> bnhwi", camera_poses,
                              homogenize_points(local_points))[..., :3] * pixel
    return dict(points=points, local_points=local_points, conf=conf, camera_poses=camera_poses)
