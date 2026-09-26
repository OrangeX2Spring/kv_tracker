"""Object-mode Pi3 forward over the patches that touch the object mask only.

Object mode already zeroes every pixel outside the SAM mask; this skips those
patches instead of computing them. Kept patches keep their original encoder
position embedding and decoder RoPE position, frames keep their five register
tokens, and the KV cache stores only kept tokens. With every patch kept this is
the same computation as ``Pi3.forward`` without a token mask, split per frame.
Dropped patches have no prediction: points 0 and confidence 0.

A2 options: ``background`` marks kept patches outside the object (see
``background_keep``). With ``config["mass"]`` each such key gets an additive
attention bias log(n_background / n_kept_background) in every decoder attention,
so k representatives carry the softmax mass of the whole background (Co-Me's
mass correction). The DINOv2 encoder is unchanged. ``config["probe"]`` records
attention mass on the global (odd) decoder layers; it only reads.
"""
import json
import math

import torch

from pi3.utils.geometry import homogenize_points

PATCH = 14
REGISTER, OBJECT, BACKGROUND = 0, 1, 2
# Totals since the last reset; the ARCTIC driver reports them per sequence.
stats = dict(calls=0, kept=0, total=0, background=0)
# Run-level options, set by main.py from its flags before each run.
config = dict(mass=False, probe=None)


def patch_keep(mask):
    """(N, H, W) bool mask at Pi3 input size -> (N, h*w) patches touching it."""
    assert mask.dtype == torch.bool and mask.ndim == 3, (mask.dtype, mask.shape)
    N, H, W = mask.shape
    assert H % PATCH == 0 and W % PATCH == 0, mask.shape
    keep = mask.reshape(N, H // PATCH, PATCH, W // PATCH, PATCH).any(dim=4).any(dim=2)
    keep = keep.reshape(N, -1)
    assert keep.any(dim=1).all(), "every frame needs at least one object patch"
    return keep


def background_keep(keep, k):
    """Add k background patches per frame, evenly spaced in raster order.

    k=None keeps every background patch (the dense computation). Returns the new
    keep and the (N, h*w) mask of kept background patches.
    """
    assert keep.dtype == torch.bool and keep.ndim == 2, (keep.dtype, keep.shape)
    assert k is None or k > 0, k
    background = torch.zeros_like(keep)
    for n in range(len(keep)):
        free = (~keep[n]).nonzero()[:, 0]
        if k is None or k >= len(free):
            background[n, free] = True
        else:
            pick = torch.linspace(0, len(free) - 1, k, device=keep.device).round().long()
            background[n, free[pick]] = True
    return keep | background, background


class Probe:
    """Attention mass on the global decoder layers, one JSON line per call and layer.

    Probes every rebuild (store_cache) and the first cached query after it.
    """
    CHUNK = 128

    def __init__(self, path):
        self.file = open(path, "w", buffering=1)
        self.calls = 0
        self.after_rebuild = False

    def begin(self, store_cache, use_cache):
        self.calls += 1
        active = store_cache or (use_cache and self.after_rebuild)
        self.after_rebuild = store_cache
        self.kind = "rebuild" if store_cache else "query"
        return active

    def layer(self, layer, blk, x, xpos, cache, key_labels, key_distance, key_bias, frames):
        attn = blk.attn
        y = blk.norm1(x)
        B, L, C = y.shape
        # Mirrors FlashAttentionRope.forward_w_cache up to the softmax.
        qkv = attn.qkv(y).reshape(B, L, 3, attn.num_heads, C // attn.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, j] for j in range(3)]
        q, k = attn.q_norm(q).to(v.dtype), attn.k_norm(k).to(v.dtype)
        if attn.rope is not None:
            q, k = attn.rope(q, xpos), attn.rope(k, xpos)
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
        heads, keys = q.shape[1], k.shape[2]
        assert key_labels.shape == key_distance.shape == (keys,), (key_labels.shape, keys)
        query_labels = key_labels[-L:]
        k = k[0].float().transpose(-1, -2)
        mass = torch.zeros(3, keys, device=x.device)
        for start in range(0, L, self.CHUNK):
            score = q[0, :, start:start + self.CHUNK].float() @ k * q.shape[-1] ** -0.5
            if key_bias is not None:
                score = score + key_bias.float().reshape(-1)
            p = score.softmax(dim=-1).sum(dim=0)  # (chunk, keys), summed over heads
            labels = query_labels[start:start + self.CHUNK]
            for c in (REGISTER, OBJECT):
                mass[c] += p[labels == c].sum(dim=0)
        record = dict(call=self.calls, kind=self.kind, layer=layer, frames=frames, keys=keys,
                      cached_keys=keys - L)
        norms = x[0].float().norm(dim=-1)
        record["hidden_norm"] = {name: torch.quantile(norms[query_labels == c],
                                                      torch.tensor([.5, .9, .99, 1.], device=x.device)).tolist()
                                 for name, c in (("register", REGISTER), ("object", OBJECT),
                                                 ("background", BACKGROUND))
                                 if (query_labels == c).any()}
        is_bg = key_labels == BACKGROUND
        for name, c in (("register_queries", REGISTER), ("object_queries", OBJECT)):
            queries = int((query_labels == c).sum())
            share = mass[c] / (queries * heads)
            bg = share[is_bg]
            entry = dict(queries=queries,
                         register=float(share[key_labels == REGISTER].sum()),
                         object=float(share[key_labels == OBJECT].sum()),
                         background=float(bg.sum()))
            if bg.numel() and bg.sum() > 0:
                order = bg.sort(descending=True).values
                total = order.sum()
                probs = order / total
                entry.update(
                    background_keys=int(bg.numel()),
                    top4_share=float(order[:4].sum() / total),
                    top16_share=float(order[:16].sum() / total),
                    # exp(entropy) / count: 1 for uniform mass, small for sinks.
                    effective_fraction=float(torch.exp(-(probs * probs.clamp_min(1e-30).log()).sum())
                                             / bg.numel()),
                    top16_distance=key_distance[is_bg][bg.argsort(descending=True)[:16]].tolist())
                distance = key_distance[is_bg]
                entry["mass_by_distance"] = {
                    label: float(bg[(distance >= lo) & (distance <= hi)].sum() / total)
                    for label, lo, hi in (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5),
                                          ("6-10", 6, 10), (">10", 11, 10 ** 6))}
            record[name] = entry
        self.file.write(json.dumps(record) + "\n")


def forward_kept(model, imgs, keep, cam_only=False, store_cache=False, use_cache=False,
                 background=None):
    B, N, _, H, W = imgs.shape
    h, w = H // PATCH, W // PATCH
    assert B == 1 and H % PATCH == 0 and W % PATCH == 0, imgs.shape
    assert keep.dtype == torch.bool and keep.shape == (N, h * w), (keep.dtype, keep.shape)
    assert not (store_cache and use_cache)
    if background is None:
        background = torch.zeros_like(keep)
    assert background.shape == keep.shape and not (background & ~keep).any()
    encoder = model.encoder
    assert not encoder.chunked_blocks
    start = 1 + encoder.num_register_tokens
    special = model.patch_start_idx
    index = [row.nonzero()[:, 0] for row in keep]
    stats["calls"] += 1
    stats["kept"] += int(keep.sum())
    stats["total"] += keep.numel()
    stats["background"] += int(background.sum())

    # Per-token labels, with the mass correction the additive log-weight of each
    # key, and for the probe the Chebyshev patch distance to the frame's object.
    probe = config["probe"]
    probing = probe is not None and probe.begin(store_cache, use_cache)
    obj = keep & ~background
    if probe is not None:
        grid_ij = torch.stack(torch.meshgrid(torch.arange(h, device=keep.device),
                                             torch.arange(w, device=keep.device), indexing="ij"),
                              dim=-1).reshape(-1, 2)
    labels, distances, biases = [], [], []
    dtype = torch.bfloat16 if torch.is_autocast_enabled() else torch.float32
    for n in range(N):
        kept_bg = background[n][index[n]]
        labels.append(torch.cat([torch.full((special,), REGISTER, device=keep.device),
                                 torch.where(kept_bg, BACKGROUND, OBJECT)]))
        if probe is not None:
            near = (grid_ij[:, None] - grid_ij[obj[n]][None]).abs().amax(dim=-1).amin(dim=1)
            distances.append(torch.cat([torch.full((special,), -1, device=keep.device),
                                        near[index[n]]]))
        if config["mass"]:
            count = int(background[n].sum())
            assert count > 0, "the mass correction needs kept background"
            weight = math.log((h * w - int(obj[n].sum())) / count)
            biases.append((labels[-1] == BACKGROUND).to(dtype)[None, None, None] * weight)
    mass = biases if config["mass"] else [None] * N

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
    joint_labels = torch.cat(labels)
    joint_distance = torch.cat(distances) if probe is not None else None
    joint_bias = torch.cat(biases, dim=-1) if config["mass"] else None
    if use_cache:
        previous = model.kept_cache
        key_labels = torch.cat([previous["labels"], joint_labels])
        key_distance = (torch.cat([previous["distance"], joint_distance])
                        if probe is not None else None)
        key_bias = None if joint_bias is None else torch.cat([previous["bias"], joint_bias], dim=-1)
    else:
        key_labels, key_distance, key_bias = joint_labels, joint_distance, joint_bias
    if store_cache:
        model.kept_cache = dict(labels=joint_labels, distance=joint_distance, bias=joint_bias)
    last = []
    for i, blk in enumerate(model.decoder):
        if i % 2 == 0:
            hidden = [blk(x, xpos=p, attn_mask=b) for x, p, b in zip(hidden, pos, mass)]
        else:
            x, p = torch.cat(hidden, dim=1), torch.cat(pos, dim=1)
            if probing:
                probe.layer(i, blk, x, p, model.cache[i] if use_cache else None,
                            key_labels, key_distance, key_bias, N)
            if store_cache:
                x, k, v = blk(x, xpos=p, ret_kv=True, attn_mask=key_bias)
                model.cache[i] = {"k": k, "v": v}
            elif use_cache:
                x = blk(x, xpos=p, kv_cache=model.cache[i], ret_kv=False, attn_mask=key_bias)
            else:
                x = blk(x, xpos=p, attn_mask=key_bias)
            hidden = list(torch.split(x, lengths, dim=1))
        if i + 1 >= len(model.decoder) - 1:
            last.append(hidden)
    hidden = [torch.cat([a, b], dim=-1) for a, b in zip(*last)]

    poses = []
    for x, p, b in zip(hidden, pos, mass):
        camera = model.camera_decoder(x, xpos=p, attn_mask=b)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            poses.append(model.camera_head(camera.float()[:, special:], h, w))
    camera_poses = torch.cat(poses).reshape(1, N, 4, 4)
    if cam_only:
        return dict(camera_poses=camera_poses)

    def dense(decoder, head, dim):
        # Decoders run under the caller's autocast, heads in fp32, as in Pi3.forward.
        # Kept tokens are scattered into the full grid; dropped patches get zero features.
        out = []
        for n, (x, p, b) in enumerate(zip(hidden, pos, mass)):
            y = decoder(x, xpos=p, attn_mask=b)
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
