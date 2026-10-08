"""Opt-in recurrent Pi3 experiments. No pretrained quality or speed guarantee."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import inspect
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

from pi3.models.pi3 import Pi3
from pi3.curope.curope2d import cuRoPE2D, curope_backend

VARIANTS = ('elastic', 'relaxed', 'shared_kv', 'adaptive', 'token', 'refiner', 'nested', 'combined')
PI3_SHA256 = 'cbcf68b3c05baab7680f6e24afda42501dc0ba799e85ca18668ba3e8a5812979'
PROJECTIONS = ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')


@dataclass(frozen=True)
class LoopConfig:
    variant: str
    core_pairs: int = 4
    full_loops: int = 4
    short_loops: int = 2
    rank: int = 16
    active_fraction: float = .5
    refiner_dim: int = 256
    refiner_steps: int = 3
    map_tokens_per_frame: int = 64

    def __post_init__(self):
        assert self.variant in VARIANTS + ('native', 'native_compact')
        assert 2 * self.core_pairs * self.full_loops + 4 == 36
        assert self.short_loops > 0 and self.full_loops % self.short_loops == 0
        assert self.rank > 0 and 0 < self.active_fraction <= 1
        assert self.refiner_dim % 8 == 0 and self.refiner_steps > 0
        assert self.map_tokens_per_frame > 0


class _TrainingRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, positions, base, frequency):
        ctx.save_for_backward(positions)
        ctx.base, ctx.frequency = base, frequency
        # The CUDA kernel mutates B,N,H,D storage; never alias its caller.
        buffer = tokens.transpose(1, 2).clone(memory_format=torch.contiguous_format)
        curope_backend.rope_2d(buffer, positions, base, frequency)
        return buffer.transpose(1, 2).contiguous()

    @staticmethod
    def backward(ctx, gradient):
        positions, = ctx.saved_tensors
        # Clone even a contiguous gradient: other graph branches may share it.
        buffer = gradient.transpose(1, 2).clone(memory_format=torch.contiguous_format)
        curope_backend.rope_2d(buffer, positions, ctx.base, -ctx.frequency)
        return buffer.transpose(1, 2).contiguous(), None, None, None


class TrainingRoPE(cuRoPE2D):
    """Native inference with an owned contiguous buffer for training backward."""
    def forward(self, tokens, positions):
        if torch.is_grad_enabled() and tokens.requires_grad:
            return _TrainingRoPEFunction.apply(tokens, positions, self.base, self.F0)
        return super().forward(tokens, positions)


class LowRankLinear(nn.Module):
    """A shared dense projection plus an unmerged depth-specific residual."""
    def __init__(self, base, original, rank, initialize=True):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.down = nn.Parameter(base.weight.new_zeros(rank, self.in_features))
        self.up = nn.Parameter(base.weight.new_zeros(self.out_features, rank))
        self.bias_delta = nn.Parameter(torch.zeros_like(base.bias))
        if initialize:
            with torch.no_grad():
                # Randomized truncated SVD avoids a full large projection SVD.
                delta = original.weight.float() - base.weight.float()
                u, s, v = torch.svd_lowrank(delta, q=min(rank + 8, min(delta.shape)), niter=2)
                self.up.copy_(u[:, :rank] * s[:rank])
                self.down.copy_(v[:, :rank].T)
                self.bias_delta.copy_(original.bias - base.bias)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.down), self.up, self.bias_delta)


def active_rows(scores, fraction, special=5):
    """Half top scores, half spatial spread, plus every register; query N=1."""
    assert scores.ndim == 1 and scores.numel() > 0
    count = max(1, round(scores.numel() * fraction))
    top = scores.topk(count // 2).indices
    available = torch.ones_like(scores, dtype=torch.bool)
    available[top] = False
    free = available.nonzero()[:, 0]
    slots = torch.linspace(0, len(free) - 1, count - len(top), device=scores.device).round().long()
    patches = torch.cat((top, free[slots])).sort().values + special
    return torch.cat((torch.arange(special, device=scores.device), patches))


def selected_block(blk, x, pos, rows, cache=None):
    """Update compact query/MLP rows; inactive current tokens remain K/V context."""
    assert x.ndim == 3 and x.shape[0] == 1 and rows.dtype == torch.long
    assert blk.sample_drop_ratio == 0 and isinstance(blk.attn.qkv, nn.Linear)
    attn = blk.attn
    y = blk.norm1(x)
    width, heads = x.shape[-1], attn.num_heads
    weight, bias = attn.qkv.weight, attn.qkv.bias
    q = F.linear(y[:, rows], weight[:width], bias[:width])
    kv = F.linear(y, weight[width:], bias[width:])
    q = q.reshape(1, len(rows), heads, width // heads).transpose(1, 2)
    kv = kv.reshape(1, x.shape[1], 2, heads, width // heads).transpose(1, 3)
    k, v = kv[:, :, 0], kv[:, :, 1]
    q, k = attn.q_norm(q).to(v.dtype), attn.k_norm(k).to(v.dtype)
    q, k = attn.rope(q, pos[:, rows]), attn.rope(k, pos)
    if cache is not None:
        k, v = torch.cat((cache['k'], k), 2), torch.cat((cache['v'], v), 2)
    # Match native FlashAttentionRope's backend policy for both tested dtypes.
    backend = (SDPBackend.FLASH_ATTENTION if q.dtype == torch.bfloat16
               else [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
    with sdpa_kernel(backend):
        update = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, len(rows), width)
    current = x[:, rows] + blk.ls1(attn.proj_drop(attn.proj(update)))
    current = current + blk.ls2(blk.mlp(blk.norm2(current)))
    return x.index_copy(1, rows, current)


def pose_distance(a, b, scale):
    """Translation in map-scale units; rotation chordal distance, not degrees."""
    assert a.shape == b.shape == (4, 4)
    return torch.stack(((a[:3, 3] - b[:3, 3]).norm() / scale,
                        (a[:3, :3] - b[:3, :3]).norm() / (2 ** .5)))


def se3_update(twist, scale):
    """Body-frame SE(3) exponential; differentiable even at zero rotation."""
    assert twist.shape == (6,) and twist.dtype == torch.float32
    x, y, z = twist[3:].unbind()
    zero = x * 0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    algebra = twist.new_zeros(4, 4)
    algebra[:3, :3] = skew
    algebra[:3, 3] = twist[:3] * scale
    return torch.matrix_exp(algebra)


class PoseRefiner(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        dim = config.refiner_dim
        self.query = nn.Linear(input_dim, dim)
        self.memory = nn.Linear(input_dim + 3, dim)
        self.pose_condition = nn.Linear(12, dim)
        self.cross = nn.MultiheadAttention(dim, 8, batch_first=True, dropout=0.)
        self.block = nn.TransformerEncoderLayer(dim, 8, 2 * dim, dropout=0., batch_first=True,
                                               norm_first=True)
        self.readout = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 6))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, features, memory_features, points, pose, scale, steps):
        state = self.query(features)
        for _ in range(steps):
            with torch.autocast('cuda', enabled=False):
                camera = (points.float() - pose[:3, 3]) @ pose[:3, :3] / scale
                normalized_pose = torch.cat((pose[:3, :3], pose[:3, 3:4] / scale), 1)
            memory = self.memory(torch.cat((memory_features, camera[None].to(memory_features.dtype)), -1))
            condition = self.pose_condition(normalized_pose.reshape(1, 1, 12).to(state.dtype))
            attended = self.cross(state + condition, memory, memory, need_weights=False)[0]
            state = self.block(state + attended)
            twist = self.readout(state.mean(1))[0].float()
            with torch.autocast('cuda', enabled=False):
                pose = pose @ se3_update(twist, scale)
        return pose


class LoopedPi3(Pi3):
    """Reuses pinned native heads/forward; replaces decoder execution only.

    Rebuilds always use the full trained schedule. Query short paths read their
    matching full-schedule cache slots. Historical cache tensors are never written
    by queries. Dense input only: this pilot does not combine stock token_drop,
    cache transforms, or native decoder-index hooks with a new architecture.
    """
    def __init__(self, native, config, initialize=True):
        nn.Module.__init__(self)
        assert hashlib.sha256(Path(inspect.getfile(Pi3)).read_bytes()).hexdigest() == PI3_SHA256
        assert len(native.decoder) == 36 and native.patch_start_idx == 5
        self.config = config
        for name, module in native.named_children():
            if name != 'decoder':
                self.add_module(name, module)
        for name, param in native.named_parameters(recurse=False):
            self.register_parameter(name, param)
        for name, tensor in native.named_buffers(recurse=False):
            self.register_buffer(name, tensor)
        for name in ('patch_size', 'pos_type', 'position_getter', 'patch_start_idx', 'dec_embed_dim'):
            setattr(self, name, getattr(native, name))
        self.cache = {}
        self.kept_cache = None
        self.query_loops = config.short_loops
        self.collect_exits = False
        self.record_details = False
        self.profile_compact = False
        self.compact_events = []
        self.checkpoint_blocks = True
        self.geometry_calibration = None
        self.exit_records = []
        self.last_features = self.routing_features = self.routing_scores = None
        self.exit_features = {}
        self.last_execution = {}
        self.register_buffer('halt_thresholds', torch.full((3,), -1., device=native.image_mean.device))
        if config.variant in ('refiner', 'native', 'native_compact'):
            self.decoder = native.decoder
            self.requires_grad_(False)
            if config.variant == 'refiner':
                self.refiner = PoseRefiner(config, native.dec_embed_dim).to(native.image_mean.device)
                self.refiner_state = None
            return
        self.entry = nn.ModuleList(deepcopy(list(native.decoder[:2])))
        self.exit = nn.ModuleList(deepcopy(list(native.decoder[-2:])))
        self.core = nn.ModuleList()
        for slot in range(2 * config.core_pairs):
            if config.variant == 'nested' and slot % 4 == 1:
                self.core.append(nn.Identity())
                continue
            originals = [native.decoder[2 + 2 * config.core_pairs * r + slot]
                         for r in range(config.full_loops)]
            block = deepcopy(originals[0])
            if initialize:
                with torch.no_grad():
                    for name, value in block.named_parameters():
                        value.copy_(torch.stack([dict(b.named_parameters())[name].float()
                                                for b in originals]).mean(0))
            self.core.append(block)
        self.depth_blocks = nn.ModuleList()
        if config.variant in ('relaxed', 'combined'):
            for r in range(config.full_loops):
                for slot, core in enumerate(self.core):
                    original = native.decoder[2 + len(self.core) * r + slot]
                    block = deepcopy(original)
                    for path in PROJECTIONS:
                        owner, name = path.split('.')
                        shared = getattr(getattr(core, owner), name)
                        source = getattr(getattr(original, owner), name)
                        setattr(getattr(block, owner), name,
                                LowRankLinear(shared, source, config.rank, initialize))
                    self.depth_blocks.append(block)
        self.requires_grad_(False)
        self.core.requires_grad_(True)
        self.depth_blocks.requires_grad_(True)
        self.core.float()
        self.depth_blocks.float()
        self.condition = nn.Linear(2, native.dec_embed_dim).to(native.image_mean.device)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)
        self.router = nn.Sequential(nn.LayerNorm(native.dec_embed_dim),
                                    nn.Linear(native.dec_embed_dim, 1)).to(native.image_mean.device)
        self.router.requires_grad_(config.variant == 'token')
        # Frozen heads still backpropagate to decoder features. Copy module shells
        # while sharing frozen tensors, so installing RoPE does not mutate native.
        memo = {id(tensor): tensor for tensor in (*native.parameters(), *native.buffers())}
        for name in ('camera_decoder', 'point_decoder', 'conf_decoder'):
            setattr(self, name, deepcopy(getattr(self, name), memo))
        for collection in (self.entry, self.exit, self.core, self.depth_blocks,
                           self.camera_decoder, self.point_decoder, self.conf_decoder):
            for module in collection.modules():
                if isinstance(getattr(module, 'rope', None), cuRoPE2D):
                    module.rope = TrainingRoPE(module.rope.base, module.rope.F0)

    def train(self, mode=True):
        super().train(mode)
        # The frozen visual backbone and heads retain their pretrained eval behavior.
        self.encoder.eval()
        self.camera_decoder.eval()
        self.point_decoder.eval()
        self.conf_decoder.eval()
        if not mode:
            self.last_features = self.routing_features = self.routing_scores = None
            self.exit_features = {}
        return self

    def encode(self, imgs):
        b, n, _, h, w = imgs.shape
        encoded = self.encoder(((imgs - self.image_mean) / self.image_std).reshape(b * n, 3, h, w),
                               is_training=True)
        return encoded['x_norm_patchtokens']

    def read_pose(self, hidden, pos, h, w):
        features = self.camera_decoder(hidden, xpos=pos, attn_mask=None)
        with torch.autocast('cuda', enabled=False):
            return self.camera_head(features.float()[:, self.patch_start_idx:], h // 14, w // 14)

    def read_local(self, hidden, pos, h, w):
        features = self.point_decoder(hidden, xpos=pos, attn_mask=None)
        with torch.autocast('cuda', enabled=False):
            raw = self.point_head([features.float()[:, self.patch_start_idx:]], (h, w))
            xy, z = raw.reshape(-1, h, w, 3).split((2, 1), -1)
            return torch.cat((xy * z.exp(), z.exp()), -1)

    def block(self, blk, x, pos, global_block, key, store, use, rows=None):
        if global_block:
            x, pos = x.reshape(1, -1, x.shape[-1]), pos.reshape(1, -1, 2)
        historical = self.cache[key] if global_block and use else None
        if rows is not None:
            if self.profile_compact:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            result = selected_block(blk, x, pos, rows, historical)
            if self.profile_compact:
                end.record()
                self.compact_events.append(('global' if global_block else 'local', start, end))
        elif global_block and store:
            # Shared cache is generated once; later recurrences consume that bank
            # on queries, but must not duplicate it inside the joint rebuild.
            # A grad-enabled training rebuild keeps the bank in the query's graph.
            result, k, v = blk(x, xpos=pos, ret_kv=True)
            if key not in self.cache:
                self.cache[key] = dict(k=k.contiguous().clone(), v=v.contiguous().clone())
        else:
            def apply(value):
                return blk(value, xpos=pos, kv_cache=historical, ret_kv=False)
            result = (checkpoint(apply, x, use_reentrant=False)
                      if torch.is_grad_enabled() and self.checkpoint_blocks else apply(x))
        return result

    def cache_key(self, loop, pair):
        shared = self.config.variant in ('shared_kv', 'combined')
        return f'core_{loop // 2 if shared else loop}_{pair}'

    def end_pair(self, x, pos, n, h, w, store=False, use=False):
        before = self.block(self.exit[0], x, pos, False, 'unused', False, False)
        after = self.block(self.exit[1], before, pos, True, 'exit', store, use)
        return torch.cat((before.reshape(n, -1, self.dec_embed_dim),
                          after.reshape(n, -1, self.dec_embed_dim)), -1)

    def geometry_residual(self, hidden, pos, pose, h, w):
        from .scene_routing import project_cache
        intrinsics, distortion = self.geometry_calibration
        with torch.autocast('cuda', enabled=False):
            projected = project_cache(self.map_points, self.map_confidence, pose.float(),
                                      intrinsics, h, w, distortion)
        local = self.read_local(hidden, pos, h, w)[0, 7::14, 7::14, 2].flatten()
        supported = ((projected['support'] > 0) & (projected['confidence'] > .5)
                     & torch.isfinite(local) & (local > 0))
        if not supported.any():
            return local.new_tensor(float('inf'))
        return ((local[supported] - projected['front_depth'][supported]).abs()
                / projected['front_depth'][supported]).median()

    def decode(self, hidden, N, H, W, store_cache=False, use_cache=False, tokens_mask=None):
        assert tokens_mask is None and hidden.shape[0] == N
        assert not (store_cache and use_cache)
        assert H % 14 == W % 14 == 0
        if self.config.variant in ('refiner', 'native', 'native_compact'):
            if self.config.variant == 'refiner' and store_cache:
                self.encoded_bank = hidden.detach()
            return super().decode(hidden, N, H, W, store_cache, use_cache, tokens_mask)
        if store_cache:
            self.cache = {}
        registers = self.register_token.reshape(1, 5, -1).expand(N, -1, -1)
        x = torch.cat((registers, hidden), 1)
        length = x.shape[1]
        pos = self.position_getter(N, H // 14, W // 14, x.device) + 1
        pos = torch.cat((pos.new_zeros(N, 5, 2), pos), 1)
        x = self.block(self.entry[0], x, pos, False, 'unused', False, False)
        x = self.block(self.entry[1], x, pos, True, 'entry', store_cache, use_cache)
        x = x.reshape(N, length, -1)
        loops = self.config.full_loops if store_cache or not use_cache else self.query_loops
        if self.config.variant in ('relaxed', 'token', 'nested'):
            loops = self.config.full_loops
        adaptive = self.config.variant == 'adaptive' and use_cache
        if adaptive:
            loops = self.config.full_loops
        schedule = [(step + 1) * self.config.full_loops // loops - 1 for step in range(loops)]
        self.exit_records = []
        self.exit_features = {}
        self.routing_features = self.routing_scores = None
        self.last_execution = dict(loops=0, local_blocks=0, global_blocks=0, active_rows=[],
                                   halted=False, cache_slots=0)
        previous_pose = None
        for step, r in enumerate(schedule):
            t = x.new_tensor([step / loops, 1 / loops])
            rows = None
            if self.config.variant == 'token' and use_cache and step > 0:
                assert N == 1
                if self.routing_features is None:
                    self.routing_features = x[:, 5:]
                    self.routing_scores = self.router(self.routing_features).flatten()
                rows = active_rows(self.routing_scores, self.config.active_fraction)
                if self.record_details:
                    self.last_execution['active_rows'].append(rows.detach().cpu().tolist())
            delta = self.condition(t).to(x.dtype)
            x = (x + delta if rows is None else
                 x.index_add(1, rows, delta.reshape(1, 1, -1).expand(1, len(rows), -1)))
            blocks = (self.depth_blocks if self.depth_blocks else self.core)
            offset = r * len(self.core) if self.depth_blocks else 0
            for pair in range(self.config.core_pairs):
                slot = offset + 2 * pair
                x = self.block(blocks[slot], x, pos, False, 'unused', False, False, rows)
                self.last_execution['local_blocks'] += 1
                if self.config.variant != 'nested' or pair % 2 == 1:
                    x = self.block(blocks[slot + 1], x, pos, True, self.cache_key(r, pair),
                                   store_cache, use_cache, rows)
                    self.last_execution['global_blocks'] += 1
                x = x.reshape(N, length, -1)
            self.last_execution['loops'] += 1
            if (adaptive and self.training
                    and self.config.short_loops <= step + 1 < self.config.full_loops):
                self.exit_features[step + 1] = self.end_pair(x, pos, N, H, W, use=True)
            if adaptive and step + 1 < loops and (self.collect_exits or not self.training):
                assert N == 1 and self.geometry_calibration is not None
                candidate = self.end_pair(x, pos, N, H, W, use=True)
                pose = self.read_pose(candidate, pos, H, W).reshape(4, 4)
                residual = (self.geometry_residual(candidate, pos, pose, H, W)
                            if step + 1 >= self.config.short_loops
                            else pose.new_tensor(float('inf')))
                change = (pose_distance(pose, previous_pose, self.map_scale)
                          if previous_pose is not None else pose.new_full((2,), float('inf')))
                signals = torch.cat((change, residual[None]))
                self.exit_records.append(dict(loop=step + 1, signals=signals.detach(),
                                              pose=pose.detach()))
                previous_pose = pose.detach()
                if (not self.collect_exits and (self.halt_thresholds >= 0).all()
                        and step + 1 >= self.config.short_loops
                        and bool((signals <= self.halt_thresholds).all())):
                    self.last_execution['halted'] = step + 1 < loops
                    self.last_features = candidate
                    self.last_execution['cache_slots'] = len(self.cache)
                    return candidate, pos, None
        output = self.end_pair(x, pos, N, H, W, store_cache, use_cache)
        self.last_features = output
        self.last_execution['cache_slots'] = len(self.cache)
        return output, pos, None

    def forward(self, imgs, cam_only=False, store_cache=False, use_cache=False, tokens_mask=None, **kwargs):
        assert imgs.ndim == 5 and imgs.shape[0] == 1 and tokens_mask is None and not kwargs
        assert not (store_cache and use_cache)
        if self.config.variant == 'refiner' and use_cache:
            assert cam_only and imgs.shape[1] == 1 and self.refiner_state is not None
            encoded = self.encode(imgs)
            start_pose = self.refiner_state
            if self.refiner_previous is not None:
                with torch.autocast('cuda', enabled=False):
                    start_pose = start_pose @ (torch.linalg.inv(self.refiner_previous) @ start_pose)
            pose = self.refiner(encoded, self.memory_features, self.memory_points,
                                start_pose, self.map_scale, self.config.refiner_steps)
            self.refiner_previous = self.refiner_state
            # Grad-enabled training replays backpropagate through the rollout.
            self.refiner_state = pose
            self.last_execution = dict(refiner_steps=self.config.refiner_steps,
                                       map_tokens=self.memory_points.shape[0])
            return dict(camera_poses=pose.reshape(1, 1, 4, 4))
        if self.config.variant == 'adaptive' and use_cache and not self.training and not self.collect_exits:
            assert (self.halt_thresholds >= 0).all(), 'Adaptive deployment requires calibrated thresholds'
        output = super().forward(imgs, cam_only, store_cache, use_cache, tokens_mask)
        if store_cache and self.config.variant == 'native_compact':
            self.cache = {slot: {key: value.contiguous().clone() for key, value in row.items()}
                          for slot, row in self.cache.items()}
        if store_cache and self.config.variant not in ('native', 'native_compact'):
            assert not cam_only
            points = output['points'][0, :, 7::14, 7::14].reshape(-1, 3).detach()
            self.map_scale = (points - points.mean(0)).norm(dim=-1).median().clamp_min(1e-6)
            if self.config.variant == 'adaptive':
                self.map_points = points.clone()
                self.map_confidence = output['conf'][0, :, 7::14, 7::14, 0].sigmoid().flatten().detach().clone()
            if self.config.variant == 'refiner':
                cells = self.encoded_bank.shape[1]
                count = min(cells, self.config.map_tokens_per_frame)
                picks = torch.linspace(0, cells - 1, count, device=imgs.device).round().long()
                n = imgs.shape[1]
                world = points.reshape(n, cells, 3)
                self.memory_points = world[:, picks].reshape(-1, 3).clone()
                self.memory_features = self.encoded_bank[:, picks].reshape(1, -1, self.dec_embed_dim).clone()
                self.refiner_state = output['camera_poses'][0, -1].detach().float().clone()
                self.refiner_previous = None
                del self.encoded_bank
        if not self.training:
            self.last_features = self.routing_features = self.routing_scores = None
            self.exit_features = {}
        return output

    def storage_report(self):
        cache = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                 for row in self.cache.values() for t in row.values()}
        weights = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                   for t in self.parameters()}
        auxiliary = []
        if self.cache and self.config.variant not in ('native', 'native_compact'):
            auxiliary.append(self.map_scale)
            if self.config.variant == 'adaptive':
                auxiliary += [self.map_points, self.map_confidence]
        if self.config.variant == 'refiner' and self.refiner_state is not None:
            auxiliary += [self.memory_points, self.memory_features, self.refiner_state]
            if self.refiner_previous is not None:
                auxiliary.append(self.refiner_previous)
        return dict(unique_weight_bytes=sum(weights.values()), unique_kv_bytes=sum(cache.values()),
                    auxiliary_tensor_bytes=sum(t.numel() * t.element_size() for t in auxiliary),
                    cache_slots=len(self.cache))

    def save(self, path, provenance):
        prefixes = ('core.', 'depth_blocks.', 'condition.', 'refiner.')
        if self.config.variant == 'token':
            prefixes += ('router.',)
        state = {k: v for k, v in self.state_dict().items()
                 if k.startswith(prefixes) or k == 'halt_thresholds'}
        torch.save(dict(format=1, config=asdict(self.config), state_dict=state,
                        pi3_sha256=PI3_SHA256, provenance=provenance), path)


def load_looped(native, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['format'] == 1 and saved['pi3_sha256'] == PI3_SHA256
    model = LoopedPi3(native, LoopConfig(**saved['config']), initialize=False)
    expected = model.state_dict()
    assert all(k in expected and expected[k].shape == v.shape for k, v in saved['state_dict'].items())
    incompatible = model.load_state_dict(saved['state_dict'], strict=False)
    assert not incompatible.unexpected_keys
    trained = ('core.', 'depth_blocks.', 'condition.', 'refiner.')
    if model.config.variant == 'token':
        trained += ('router.',)
    assert not any(k.startswith(trained) or k == 'halt_thresholds' for k in incompatible.missing_keys)
    return model.eval(), saved['provenance']
