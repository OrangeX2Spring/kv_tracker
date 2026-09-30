"""Experimental matrix-level cache readers, independent of Pi3/model imports.

Inputs are one head's post-normalization/post-position-encoding Q/K/V. All
historical keys are visible; the caller supplies only causally eligible history.
Current keys remain exact. Construction/calibration is separate from reads.
Probe balancing is a finite-probe greedy discrepancy heuristic, not BalanceKV.
"""
import heapq
import math

import torch


def tensor_bytes(tree):
    if isinstance(tree, torch.Tensor):
        return tree.untyped_storage().nbytes()
    if isinstance(tree, dict):
        return sum(tensor_bytes(x) for x in tree.values())
    if isinstance(tree, (tuple, list)):
        return sum(tensor_bytes(x) for x in tree)
    return 0


def dense_response(q, k, v, log_weights=None, scale=None):
    assert q.ndim == k.ndim == v.ndim == 2
    assert q.shape[1] == k.shape[1] and k.shape[0] == v.shape[0] > 0
    logits = q.float() @ k.float().T * (q.shape[1] ** -.5 if scale is None else scale)
    if log_weights is not None:
        logits = logits + log_weights
    return logits.softmax(-1) @ v.float(), logits.logsumexp(-1)


def merge_responses(a, log_a, b, log_b):
    weights = torch.stack((log_a, log_b), -1).softmax(-1)
    return weights[:, :1] * a + weights[:, 1:] * b


def pack_codes(codes, bits):
    """One channel, unsigned codes, truly packed into uint8 storage."""
    assert codes.ndim == 1 and 1 <= bits <= 8
    positions = torch.arange(len(codes), device=codes.device) * bits
    packed = torch.zeros(math.ceil(len(codes) * bits / 8), device=codes.device,
                         dtype=torch.int64)
    for bit in range(bits):
        target = positions + bit
        packed.scatter_add_(0, target // 8, ((codes.long() >> bit) & 1) << (target % 8))
    return packed.to(torch.uint8)


def unpack_codes(packed, bits, count):
    positions = torch.arange(count, device=packed.device) * bits
    result = torch.zeros(count, device=packed.device, dtype=torch.int64)
    for bit in range(bits):
        source = positions + bit
        result |= ((packed[source // 8].long() >> (source % 8)) & 1) << bit
    return result


def allocate_bits(alpha, budget, maximum=8):
    """Exact integer allocation for sum alpha*4**(-bits), unit-bit increments.

    Greedy marginal gains solve this separable diminishing-returns model.
    This does not claim optimal distortion for real nonuniform rounding noise.
    """
    values = alpha.detach().double().cpu().tolist()
    assert 0 <= budget <= len(values) * maximum
    bits = [0] * len(values)
    heap = [(-.75 * value, i) for i, value in enumerate(values)]
    heapq.heapify(heap)
    for _ in range(budget):
        _, i = heapq.heappop(heap)
        bits[i] += 1
        if bits[i] < maximum:
            heapq.heappush(heap, (-.75 * values[i] * 4. ** (-bits[i]), i))
    return torch.tensor(bits, dtype=torch.int32, device=alpha.device)


class PackedMatrix:
    def __init__(self, x, bits):
        assert x.ndim == 2 and bits.shape == (x.shape[1],)
        self.count = len(x)
        mean = x.float().mean(0)
        centered = x.float() - mean
        span = centered.abs().amax(0)
        widths = bits.cpu().tolist()
        scale = torch.zeros_like(mean)
        offset = mean.clone()
        codes = []
        for i, width in enumerate(widths):
            if width == 0:
                codes.append(torch.empty(0, device=x.device, dtype=torch.uint8))
                continue
            scale[i] = 2 * span[i] / (2 ** width - 1)
            offset[i] -= span[i]
            divisor = torch.where(scale[i] == 0, torch.ones_like(scale[i]), scale[i])
            code = ((x[:, i].float() - offset[i]) / divisor).round().clamp(0, 2 ** width - 1)
            codes.append(pack_codes(code.long(), width))
        self.state = dict(bits=bits.clone(), scale=scale, offset=offset, codes=codes)
        self.widths = widths

    def decode(self):
        s = self.state
        x = s['offset'].expand(self.count, -1).clone()
        for i, width in enumerate(self.widths):
            if width:
                x[:, i] += unpack_codes(s['codes'][i], width, self.count) * s['scale'][i]
        return x


class MathCache:
    METHODS = ('exact', 'uniform', 'probe_balance', 'kernel', 'centroid',
               'moments', 'lowrank', 'query_lowrank', 'quant', 'waterfill')

    def __init__(self, config):
        self.config = dict(config)
        self.method = self.config['method']
        assert self.method in self.METHODS
        self.state = {}

    @torch.no_grad()
    def fit(self, k, v, calibration_q):
        assert k.ndim == v.ndim == calibration_q.ndim == 2
        assert len(k) == len(v) > 0 and k.shape[1] == calibration_q.shape[1]
        assert k.device == v.device == calibration_q.device
        assert k.is_floating_point() and v.is_floating_point() and calibration_q.is_floating_point()
        assert len(calibration_q) > 0
        self.dk, self.dv, self.count = k.shape[1], v.shape[1], len(k)
        self.scale = self.dk ** -.5
        method, cfg = self.method, self.config
        if method == 'exact':
            self.state = dict(k=k.clone(), v=v.clone())
        elif method in ('uniform', 'probe_balance'):
            budget = min(cfg['tokens'], len(k))
            assert budget > 0
            if method == 'uniform':
                index = torch.linspace(0, len(k) - 1, budget, device=k.device).round().long()
                weights = torch.full((budget,), len(k) / budget, device=k.device)
            else:
                probes = calibration_q.float()
                logits = probes @ k.float().T * self.scale
                p = (logits - logits.logsumexp(-1, keepdim=True)).exp()
                features = (p.T[:, :, None] * torch.cat((v.float(),
                    v.new_ones(len(v), 1).float()), -1)[:, None]).flatten(1).double().cpu()
                # CPU construction avoids thousands of serialized GPU launches.
                index = torch.arange(len(k))
                weights = torch.ones(len(k), dtype=torch.float64)
                while len(index) > budget:
                    order = (features[index] * weights[:, None]).norm(dim=1).argsort()
                    paired = min(len(index) // 2, len(index) - budget)
                    chosen, updated = [], []
                    discrepancy = torch.zeros(features.shape[1], dtype=torch.float64)
                    for pair in range(paired):
                        ia, ib = order[2 * pair:2 * pair + 2].tolist()
                        # Equal-weight pairs arise from each halving level. Odd
                        # tails are retained; differing masses use weighted means.
                        total = weights[ia] + weights[ib]
                        error_a = total * features[index[ia]] - (
                            weights[ia] * features[index[ia]] + weights[ib] * features[index[ib]])
                        error_b = total * features[index[ib]] - (
                            weights[ia] * features[index[ia]] + weights[ib] * features[index[ib]])
                        pick_a = (discrepancy + error_a).square().sum() <= (
                            discrepancy + error_b).square().sum()
                        selected = ia if pick_a else ib
                        discrepancy += error_a if pick_a else error_b
                        chosen.append(index[selected])
                        updated.append(total)
                    tail = order[2 * paired:]
                    index = torch.cat((torch.stack(chosen), index[tail]))
                    weights = torch.cat((torch.stack(updated), weights[tail]))
                index, weights = index.to(k.device), weights.float().to(k.device)
            self.state = dict(k=k[index].clone(), v=v[index].clone(), log_weights=weights.log())
        elif method == 'kernel':
            rank = cfg['features']
            assert rank > 0
            generator = torch.Generator(device=k.device).manual_seed(cfg.get('seed', 17))
            omega = torch.randn(self.dk, rank, device=k.device, generator=generator)
            self.state = dict(omega=omega, key_shift=k.new_tensor(float('-inf')).float(),
                S=torch.zeros(rank, self.dv, device=k.device), z=torch.zeros(rank, device=k.device))
            self.count = 0
            self.append(k, v)
        elif method in ('centroid', 'moments'):
            kf, vf = k.float(), v.float()
            groups = min(cfg['groups'], len(k))
            assert groups > 0
            centers = kf[torch.linspace(0, len(k) - 1, groups, device=k.device).round().long()].clone()
            for _ in range(cfg.get('iterations', 8)):
                distance = kf.square().sum(-1, keepdim=True) - 2 * kf @ centers.T + centers.square().sum(-1)
                assignment = distance.argmin(-1)
                count = torch.bincount(assignment, minlength=groups)
                totals = torch.zeros_like(centers).index_add_(0, assignment, kf)
                centers = torch.where(count[:, None] > 0, totals / count.clamp_min(1)[:, None], centers)
            # Drop empty clusters rather than fabricating observations.
            alive = (count > 0).nonzero().flatten()
            means_v = torch.zeros(groups, self.dv, device=k.device).index_add_(0, assignment, vf)
            self.state = dict(mean_k=centers[alive].clone(),
                mean_v=(means_v[alive] / count[alive, None]).clone(),
                log_count=count[alive].float().log())
            if method == 'moments':
                rank = min(cfg['rank'], self.dk, len(k))
                assert rank > 0
                _, _, vh = torch.linalg.svd(kf - kf.mean(0), full_matrices=False)
                basis = vh[:rank].T.contiguous()
                cov, cross = [], []
                for group in alive:
                    member = assignment == group
                    residual_k = (kf[member] - centers[group]) @ basis
                    residual_v = vf[member] - means_v[group] / count[group]
                    cov.append(residual_k.T @ residual_k / count[group])
                    cross.append(residual_v.T @ residual_k / count[group])
                self.state.update(basis=basis, cov=torch.stack(cov), cross=torch.stack(cross))
        elif method in ('lowrank', 'query_lowrank'):
            kf, vf = k.float(), v.float()
            rank_k = min(cfg['rank'], self.dk, len(k))
            rank_v = min(cfg['rank'], self.dv, len(v))
            assert rank_k > 0 and rank_v > 0
            if method == 'query_lowrank':
                assert len(calibration_q) > 0 and cfg.get('ridge', 1e-4) > 0
                cq = calibration_q.float().T @ calibration_q.float() / len(calibration_q)
                ridge = cfg.get('ridge', 1e-4) * cq.trace().clamp_min(1e-12) / self.dk
                factor = torch.linalg.cholesky(cq + ridge * torch.eye(self.dk, device=k.device))
            else:
                factor = torch.eye(self.dk, device=k.device)
            _, _, vh = torch.linalg.svd(kf @ factor, full_matrices=False)
            basis_k = vh[:rank_k].T.contiguous()
            query_map = torch.linalg.solve_triangular(factor.T, basis_k, upper=True)
            _, _, vh = torch.linalg.svd(vf, full_matrices=False)
            basis_v = vh[:rank_v].T.contiguous()
            self.state = dict(k=(kf @ factor @ basis_k).contiguous(),
                v=(vf @ basis_v).contiguous(), query_map=query_map.contiguous(), value_map=basis_v)
        elif method in ('quant', 'waterfill'):
            average = cfg['bits']
            assert isinstance(average, int) and 1 <= average <= 8
            kf, vf = k.float(), v.float()
            if method == 'waterfill':
                alpha_k = kf.var(0, unbiased=False) * calibration_q.float().square().mean(0) / self.dk
                # Value metric is identity; no downstream projection is assumed.
                alpha_v = vf.var(0, unbiased=False)
                widths = allocate_bits(torch.cat((alpha_k, alpha_v)), average * (self.dk + self.dv))
            else:
                widths = torch.full((self.dk + self.dv,), average, device=k.device, dtype=torch.int32)
            self.packed_k = PackedMatrix(kf, widths[:self.dk])
            self.packed_v = PackedMatrix(vf, widths[self.dk:])
            self.state = dict(k=self.packed_k.state, v=self.packed_v.state)
            if cfg.get('bias_correction', False):
                error = self.packed_k.decode() - kf
                self.state['key_error_mean'] = error.mean(0)
                self.state['key_error_cov'] = ((error - error.mean(0)).T @ (error - error.mean(0))) / len(error)
        return self

    @torch.no_grad()
    def append(self, k, v):
        """Exact additive update of the chosen random-feature approximation."""
        assert self.method == 'kernel' and k.ndim == v.ndim == 2
        assert k.shape == (len(v), self.dk) and v.shape[1] == self.dv and len(k) > 0
        s = self.state
        x = k.float() / self.dk ** .25
        log_phi = x @ s['omega'] - .5 * x.square().sum(-1, keepdim=True)
        shift = torch.maximum(s['key_shift'], log_phi.max())
        old_scale = (s['key_shift'] - shift).exp()
        phi = (log_phi - shift).exp() / math.sqrt(s['omega'].shape[1])
        s['S'] = (s['S'] * old_scale + phi.T @ v.float()).contiguous()
        s['z'] = (s['z'] * old_scale + phi.sum(0)).contiguous()
        s['key_shift'] = shift.clone()
        self.count += len(k)

    @torch.no_grad()
    def history_response(self, q):
        assert q.ndim == 2 and q.shape[1] == self.dk
        q, s, method = q.float(), self.state, self.method
        if method in ('exact', 'uniform', 'probe_balance'):
            return dense_response(q, s['k'], s['v'], s.get('log_weights'))
        if method == 'kernel':
            x = q / self.dk ** .25
            log_phi = x @ s['omega'] - .5 * x.square().sum(-1, keepdim=True)
            shift = log_phi.amax(-1, keepdim=True)
            phi = (log_phi - shift).exp() / math.sqrt(s['omega'].shape[1])
            denominator = phi @ s['z']
            return phi @ s['S'] / denominator[:, None], denominator.log() + shift[:, 0] + s['key_shift']
        if method in ('centroid', 'moments'):
            u = q * self.scale
            logits = u @ s['mean_k'].T + s['log_count']
            if method == 'centroid':
                return logits.softmax(-1) @ s['mean_v'], logits.logsumexp(-1)
            projected = u @ s['basis']
            logits += .5 * torch.einsum('qr,grs,qs->qg', projected, s['cov'], projected)
            weights = logits.softmax(-1)
            out = weights @ s['mean_v'] + torch.einsum('qg,gvr,qr->qv', weights, s['cross'], projected)
            return out, logits.logsumexp(-1)
        if method in ('lowrank', 'query_lowrank'):
            # Keep the ORIGINAL head-dimension scale, not sqrt(reduced rank).
            logits = (q @ s['query_map']) @ s['k'].T * self.scale
            return (logits.softmax(-1) @ s['v']) @ s['value_map'].T, logits.logsumexp(-1)
        k, v = self.packed_k.decode(), self.packed_v.decode()
        logits = q @ k.T * self.scale
        if self.config.get('bias_correction', False):
            # Empirical mean correction plus second cumulant. Not an unbiased
            # softmax guarantee; covariance is measured during construction.
            correction = q @ s['key_error_mean'] * self.scale
            correction += .5 * (q @ s['key_error_cov'] * q).sum(-1) * self.scale ** 2
            logits -= correction[:, None]
        return logits.softmax(-1) @ v, logits.logsumexp(-1)

    @torch.no_grad()
    def read(self, q, current_k=None, current_v=None):
        out, mass = self.history_response(q)
        if current_k is not None:
            current, current_mass = dense_response(q, current_k, current_v)
            out = merge_responses(out, mass, current, current_mass)
        return out

    def persistent_bytes(self):
        """Resident tensor storage, including maps, packed metadata and correction.

        Every state tensor owns storage; Python object overhead is excluded.
        Decoded quantization tensors are transient and counted by runtime peaks.
        """
        return tensor_bytes(self.state)
