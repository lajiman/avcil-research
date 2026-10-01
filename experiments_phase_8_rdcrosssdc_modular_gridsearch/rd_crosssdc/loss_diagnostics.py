"""Opt-in loss accounting and read-only gradient probes on the existing graph.

Scalar accounting is independent of torch. It averages batches, exactly like
train_loss. Probes do not call backward, step, forward, or any RNG operation.
"""
import math


def scalar(value):
    return float(value.detach().item()) if hasattr(value, 'detach') else float(value)


class Moments:
    def __init__(self):
        self.n = 0
        self.mean = self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value):
        if not math.isfinite(value):
            raise ValueError('Non-finite loss diagnostic')
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)


class LossEpochRecorder:
    """Keep raw and weighted components, including computed zero-weight terms."""
    def __init__(self):
        self.total = Moments()
        self.residual = Moments()
        self.max_abs_residual = 0.0
        self.components = {}

    def add(self, total, components):
        # components: name -> (scalar tensor or None if disabled, coefficient)
        if self.total.n and set(components) != set(self.components):
            raise ValueError('Loss components changed within an epoch')
        actual, reconstructed = scalar(total), 0.0
        for name, (value, coefficient) in components.items():
            coefficient = float(coefficient)
            if not math.isfinite(coefficient):
                raise ValueError('Non-finite loss coefficient')
            entry = self.components.setdefault(name, (coefficient, Moments()))
            if entry[0] != coefficient:
                raise ValueError('Loss coefficient changed within an epoch')
            if value is not None:
                raw = scalar(value)
                entry[1].add(raw)
                reconstructed += coefficient * raw
        residual = actual - reconstructed
        self.total.add(actual)
        self.residual.add(residual)
        self.max_abs_residual = max(self.max_abs_residual, abs(residual))

    def rows(self, *, step, epoch, val_acc):
        if not self.total.n:
            return []
        result = []
        for name, (coefficient, raw) in self.components.items():
            if raw.n not in (0, self.total.n):
                raise ValueError('Component enabled for only part of epoch: '+name)
            weighted_mean = coefficient * raw.mean if raw.n else 0.0
            sd = math.sqrt(max(raw.m2 / (raw.n-1), 0.0)) if raw.n > 1 else None
            result.append(dict(step=step, epoch=epoch, component=name,
                enabled=int(raw.n > 0), coefficient=coefficient, num_batches=self.total.n,
                raw_mean=raw.mean if raw.n else None, raw_batch_sd=sd,
                raw_min=raw.minimum if raw.n else None, raw_max=raw.maximum if raw.n else None,
                weighted_mean=weighted_mean,
                weighted_batch_sd=abs(coefficient)*sd if sd is not None else None,
                total_mean=self.total.mean,
                signed_weighted_over_total=weighted_mean/self.total.mean if abs(self.total.mean)>1e-12 else None,
                reconstruction_residual_mean=self.residual.mean,
                reconstruction_max_abs_residual=self.max_abs_residual, val_acc=scalar(val_acc)))
        return result


def gradient_probe_rows(total, components, parameters, *, step, epoch, batch):
    """Full-model Euclidean gradients BEFORE the ordinary training backward.

The common reference is the sum of all weighted non-CMR loss terms. A zero
gradient has undefined cosine/ratio when a denominator is zero. These are
instantaneous gradients, not Adam updates or a lambda counterfactual rollout.
Requires extra autograd traversals; only called when explicitly enabled.
"""
    import torch
    params = tuple(p for p in parameters if p.requires_grad)
    if not params:
        raise ValueError('No trainable parameters for gradient probe')

    def grads(loss):
        if loss is None or not getattr(loss, 'requires_grad', False):
            return (None,) * len(params)
        return torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)

    def dot(left, right):
        value = torch.zeros((), device=params[0].device, dtype=torch.float64)
        for a,b in zip(left,right):
            if a is not None and b is not None:
                value += (a.detach().float()*b.detach().float()).sum(dtype=torch.float64)
        return float(value.item())

    base = None
    for name,(value,coefficient) in components.items():
        if name != 'cmr' and value is not None and coefficient != 0:
            term = value * coefficient
            base = term if base is None else base + term
    base_grads = grads(base)
    base_norm = math.sqrt(max(dot(base_grads,base_grads),0.0))
    result = []
    for name,(value,coefficient) in list(components.items()) + [('total',(total,1.0))]:
        g = grads(value)
        raw_norm = math.sqrt(max(dot(g,g),0.0))
        weighted_norm = abs(coefficient)*raw_norm
        cosine = coefficient*dot(g,base_grads)/(weighted_norm*base_norm) if weighted_norm>0 and base_norm>0 else None
        result.append(dict(step=step,epoch=epoch,batch=batch,component=name,
            enabled=int(value is not None),coefficient=coefficient,
            raw_grad_norm=raw_norm,weighted_grad_norm=weighted_norm,
            non_cmr_grad_norm=base_norm,
            weighted_grad_over_non_cmr=weighted_norm/base_norm if base_norm>0 else None,
            cosine_with_non_cmr=max(-1.0,min(1.0,cosine)) if cosine is not None else None,
            parameter_scope='all_trainable_parameters',timing='before_optimizer_step'))
        del g
    return result
