"""Descriptive diagnostics, not evidence by themselves of useful computation."""

import numpy as np
import torch


def grid_distances(grid_shape, boundary='zeros'):
    row, col = np.indices(grid_shape).reshape(2, -1)
    dy, dx = np.abs(row[:, None] - row), np.abs(col[:, None] - col)
    if boundary == 'circular':
        dy = np.minimum(dy, grid_shape[0] - dy)
        dx = np.minimum(dx, grid_shape[1] - dx)
    elif boundary != 'zeros':
        raise ValueError('unknown boundary')
    return dy + dx


def _phase(x):
    # FFT Hilbert transform along time; endpoints are periodic. Interpret only
    # sufficiently long oscillatory traces and check spectral concentration.
    length = len(x)
    multiplier = np.zeros(length)
    multiplier[0] = 1
    multiplier[1:(length+1)//2] = 2
    if length % 2 == 0:
        multiplier[length//2] = 1
    analytic = np.fft.ifft(np.fft.fft(x, axis=0) * multiplier[:, None, None], axis=0)
    return analytic / np.maximum(np.abs(analytic), 1e-12)


def synchronization_metrics(trace, grid_shape, boundary='zeros', seed=0):
    """trace is [ticks, batch, neurons]; Pearson and phase metrics exclude DC.

    Silent sites are excluded, never treated as synchronized. The shuffle null
    independently permutes each site's time axis, preserving its marginals but
    destroying temporal structure (it is not a spectrum-preserving null).
    """
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim != 3 or trace.shape[-1] != np.prod(grid_shape) or trace.shape[0] < 4:
        raise ValueError('expected at least four ticks of [ticks, batch, H*W] activity')
    if not np.isfinite(trace).all():
        raise ValueError('nonfinite activity')
    x = trace - trace.mean(axis=0, keepdims=True)
    std = x.std(axis=0)
    active = std > 1e-8
    norm = x / np.maximum(std, 1e-8)
    corr = np.einsum('tbi,tbj->bij', norm, norm) / len(x)
    distances = grid_distances(grid_shape, boundary)
    upper = np.triu(np.ones_like(distances, dtype=bool), 1)
    near = upper & (distances == 1)
    far = upper & (distances >= max(2, max(grid_shape)//2))
    valid = active[:, :, None] & active[:, None, :]

    def average(values, mask):
        selected = values[valid & mask[None]]
        return float(selected.mean()) if selected.size else None

    phase = _phase(x)
    plv = np.abs(np.einsum('tbi,tbj->bij', phase, phase.conj()) / len(x))
    rng = np.random.default_rng(seed)
    shuffled = np.take_along_axis(x, rng.random(x.shape).argsort(axis=0), axis=0)
    null_phase = _phase(shuffled)
    null_plv = np.abs(np.einsum('tbi,tbj->bij', null_phase, null_phase.conj()) / len(x))

    def order(phases):
        values = np.abs((phases * active[None]).sum(-1)) / np.maximum(active.sum(-1), 1)
        return float(values[:, active.any(-1)].mean()) if active.any() else None

    power = (np.abs(np.fft.rfft(x, axis=0))**2).sum(axis=(1, 2))
    power[0] = 0
    spectral = power.sum() > 1e-12
    return dict(
        active_fraction=float(active.mean()), temporal_std_mean=float(std.mean()),
        neighbor_correlation=average(corr, near), distant_correlation=average(corr, far),
        neighbor_phase_locking=average(plv, near), distant_phase_locking=average(plv, far),
        shuffled_neighbor_phase_locking=average(null_plv, near),
        phase_order=order(phase), shuffled_phase_order=order(null_phase),
        dominant_cycles_per_tick=float(power.argmax()/len(x)) if spectral else None,
        spectral_peak_fraction=float(power.max()/power.sum()) if spectral else None,
        ticks=len(x),
    )


def propagation_metrics(response, grid_shape, source, boundary='zeros', threshold=1e-6):
    """Finite-difference impulse response [ticks, neurons], normalized by input pulse."""
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 2 or response.shape[1] != np.prod(grid_shape):
        raise ValueError('expected [ticks, H*W] impulse response')
    if not np.isfinite(response).all():
        raise ValueError('nonfinite impulse response')
    distance = grid_distances(grid_shape, boundary)[source]
    energy = response**2
    total = energy.sum(axis=-1)
    arrivals = np.abs(response) > threshold
    first = [int(arrivals[:, i].argmax()) + 1 if arrivals[:, i].any() else None
             for i in range(response.shape[1])]
    mean_distance = [float((e*distance).sum()/s) if s > 0 else None
                     for e, s in zip(energy, total)]
    return dict(source_index=source, threshold=threshold, first_arrival_tick=first,
                distance_from_source=distance.tolist(), mean_energy_distance=mean_distance,
                response_l2=np.sqrt(total).tolist(),
                far_energy_fraction=[float(e[distance >= 3].sum()/s) if s > 0 else None
                                     for e, s in zip(energy, total)])


@torch.no_grad()
def probe_synapses(model, grid_shape, ticks=16, amplitude=1e-3, source=None):
    """Isolate synapses: clamp attention to zero; feed output back as activity.

    This intentionally bypasses NLMs and the attention feedback loop. It is
    an operator diagnostic, not the speed of information in the complete CTM.
    Both trajectories share weights/biases and independent explicit wave state.
    The caller must put model in eval mode to disable dropout.
    """
    if model.training:
        raise ValueError('probe_synapses requires eval mode')
    source = model.d_model//2 + grid_shape[1]//2 if source is None else source
    parameter = next(model.parameters())
    z = parameter.new_zeros(2, model.d_model)
    z[1, source] = amplitude
    observation = parameter.new_zeros(2, model.d_input)
    wave_state = None
    responses = []
    for _ in range(ticks):
        x = torch.cat((observation, z), -1)
        if model.synapse_type == 'dense':
            z = model.synapses(x)
        else:
            z, wave_state = model.synapses(x, wave_state)
        responses.append(((z[1]-z[0])/amplitude).cpu().numpy())
    boundary = 'zeros' if model.synapse_type == 'dense' else model.synapses.config.boundary
    return propagation_metrics(np.stack(responses), grid_shape, source, boundary)
