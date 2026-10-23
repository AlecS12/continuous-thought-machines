"""CPU tests requiring only PyTorch, NumPy and pytest; no datasets."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from models.ctm import ContinuousThoughtMachine
from models.spatial_synapses import SpatialSynapseConfig, SpatialSynapses
from tasks.mazes.spatial_config import spatial_model_kwargs


def small_ctm(kind='dense', depth=1):
    return ContinuousThoughtMachine(
        iterations=6, d_model=24, d_input=8, heads=2, n_synch_out=8,
        n_synch_action=8, synapse_depth=depth, memory_length=4,
        deep_nlms=True, memory_hidden_dims=3, do_layernorm_nlm=False,
        backbone_type='none', positional_embedding_type='none', out_dims=5,
        synapse_type=kind, spatial_config=None if kind == 'dense' else
        SpatialSynapseConfig((4, 6)),
    )


@pytest.mark.parametrize('boundary', ['zeros', 'circular'])
@pytest.mark.parametrize('radius', [0, 1, 2])
def test_local_jacobian_has_exact_support(boundary, radius):
    h, w = 5, 7
    module = SpatialSynapses(h*w, 3, SpatialSynapseConfig((h, w), radius, boundary)).double()
    x = torch.randn(1, 3+h*w, dtype=torch.double, requires_grad=True)
    # Include a corner: catches flattened-row wraparound and unintended padding.
    for row, col in [(0, 0), (2, 3)]:
        grad, = torch.autograd.grad(module(x)[0][0, row*w+col], x, retain_graph=True)
        dy, dx = np.abs(np.arange(h)-row), np.abs(np.arange(w)-col)
        if boundary == 'circular':
            dy, dx = np.minimum(dy, h-dy), np.minimum(dx, w-dx)
        allowed = torch.from_numpy((dy[:, None] <= radius) & (dx[None, :] <= radius))
        recurrent_grad = grad[0, 3:].reshape(h, w)
        assert torch.count_nonzero(recurrent_grad[~allowed]) == 0
        assert torch.all(recurrent_grad[allowed] != 0)
        assert grad[0, :3].abs().sum() > 0  # external drive remains trainable


@pytest.mark.parametrize('boundary', ['zeros', 'circular'])
def test_wave_front_finite_speed_and_boundary(boundary):
    module = SpatialSynapses(49, 2, SpatialSynapseConfig((7, 7), boundary=boundary), wave=True)
    impulse = torch.zeros(1, 49)
    impulse[0, 0] = 1
    out, state = module.propagate(impulse)
    assert torch.count_nonzero(out) == 1
    out, state = module.propagate(torch.zeros_like(impulse), state)
    assert out[0, 1] > 0 and out[0, 7] > 0
    assert out[0, 2] == 0 and out[0, 8] == 0
    assert (out[0, 6] > 0).item() == (boundary == 'circular')
    assert (out[0, 42] > 0).item() == (boundary == 'circular')
    out, _ = module.propagate(torch.zeros_like(impulse), state)
    assert out[0, 2] > 0 and out[0, 8] > 0


def test_wave_substeps_match_individual_steps_and_preserve_gradients():
    cfg = SpatialSynapseConfig((4, 6))
    single = SpatialSynapses(24, 3, cfg, wave=True).double()
    multi = SpatialSynapses(24, 3, replace(cfg, wave_substeps=3), wave=True).double()
    drive = torch.randn(2, 24, dtype=torch.double, requires_grad=True)
    state = None
    for _ in range(3):
        expected, state = single.propagate(drive, state)
    actual, actual_state = multi.propagate(drive)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    loss = sum(t.square().sum() for t in actual_state)
    loss.backward()
    assert torch.isfinite(drive.grad).all() and drive.grad.abs().sum() > 0
    assert torch.autograd.gradcheck(lambda d: multi.propagate(d)[0], (drive.detach().requires_grad_(),))


def test_unforced_damped_solver_decays_and_zero_speed_does_not_spread():
    cfg = SpatialSynapseConfig((5, 5), boundary='circular')
    module = SpatialSynapses(25, 2, cfg, wave=True)
    pulse = torch.zeros(1, 25)
    pulse[0, 12] = 1
    _, state = module.propagate(pulse)
    initial = sum(t.square().sum() for t in state)
    for _ in range(300):
        _, state = module.propagate(torch.zeros_like(pulse), state)
    assert sum(t.square().sum() for t in state) < initial * 1e-6
    stationary = SpatialSynapses(25, 2, replace(cfg, wave_speed=0), wave=True)
    _, state = stationary.propagate(pulse)
    for _ in range(10):
        out, state = stationary.propagate(torch.zeros_like(pulse), state)
    assert torch.count_nonzero(out) == 1


@pytest.mark.parametrize('kind', ['spatial', 'wave'])
def test_ctm_shapes_gradients_tracking_reset_and_checkpoint(kind):
    model = small_ctm(kind).eval()
    x = torch.randn(2, 3, 3, 5, requires_grad=True)
    pred, certainty, synch = model(x)
    assert pred.shape == (2, 5, 6) and certainty.shape == (2, 2, 6)
    assert synch.shape == (2, 8)
    pred.square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    for param in [model.synapses.local_weight, model.synapses.input_projection.weight,
                  model.start_trace, model.start_activated_state]:
        assert param.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0 and torch.isfinite(x.grad).all()
    with torch.no_grad():
        tracked = model(x, track=True)
        assert len(tracked) == 6
        assert tracked[3].shape == tracked[4].shape == (6, 2, 24)
        torch.testing.assert_close(pred, tracked[0], rtol=0, atol=0)
        model(torch.randn(3, 3, 2, 4))  # interleaved, different batch
        torch.testing.assert_close(pred, model(x)[0], rtol=0, atol=0)
        restored = small_ctm(kind).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(pred, restored(x)[0], rtol=0, atol=0)


@pytest.mark.parametrize('depth', [1, 3])
def test_dense_default_state_dict_and_outputs_unchanged(depth, base_params):
    # An omitted option must not add keys or change initialization RNG usage.
    kwargs = dict(base_params, synapse_depth=depth)
    torch.manual_seed(37)
    np.random.seed(37)
    implicit = ContinuousThoughtMachine(**kwargs).eval()
    x = torch.randn(2, 3, 4, 4)
    implicit(x)  # initialize lazy layers
    torch.manual_seed(37)
    np.random.seed(37)
    explicit = ContinuousThoughtMachine(**kwargs, synapse_type='dense').eval()
    other_x = torch.randn(2, 3, 4, 4)
    explicit(other_x)
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for key, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    explicit.load_state_dict(copy.deepcopy(implicit.state_dict()), strict=True)
    for left, right in zip(implicit(x), explicit(x)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('change', [
    dict(grid_shape=(3, 0)), dict(grid_shape=(3.5, 4)), dict(radius=-1),
    dict(radius=4), dict(boundary='reflect'), dict(wave_speed=float('nan')),
    dict(wave_damping=-1), dict(wave_frequency=float('inf')), dict(wave_dt=0),
    dict(wave_dt=2), dict(wave_substeps=0), dict(wave_substeps=1.5),
])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        SpatialSynapseConfig(**(dict(grid_shape=(4, 6)) | change))


def test_invalid_shapes_and_explicit_state():
    cfg = SpatialSynapseConfig((4, 6))
    with pytest.raises(ValueError):
        SpatialSynapses(25, 3, cfg)
    module = SpatialSynapses(24, 3, cfg, wave=True)
    with pytest.raises(ValueError):
        module(torch.zeros(2, 24))
    _, state = module(torch.zeros(2, 27))
    with pytest.raises(ValueError):
        module(torch.zeros(1, 27), state)
    with pytest.raises(ValueError):
        SpatialSynapses(24, 3, cfg)(torch.zeros(2, 27), state)


def test_cli_and_legacy_checkpoint_config():
    import argparse
    from tasks.mazes.spatial_config import add_spatial_arguments
    assert spatial_model_kwargs(argparse.Namespace()) == {'synapse_type': 'dense'}
    parser = argparse.ArgumentParser()
    add_spatial_arguments(parser)
    args = parser.parse_args(['--synapse_type', 'wave', '--spatial_grid', '4', '6'])
    module = SpatialSynapses(24, 3, spatial_model_kwargs(args)['spatial_config'])
    assert module.config.grid_shape == (4, 6)
    with pytest.raises(ValueError):
        spatial_model_kwargs(parser.parse_args(['--synapse_type', 'wave']))


def test_maze_checkpoint_formats():
    from tasks.mazes.spatial_config import maze_checkpoint_weights
    weights = {'synapses.local_weight': torch.randn(2, 24, 9)}
    assert maze_checkpoint_weights({'model_state_dict': weights}) is weights
    assert maze_checkpoint_weights({'state_dict': weights}) is weights
    with pytest.raises(KeyError):
        maze_checkpoint_weights({})
