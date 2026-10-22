import argparse
import json

import numpy as np
import pytest
import torch

from tasks.mazes.spatial_diagnostics import synchronization_metrics, propagation_metrics
from tasks.mazes.spatial_experiment import make_dataset, rotate_dataset, route_metrics, run


def test_generated_routes_and_disjoint_reproducible_splits():
    x, y, hashes = make_dataset(12, 4, 3, 64)
    repeated = make_dataset(12, 4, 3, 64)
    torch.testing.assert_close(x, repeated[0], rtol=0, atol=0)
    assert hashes == repeated[2]
    test_x, test_y, test_hashes = make_dataset(8, 4, 4, 64, hashes)
    assert not set(hashes) & set(test_hashes)
    for observations, targets in [(x, y), (test_x, test_y), rotate_dataset(x, 64)]:
        for observation, target in zip(observations.numpy(), targets.numpy()):
            assert route_metrics(target[target != -100], observation, target) == (True, False, 1)
            assert route_metrics([4], observation, target)[0] is False
            assert route_metrics([], observation, target)[0] is False
    with pytest.raises(ValueError, match='never truncated'):
        make_dataset(4, 4, 3, 2)


def test_constant_activity_is_not_synchronization():
    metrics = synchronization_metrics(np.ones((16, 2, 12)), (3, 4))
    assert metrics['active_fraction'] == 0
    assert metrics['phase_order'] is None
    assert metrics['neighbor_phase_locking'] is None
    assert metrics['dominant_cycles_per_tick'] is None
    json.dumps(metrics, allow_nan=False)


def test_phase_diagnostic_detects_known_oscillation_and_shuffle():
    oscillation = np.sin(2*np.pi*np.arange(64)/8)
    trace = np.broadcast_to(oscillation[:, None, None], (64, 2, 12))
    metrics = synchronization_metrics(trace, (3, 4))
    assert metrics['phase_order'] == pytest.approx(1)
    assert metrics['neighbor_phase_locking'] == pytest.approx(1)
    assert metrics['dominant_cycles_per_tick'] == 1/8
    assert metrics['spectral_peak_fraction'] == pytest.approx(1)
    assert metrics['shuffled_neighbor_phase_locking'] < 0.3


def test_propagation_arrivals_and_silent_response():
    trace = np.zeros((4, 12))
    trace[0, 0] = trace[1, 1] = trace[2, 2] = 1
    metrics = propagation_metrics(trace, (3, 4), source=0)
    assert metrics['first_arrival_tick'][:4] == [1, 2, 3, None]
    assert metrics['mean_energy_distance'] == [0, 1, 2, None]
    json.dumps(metrics, allow_nan=False)


def test_three_variant_training_smoke_and_checkpoint(tmp_path):
    args = argparse.Namespace(steps=1, train_examples=4, test_examples=2,
                              batch_size=2, cells=4, horizon=64, iterations=4,
                              seeds=[0], output=str(tmp_path/'pilot'), threads=1,
                              data_seed=42, spatial_grid=[4, 4], synapse_type='dense',
                              spatial_boundary='zeros', device='cpu', lr=1e-3)
    results = run(args)
    assert [item['variant'] for item in results['runs']] == ['dense', 'spatial', 'wave']
    assert len(json.loads((tmp_path/'pilot/results.json').read_text())['runs']) == 3
    from models.ctm import ContinuousThoughtMachine
    for item in results['runs']:
        path = tmp_path/f"pilot/{item['variant']}-seed0.pt"
        checkpoint = torch.load(path, weights_only=True)
        restored = ContinuousThoughtMachine(**checkpoint['model_kwargs']).eval()
        restored.load_state_dict(checkpoint['model_state_dict'], strict=True)
        x, _, _ = make_dataset(2, 4, 43, 64)
        assert torch.isfinite(restored(x)[0]).all()
        assert np.isfinite(item['loss']).all()
        assert set(item['evaluations']) == {'id', 'larger', 'rotated'}
