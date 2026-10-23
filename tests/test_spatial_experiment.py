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
        assert set(item['evaluations']) == {'train', 'id', 'larger', 'rotated'}


def test_metrics_stop_padding_exactness_and_token_weighted_ce():
    from tasks.mazes.spatial_experiment import encode_maze, prediction_metrics
    # A valid detour reaches the goal but is not the canonical shortest route.
    maze = np.ones((3, 4), dtype=bool)
    encoded = [encode_maze(maze, (1, 1), goal, 6) for goal in [(1, 2), (1, 3)]]
    x, y = (torch.from_numpy(np.stack(v)) for v in zip(*encoded))
    logits = torch.zeros(2, 6, 5, 2)
    logits[0, 0, 1, :] = 2
    metrics = prediction_metrics(logits, x, y)
    expected = torch.nn.functional.cross_entropy(logits[..., -1].reshape(-1, 5), y.flatten())
    assert metrics['cross_entropy'] == pytest.approx(expected.item())
    assert metrics['valid_tokens'] == 5  # includes both STOPs, excludes padding
    logits[:, 3:] = 100  # changing ignored slots cannot change CE
    assert prediction_metrics(logits, x, y)['cross_entropy'] == metrics['cross_entropy']
    actions = [[0, 1, 2, 4, 0, 0], [1, 1, 4, 0, 0, 0]]
    logits.fill_(-20)
    for b, route in enumerate(actions):
        for slot, action in enumerate(route):
            logits[b, slot, action, :] = 20
    metrics = prediction_metrics(logits, x, y)
    assert metrics['success_rate'] == 1
    assert metrics['exact_sequence_rate'] == .5
    assert metrics['collision_rate'] == 0
    assert route_metrics([4], x[0].numpy(), y[0].numpy()) == (False, False, None)
    assert route_metrics([0, 0], x[0].numpy(), y[0].numpy()) == (False, True, None)
    assert route_metrics([1], x[0].numpy(), y[0].numpy()) == (False, False, None)


def diagnostic_model(kind='dense'):
    from models.ctm import ContinuousThoughtMachine
    return ContinuousThoughtMachine(
        iterations=4, d_model=16, d_input=8, heads=2, n_synch_out=12,
        n_synch_action=8, n_random_pairing_self=2, neuron_select_type='random-pairing',
        synapse_depth=1, memory_length=3, deep_nlms=True, memory_hidden_dims=3,
        do_layernorm_nlm=False, backbone_type='none', positional_embedding_type='none',
        out_dims=32*5, prediction_reshaper=[32, 5], synapse_type=kind,
        spatial_config=None if kind == 'dense' else dict(grid_shape=(4, 4)), dropout=.2)


@pytest.mark.parametrize('kind', ['dense', 'spatial', 'wave'])
def test_evaluation_is_ordered_repeatable_and_preserves_training_state(kind):
    from tasks.mazes.spatial_experiment import evaluate
    model = diagnostic_model(kind).train()
    x, y, _ = make_dataset(3, 3, 11, 32)
    model(x)  # materialize
    with torch.no_grad():
        model.decay_params_out.fill_(-1)  # forward's in-place clamp must be undone
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    first = evaluate(model, x, y, 2, 'cpu')
    assert first == evaluate(model, x, y, 2, 'cpu')
    assert model.training
    assert torch.equal(rng, torch.random.get_rng_state())
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert all(p.grad is None for p in model.parameters())
    batched = evaluate(model, x, y, 3, 'cpu')
    assert first['cross_entropy'] == pytest.approx(batched['cross_entropy'], abs=1e-6)
    assert first['per_example_success'] == batched['per_example_success']
    with pytest.raises(RuntimeError):
        evaluate(model, x[:, :4], y, 2, 'cpu')
    assert model.training
    assert torch.equal(model.decay_params_out, before['decay_params_out'])


@pytest.mark.parametrize('kind', ['dense', 'spatial', 'wave'])
def test_disconnected_control_removes_specified_paths_and_restores_on_error(kind):
    from tasks.mazes.spatial_controls import mechanism_control
    model = diagnostic_model(kind).eval()
    x, _, _ = make_dataset(2, 3, 11, 32)
    model(x)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    seen_synapse, seen_query = [], []
    with pytest.raises(RuntimeError, match='intentional'):
        with mechanism_control(model, 'disconnected'):
            def capture(module, inputs):
                seen_synapse.append(inputs)
            # Registered after the intervention, so sees the modified inputs.
            h1 = model.synapses.register_forward_pre_hook(capture)
            h2 = model.q_proj.register_forward_hook(lambda m, i, o: seen_query.append(o))
            try:
                model(x)
                assert torch.equal(model.out_neuron_indices_left, model.out_neuron_indices_right)
            finally:
                h1.remove()
                h2.remove()
            raise RuntimeError('intentional')
    assert len(seen_synapse) == model.iterations
    assert all(torch.count_nonzero(v[0][:, model.d_input:]) == 0 for v in seen_synapse)
    if kind != 'dense':
        assert all(v[1] is None for v in seen_synapse)
    assert all(torch.count_nonzero(q) == 0 for q in seen_query)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert not model.synapses._forward_pre_hooks and not model.q_proj._forward_hooks
    if kind == 'wave':
        old_config = model.synapses.config
        with mechanism_control(model, 'zero-speed'):
            assert model.synapses.config.wave_speed == 0
        assert model.synapses.config == old_config


def test_preset_overrides_and_gate_requires_two_final_measurements():
    from tasks.mazes.spatial_experiment import parse_args, overfit_gate
    args = parse_args(['--overfit', '--steps', '7', '--train_examples', '2'])
    assert (args.steps, args.train_examples, args.full_batch, args.seeds) == (7, 2, True, [0])
    good = dict(cross_entropy=.05, success_rate=1., exact_sequence_rate=1., collision_rate=0.)
    points = [dict(update=update, evaluations={'train': good.copy()}) for update in [0, 25, 50]]
    assert not overfit_gate(points[:2])['passed']
    assert overfit_gate(points)['passed']
    points[-1]['evaluations']['train']['exact_sequence_rate'] = .75
    assert not overfit_gate(points)['passed']


def test_evaluation_cadence_does_not_change_training_and_control_checkpoint(tmp_path):
    from tasks.mazes.spatial_controls import mechanism_control
    from tasks.mazes.spatial_experiment import parse_args, evaluate
    from models.ctm import ContinuousThoughtMachine
    outputs = []
    for cadence in [1, 3]:
        args = parse_args(['--overfit', '--steps', '3', '--test_examples', '2',
                           '--spatial_grid', '4', '4', '--iterations', '4',
                           '--eval_every', str(cadence), '--controls', 'none', 'disconnected',
                           '--output', str(tmp_path/f'cadence{cadence}')])
        outputs.append(run(args))
    for a, b in zip(*(r['runs'] for r in outputs)):
        assert a['loss'] == b['loss']
        assert a['evaluations'] == b['evaluations']
        tag = f"{a['variant']}-seed0" + ('-disconnected' if a['control'] != 'none' else '')
        ca = torch.load(tmp_path/f'cadence1/{tag}.pt', weights_only=True)
        cb = torch.load(tmp_path/f'cadence3/{tag}.pt', weights_only=True)
        for k, v in ca['model_state_dict'].items():
            assert torch.equal(v, cb['model_state_dict'][k]), k
        model = ContinuousThoughtMachine(**ca['model_kwargs'])
        model.load_state_dict(ca['model_state_dict'], strict=True)
        x, y, _ = make_dataset(4, 3, 1729, 32)
        with mechanism_control(model, ca['control']):
            metrics = evaluate(model, x, y, 4, 'cpu')
        assert metrics['cross_entropy'] == a['evaluations']['train']['cross_entropy']
