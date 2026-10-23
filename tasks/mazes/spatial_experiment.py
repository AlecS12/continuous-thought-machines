"""Small, download-free maze comparison. See docs/spatial_synapses.md.

Run from the repository root with `python -m tasks.mazes.spatial_experiment`.
This uses the real CTM with an identity backbone and explicit observation
coordinates. It is a controlled pilot, not a reproduction of the image-maze
benchmark. Artifacts include configuration, split hashes, checkpoints, traces
and raw per-seed results. No improvement is assumed or asserted.
"""

import argparse
from collections import deque
import csv
import platform
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from models.ctm import ContinuousThoughtMachine
from tasks.mazes.spatial_controls import CONTROLS, mechanism_control, parameter_audit
from tasks.mazes.spatial_config import add_spatial_arguments, spatial_model_kwargs
from tasks.mazes.spatial_diagnostics import probe_synapses, synchronization_metrics


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
STOP = 4


def shortest_route(open_cells, start, goal):
    queue = deque([start])
    parent = {start: None}
    while queue:
        point = queue.popleft()
        if point == goal:
            route = []
            while parent[point] is not None:
                point, direction = parent[point]
                route.append(direction)
            return route[::-1] + [STOP]
        for direction, delta in enumerate(DIRECTIONS):
            nxt = (point[0]+delta[0], point[1]+delta[1])
            if (0 <= nxt[0] < open_cells.shape[0] and 0 <= nxt[1] < open_cells.shape[1]
                    and open_cells[nxt] and nxt not in parent):
                parent[nxt] = (point, direction)
                queue.append(nxt)
    raise ValueError('maze has no solution')


def generate_maze(rng, cells, loop_probability=0.15):
    """Random DFS tree with optional extra corridors; random distinct endpoints."""
    open_cells = np.zeros((2*cells+1, 2*cells+1), dtype=bool)
    start_room = tuple(int(v) for v in rng.integers(cells, size=2))
    stack, visited = [start_room], {start_room}
    while stack:
        row, col = stack[-1]
        open_cells[2*row+1, 2*col+1] = True
        choices = [(row+dr, col+dc) for dr, dc in DIRECTIONS
                   if 0 <= row+dr < cells and 0 <= col+dc < cells
                   and (row+dr, col+dc) not in visited]
        if not choices:
            stack.pop()
            continue
        nxt = choices[int(rng.integers(len(choices)))]
        open_cells[row+nxt[0]+1, col+nxt[1]+1] = True
        stack.append(nxt)
        visited.add(nxt)
    for row in range(cells):
        for col in range(cells):
            for dr, dc in ((1, 0), (0, 1)):
                if row+dr < cells and col+dc < cells and rng.random() < loop_probability:
                    open_cells[2*row+1+dr, 2*col+1+dc] = True
    endpoints = rng.choice(cells*cells, 2, replace=False)
    start, goal = [(2*int(v//cells)+1, 2*int(v % cells)+1) for v in endpoints]
    return open_cells, start, goal


def encode_maze(open_cells, start, goal, horizon):
    route = shortest_route(open_cells, start, goal)
    if len(route) > horizon:
        raise ValueError('route exceeds horizon; increase --horizon (routes are never truncated)')
    y, x = np.meshgrid(np.linspace(-1, 1, open_cells.shape[0]),
                       np.linspace(-1, 1, open_cells.shape[1]), indexing='ij')
    observation = np.stack((open_cells, np.zeros_like(open_cells), np.zeros_like(open_cells), y, x)).astype('float32')
    observation[1, start[0], start[1]] = 1
    observation[2, goal[0], goal[1]] = 1
    target = np.full(horizon, -100, dtype='int64')
    target[:len(route)] = route
    return observation, target


def make_dataset(count, cells, seed, horizon, excluded=()):
    rng = np.random.default_rng(seed)
    used = set(excluded)
    observations, targets, fingerprints = [], [], []
    for _ in range(count * 1000):
        maze, start, goal = generate_maze(rng, cells)
        # Split by wall layout up to rotation, not just by endpoint pair.
        fingerprint = hashlib.sha256(min(np.rot90(maze, k).tobytes() for k in range(4))).hexdigest()
        if fingerprint in used:
            continue
        used.add(fingerprint)
        observation, target = encode_maze(maze, start, goal, horizon)
        observations.append(observation)
        targets.append(target)
        fingerprints.append(fingerprint)
        if len(targets) == count:
            return torch.from_numpy(np.stack(observations)), torch.from_numpy(np.stack(targets)), fingerprints
    raise ValueError('not enough distinct maze layouts; reduce dataset size or increase cells')


def rotate_dataset(observations, horizon):
    encoded = []
    for observation in observations.numpy():
        rotated = np.rot90(observation, axes=(-2, -1))
        start = tuple(np.argwhere(rotated[1] > 0)[0])
        goal = tuple(np.argwhere(rotated[2] > 0)[0])
        encoded.append(encode_maze(rotated[0] > 0, start, goal, horizon))
    return tuple(torch.from_numpy(np.stack(values)) for values in zip(*encoded))


def route_metrics(actions, observation, target):
    """Require a collision-free route ending with STOP at the goal."""
    point = tuple(np.argwhere(observation[1] > 0)[0])
    goal = tuple(np.argwhere(observation[2] > 0)[0])
    optimum = int((target != -100).sum()) - 1
    for step, action in enumerate(actions):
        if action == STOP:
            success = point == goal
            return success, False, step / max(1, optimum) if success else None
        dr, dc = DIRECTIONS[int(action)]
        point = (point[0]+dr, point[1]+dc)
        if not (0 <= point[0] < observation.shape[-2] and 0 <= point[1] < observation.shape[-1]
                and observation[0, point[0], point[1]] > 0):
            return False, True, None
    return False, False, None  # no STOP; a partial route is not a success


def prediction_metrics(logits, observations, targets):
    """Dataset-wide metrics; logits are [example, route slot, action, CTM tick].

    CE is token-weighted, including STOP, excluding -100 padding. Success is
    collision-free arrival followed by STOP; exact sequence additionally
    requires the canonical BFS route. Neither uses labels to select a tick.
    """
    actions = logits.argmax(2).numpy()
    target_np, observation_np = targets.numpy(), observations.numpy()
    outcomes = [route_metrics(action[:, -1], obs, target)
                for action, obs, target in zip(actions, observation_np, target_np)]
    success, collision, efficiency = zip(*outcomes)
    solved_by_tick = np.asarray([
        [route_metrics(a[:, tick], o, t)[0] for a, o, t in zip(actions, observation_np, target_np)]
        for tick in range(logits.shape[-1])])
    lengths = (target_np != -100).sum(-1) - 1
    mask = target_np != -100
    ce = torch.stack([F.cross_entropy(logits[..., tick].transpose(1, 2), targets, reduction='none')
                      for tick in range(logits.shape[-1])], -1)
    ce_by_tick = ce.sum((0, 1)) / mask.sum()
    first = [int(solved_by_tick[:, i].argmax())+1 if solved_by_tick[:, i].any() else None
             for i in range(len(targets))]
    return dict(
        examples=len(targets), valid_tokens=int(mask.sum()),
        cross_entropy=float(ce_by_tick[-1]), cross_entropy_by_tick=ce_by_tick.tolist(),
        success_rate=float(np.mean(success)), collision_rate=float(np.mean(collision)),
        token_accuracy=float((actions[..., -1][mask] == target_np[mask]).mean()),
        exact_sequence_rate=float(((actions[..., -1] == target_np) | ~mask).all(-1).mean()),
        per_example_success=list(map(bool, success)), per_example_collision=list(map(bool, collision)),
        per_example_cross_entropy=(ce[..., -1].sum(-1) / torch.from_numpy(mask.sum(-1))).tolist(),
        route_length_ratio_success_only=float(np.mean([v for v in efficiency if v is not None])) if any(success) else None,
        success_by_tick=solved_by_tick.mean(-1).tolist(), first_success_tick=first,
        length_buckets={f'{lo}-{hi}': dict(count=int(((lengths >= lo) & (lengths <= hi)).sum()),
                         success_rate=float(np.asarray(success)[(lengths >= lo) & (lengths <= hi)].mean())
                         if ((lengths >= lo) & (lengths <= hi)).any() else None)
                        for lo, hi in [(1, 8), (9, 16), (17, targets.shape[1]-1)]},
    )


@torch.no_grad()
def evaluate(model, observations, targets, batch_size, device, grid=None, boundary='zeros', trace_path=None):
    """Ordered, no-gradient evaluation; restore training mode and consume no RNG.

    Pass no trace_path for cheap periodic evaluation. Traces, when requested,
    cover only the first batch; predictive metrics always cover the whole set.
    """
    was_training = model.training
    # CTM.forward clamps these parameters in place, even in eval mode. Do not
    # let diagnostic frequency change the optimizer's parameter trajectory.
    decay_parameters = (model.decay_params_action, model.decay_params_out)
    saved_decay = [p.detach().clone() for p in decay_parameters]
    model.eval()
    logits, traces = [], None
    try:
        for offset in range(0, len(targets), batch_size):
            batch = observations[offset:offset+batch_size].to(device)
            track = trace_path is not None and offset == 0
            result = model(batch, track=track)
            logits.append(result[0].cpu())
            if track:
                traces = result[4]
        logits = torch.cat(logits).reshape(len(targets), targets.shape[1], 5, model.iterations)
        metrics = prediction_metrics(logits, observations, targets)
        if trace_path is not None:
            trace_path = Path(trace_path)
            np.savez_compressed(trace_path, post_activations=traces,
                                predictions=logits.argmax(2).numpy(), targets=targets.numpy(),
                                shortest_lengths=(targets.numpy() != -100).sum(-1)-1)
            metrics.update(synchronization=synchronization_metrics(traces, grid, boundary),
                           trace_file=trace_path.name)
        return metrics
    finally:
        for parameter, saved in zip(decay_parameters, saved_decay):
            parameter.copy_(saved)
        model.train(was_training)


CURVE_FIELDS = ('cross_entropy', 'success_rate', 'exact_sequence_rate', 'collision_rate', 'token_accuracy')


def write_curves(path, curve):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['update', 'split', *CURVE_FIELDS])
        writer.writeheader()
        for point in curve:
            for split, metrics in point['evaluations'].items():
                writer.writerow(dict(update=point['update'], split=split,
                                     **{name: metrics[name] for name in CURVE_FIELDS}))


def overfit_gate(curve):
    """A prerequisite, not a claim of generalization: two successful checkpoints."""
    def passed(point):
        m = point['evaluations']['train']
        return (point['update'] > 0 and m['cross_entropy'] <= 0.1
                and m['success_rate'] == m['exact_sequence_rate'] == 1
                and m['collision_rate'] == 0)
    passing = [point['update'] for point in curve if passed(point)]
    return dict(passed=len(curve) >= 2 and all(passed(p) for p in curve[-2:]),
                first_passing_evaluation=passing[0] if passing else None,
                rule='CE <= 0.1, 100% success and exact sequence, zero collisions at last two positive-update evaluations')


def run(args):
    args = argparse.Namespace(**dict(eval_every=50, full_batch=False, controls=['none'],
                                     mechanism_diagnostics=True, overfit=False) | vars(args))
    if args.steps < 1 or min(args.train_examples, args.test_examples, args.batch_size, args.threads) < 1:
        raise ValueError('steps, sample counts and batch size must be positive')
    if args.eval_every < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('eval_every and lr must be positive and finite')
    if not args.controls or len(set(args.controls)) != len(args.controls) or any(
            c not in CONTROLS[:-1] for c in args.controls):
        raise ValueError('controls must be unique supported training controls (zero-speed uses --wave_speed 0)')
    if args.cells < 3 or args.iterations < 4:
        raise ValueError('use at least 3 maze cells and 4 internal ticks')
    if args.horizon < 2*(args.cells+1)**2-1:
        raise ValueError('horizon must cover every possible larger-maze route plus STOP')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('seeds must be unique')
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    train_x, train_y, train_hash = make_dataset(args.train_examples, args.cells, args.data_seed, args.horizon)
    test_x, test_y, test_hash = make_dataset(args.test_examples, args.cells, args.data_seed+1, args.horizon, train_hash)
    larger_x, larger_y, larger_hash = make_dataset(args.test_examples, args.cells+1, args.data_seed+2, args.horizon)
    rotated_x, rotated_y = rotate_dataset(test_x, args.horizon)
    grid = tuple(args.spatial_grid or (8, 8))
    variants = ['dense', 'spatial', 'wave']
    source_paths = ['models/ctm.py', 'models/modules.py', 'models/spatial_synapses.py',
                    'tasks/mazes/spatial_experiment.py', 'tasks/mazes/spatial_config.py',
                    'tasks/mazes/spatial_diagnostics.py', 'tasks/mazes/spatial_controls.py']
    source_hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths}
    results = dict(schema_version=2, config=vars(args), python_version=platform.python_version(),
                   platform=platform.platform(), torch_version=torch.__version__, numpy_version=np.__version__,
                   device=args.device, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                   source_sha256=source_hashes,
                   split_hashes=dict(train=train_hash, id=test_hash, larger=larger_hash), runs=[])
    # Dataset arrays make reruns independent of changes to the generator.
    np.savez_compressed(out_dir/'datasets.npz', train_x=train_x, train_y=train_y,
                        id_x=test_x, id_y=test_y, larger_x=larger_x, larger_y=larger_y,
                        rotated_x=rotated_x, rotated_y=rotated_y)
    results['dataset_sha256'] = hashlib.sha256((out_dir/'datasets.npz').read_bytes()).hexdigest()
    for seed in args.seeds:
        common_weights = None
        for control in args.controls:
            for variant in variants:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                config_args = argparse.Namespace(**(vars(args) | dict(synapse_type=variant, spatial_grid=grid)))
                model_kwargs = dict(
                    iterations=args.iterations, d_model=int(np.prod(grid)), d_input=16, heads=2,
                    n_synch_out=64, n_synch_action=32, n_random_pairing_self=8,
                    synapse_depth=1, memory_length=5, deep_nlms=True, memory_hidden_dims=8,
                    do_layernorm_nlm=False, backbone_type='none', positional_embedding_type='none',
                    out_dims=args.horizon*5, prediction_reshaper=[args.horizon, 5],
                    neuron_select_type='random-pairing', **spatial_model_kwargs(config_args))
                model = ContinuousThoughtMachine(**model_kwargs).to(args.device)
                with torch.no_grad():
                    model(train_x[:1].to(args.device))  # materialize lazy parameters
                if common_weights is None:
                    common_weights = {k: v.detach().clone() for k, v in model.state_dict().items()
                                      if not k.startswith('synapses.')}
                else:
                    # Same non-synapse initialization and neuron pairings for paired seeds.
                    model.load_state_dict(model.state_dict() | common_weights, strict=True)
                tag = f'{variant}-seed{seed}' + (f'-{control}' if control != 'none' else '')
                model.eval()
                with mechanism_control(model, control), torch.no_grad():
                    initial_probe = probe_synapses(model, grid)
                    initial_sync = synchronization_metrics(model(test_x[:args.batch_size].to(args.device), track=True)[4],
                                                            grid, args.spatial_boundary)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
                generator = torch.Generator().manual_seed(seed+10000)
                losses, curve, elapsed = [], [], 0.0
                if args.device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats()
                # Curves evaluate the entire fixed training and ID sets, at update 0
                # and after optimizer updates. Their time is excluded from training.
                def record(update):
                    curve.append(dict(update=update, evaluations={
                        name: evaluate(model, x, y, args.batch_size, args.device)
                        for name, x, y in [('train', train_x, train_y), ('id', test_x, test_y)]}))
                    write_curves(out_dir/f'{tag}-curve.csv', curve)
                    print(json.dumps(dict(tag=tag, update=update, **{
                        name: {k: m[k] for k in CURVE_FIELDS}
                        for name, m in curve[-1]['evaluations'].items()})), flush=True)
                with mechanism_control(model, control):
                    record(0)
                    model.train()
                    for step in range(args.steps):
                        if args.device.startswith('cuda'):
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        indices = (torch.arange(len(train_y)) if args.full_batch else
                                   torch.randint(len(train_y), (args.batch_size,), generator=generator))
                        pred = model(train_x[indices].to(args.device))[0][..., -1]
                        loss = F.cross_entropy(pred.reshape(-1, 5), train_y[indices].to(args.device).reshape(-1))
                        if not torch.isfinite(loss):
                            raise RuntimeError(f'nonfinite loss for {tag} at step {step}')
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        if args.device.startswith('cuda'):
                            torch.cuda.synchronize()
                        elapsed += time.perf_counter()-start
                        if (step+1) % args.eval_every == 0 or step+1 == args.steps:
                            record(step+1)
                    model.eval()
                    run_result = dict(variant=variant, control=control, seed=seed, loss=losses,
                                      learning_curve=curve, overfit_gate=overfit_gate(curve),
                                      training_seconds=elapsed, seconds_per_step=elapsed/args.steps,
                                      parameters=sum(p.numel() for p in model.parameters()),
                                      parameter_audit=parameter_audit(model),
                                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                                      initial_propagation=initial_probe, final_propagation=probe_synapses(model, grid),
                                      initial_synchronization=initial_sync, evaluations={})
                    for name, x, y in [('train', train_x, train_y), ('id', test_x, test_y),
                                       ('larger', larger_x, larger_y), ('rotated', rotated_x, rotated_y)]:
                        run_result['evaluations'][name] = evaluate(model, x, y, args.batch_size, args.device, grid,
                                                                  args.spatial_boundary, out_dir/f'{tag}-{name}.npz')
                # Lesions are applied to an unmodified trained baseline, one at a time.
                # They reveal dependence, whereas retraining controls test sufficiency.
                if args.mechanism_diagnostics and control == 'none':
                    run_result['train_interventions'] = {}
                    for lesion in CONTROLS[1:]:
                        if lesion == 'zero-speed' and variant != 'wave':
                            continue
                        with mechanism_control(model, lesion):
                            run_result['train_interventions'][lesion] = evaluate(
                                model, train_x, train_y, args.batch_size, args.device)
                torch.save(dict(model_state_dict=model.state_dict(), model_kwargs=model_kwargs, seed=seed,
                                control=control, optimizer_state_dict=optimizer.state_dict(), steps=args.steps),
                           out_dir/f'{tag}.pt')
                results['runs'].append(run_result)
                (out_dir/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
                print(json.dumps(dict(variant=variant, control=control, seed=seed, final_loss=losses[-1],
                                      seconds=elapsed, overfit_gate=run_result['overfit_gate'],
                                      success={k: v['success_rate'] for k, v in run_result['evaluations'].items()})), flush=True)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_spatial_arguments(parser, include_synapse_type=False)
    # This runner always compares all three modes; reuse only their configuration.
    parser.set_defaults(spatial_grid=[8, 8])
    parser.add_argument('--overfit', action='store_true', help='tiny full-batch preset; explicit flags override it')
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--full_batch', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--controls', choices=CONTROLS[:-1], nargs='+', default=['none'])
    parser.add_argument('--mechanism_diagnostics', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--data_seed', type=int, default=1729)
    parser.add_argument('--train_examples', type=int, default=512)
    parser.add_argument('--test_examples', type=int, default=128)
    parser.add_argument('--cells', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', default='logs/spatial-pilot')
    if parser.parse_args(argv).overfit:
        parser.set_defaults(steps=500, seeds=[0], train_examples=4, test_examples=16,
                            cells=3, horizon=32, iterations=8, batch_size=4,
                            full_batch=True, eval_every=25, output='logs/spatial-overfit')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
