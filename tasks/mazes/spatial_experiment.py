"""Small, download-free maze comparison. See docs/spatial_synapses.md.

Run from the repository root with `python -m tasks.mazes.spatial_experiment`.
This uses the real CTM with an identity backbone and explicit observation
coordinates. It is a controlled pilot, not a reproduction of the image-maze
benchmark. Artifacts include configuration, split hashes, checkpoints, traces
and raw per-seed results. No improvement is assumed or asserted.
"""

import argparse
from collections import deque
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


@torch.no_grad()
def evaluate(model, observations, targets, batch_size, device, grid, boundary, trace_path):
    model.eval()
    logits, traces = [], []
    for offset in range(0, len(targets), batch_size):
        batch = observations[offset:offset+batch_size].to(device)
        result = model(batch, track=offset == 0)
        logits.append(result[0].cpu())
        if offset == 0:
            traces = result[4]
    logits = torch.cat(logits).reshape(len(targets), targets.shape[1], 5, model.iterations)
    actions = logits.argmax(2).numpy()
    target_np, observation_np = targets.numpy(), observations.numpy()
    outcomes = [route_metrics(action[:, -1], obs, target)
                for action, obs, target in zip(actions, observation_np, target_np)]
    success, collision, efficiency = zip(*outcomes)
    solved_by_tick = []
    for tick in range(model.iterations):
        solved_by_tick.append([route_metrics(a[:, tick], o, t)[0]
                               for a, o, t in zip(actions, observation_np, target_np)])
    solved_by_tick = np.asarray(solved_by_tick)
    lengths = (target_np != -100).sum(-1) - 1
    mask = target_np != -100
    first = [int(solved_by_tick[:, i].argmax())+1 if solved_by_tick[:, i].any() else None
             for i in range(len(targets))]
    np.savez_compressed(trace_path, post_activations=traces,
                        predictions=actions, targets=target_np, shortest_lengths=lengths)
    return dict(
        examples=len(targets), success_rate=float(np.mean(success)),
        collision_rate=float(np.mean(collision)),
        token_accuracy=float((actions[..., -1][mask] == target_np[mask]).mean()),
        exact_sequence_rate=float(((actions[..., -1] == target_np) | ~mask).all(-1).mean()),
        route_length_ratio_success_only=float(np.mean([v for v in efficiency if v is not None])) if any(success) else None,
        success_by_tick=solved_by_tick.mean(-1).tolist(), first_success_tick=first,
        length_buckets={f'{lo}-{hi}': dict(count=int(((lengths >= lo) & (lengths <= hi)).sum()),
                         success_rate=float(np.asarray(success)[(lengths >= lo) & (lengths <= hi)].mean())
                         if ((lengths >= lo) & (lengths <= hi)).any() else None)
                        for lo, hi in [(1, 8), (9, 16), (17, targets.shape[1]-1)]},
        synchronization=synchronization_metrics(traces, grid, boundary),
        trace_file=trace_path.name,
    )


def run(args):
    if args.steps < 1 or min(args.train_examples, args.test_examples, args.batch_size, args.threads) < 1:
        raise ValueError('steps, sample counts and batch size must be positive')
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
                    'tasks/mazes/spatial_diagnostics.py']
    source_hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths}
    results = dict(config=vars(args), torch_version=torch.__version__, numpy_version=np.__version__,
                   device=args.device, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                   source_sha256=source_hashes,
                   split_hashes=dict(train=train_hash, id=test_hash, larger=larger_hash), runs=[])
    # Dataset arrays make reruns independent of changes to the generator.
    np.savez_compressed(out_dir/'datasets.npz', train_x=train_x, train_y=train_y,
                        id_x=test_x, id_y=test_y, larger_x=larger_x, larger_y=larger_y,
                        rotated_x=rotated_x, rotated_y=rotated_y)
    for seed in args.seeds:
        common_weights = None
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
            tag = f'{variant}-seed{seed}'
            model.eval()
            initial_probe = probe_synapses(model, grid)
            initial_sync = synchronization_metrics(model(test_x[:args.batch_size].to(args.device), track=True)[4],
                                                    grid, args.spatial_boundary)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
            generator = torch.Generator().manual_seed(seed+10000)
            losses, start = [], time.perf_counter()
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
            model.train()
            for step in range(args.steps):
                indices = torch.randint(len(train_y), (args.batch_size,), generator=generator)
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
            elapsed = time.perf_counter()-start
            model.eval()
            run_result = dict(variant=variant, seed=seed, loss=losses,
                              training_seconds=elapsed, seconds_per_step=elapsed/args.steps,
                              parameters=sum(p.numel() for p in model.parameters()),
                              peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                              initial_propagation=initial_probe, final_propagation=probe_synapses(model, grid),
                              initial_synchronization=initial_sync, evaluations={})
            for name, x, y in [('id', test_x, test_y), ('larger', larger_x, larger_y), ('rotated', rotated_x, rotated_y)]:
                run_result['evaluations'][name] = evaluate(model, x, y, args.batch_size, args.device, grid,
                                                          args.spatial_boundary, out_dir/f'{tag}-{name}.npz')
            torch.save(dict(model_state_dict=model.state_dict(), model_kwargs=model_kwargs, seed=seed,
                            optimizer_state_dict=optimizer.state_dict(), steps=args.steps), out_dir/f'{tag}.pt')
            results['runs'].append(run_result)
            (out_dir/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(variant=variant, seed=seed, final_loss=losses[-1],
                                  seconds=elapsed, success={k: v['success_rate'] for k, v in run_result['evaluations'].items()})), flush=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_spatial_arguments(parser, include_synapse_type=False)
    # This runner always compares all three modes; reuse only their configuration.
    parser.set_defaults(spatial_grid=[8, 8])
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
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
