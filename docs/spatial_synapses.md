# Optional neuronal sheets for the Continuous Thought Machine

This extends Sakana's **Continuous Thought Machine (CTM)**, not Continuous
Memory Machines. It adds a controlled architectural experiment; it does not
implement Vector-HaSH, an associative-memory scaffold, STDP or Hebbian learning.
All trainable weights still use backpropagation through the CTM's internal ticks.

## Architecture and API

The default `synapse_type="dense"` retains the original Linear/GLU/LayerNorm or
SynapseUNET, initialization order, state-dict keys, and output/`track=True`
contract. Existing constructors and checkpoints need no new arguments.

Two options are available on the base `ContinuousThoughtMachine`, including
the single-device and distributed image-maze training entry points:

| Mode | Recurrent interaction | Additional state |
| --- | --- | --- |
| `dense` | Original dense mixing; `synapse_depth` still selects Linear vs UNET | None |
| `spatial` | Independent learned weights for each target neuron and each neighbor in a square radius | None |
| `wave` | The same local learned drive, driving a damped lattice-wave solver | Displacement and velocity per site |

```python
from models.ctm import ContinuousThoughtMachine
from models.spatial_synapses import SpatialSynapseConfig

# Add these keywords to an existing CTM constructor with d_model=256:
spatial_kwargs = dict(
    synapse_type="wave",  # or "spatial"
    spatial_config=SpatialSynapseConfig(
        grid_shape=(16, 16), radius=1, boundary="zeros",
        wave_speed=0.5, wave_damping=0.2, wave_frequency=0.5,
        wave_dt=0.5, wave_substeps=1, wave_drive=1.0,
    ),
)
model = ContinuousThoughtMachine(**existing_model_kwargs, **spatial_kwargs)
```

`grid_shape` is explicit, with `H*W == d_model`; index `i` maps to `(i//W, i%W)`.
Rectangular grids are supported. The neuronal sheet is independent of the
maze/pixel dimensions. `synapse_depth` affects only dense mode. Spatial mode
uses one learned local layer. `radius=0` gives a sitewise recurrent control.
`boundary="zeros"` means outside the sheet is fixed at zero (Dirichlet), **not**
reflecting or absorbing boundaries. `circular` gives a periodic sheet.

The recurrent input has shape `[B,D]`. Local neighborhoods produce two values
per site, to which a learned linear projection of the `[B,d_input]` attention
output is added. The site drive is `tanh(candidate) * sigmoid(gate)`. There is
no normalization over neurons and no hidden dense recurrent matrix; recurrent
weight storage is `2*D*(2*radius+1)^2`. External input projection remains dense.

**Locality is conditional on the external drive.** The CTM still has global
synchronization readout and synchronization-to-attention feedback. A perturbation
can therefore influence distant sites via attention on the next tick. This
variant is not a wholly local cortical simulation. Locality tests hold attention
fixed, and the propagation probe bypasses NLMs and clamps attention to zero.

The wave branch integrates, at unit grid spacing,

```text
v_next = (1 - damping*dt)*v
         + dt*(speed^2*L(u) - frequency^2*u + drive_gain*local_drive)
u_next = u + dt*v_next
```

`L` is the four-neighbor five-point Laplacian. `wave_substeps` holds the drive
fixed while performing that many updates per CTM tick; the physical duration
of a tick is `substeps*dt`. Neighbor communication by the wave stencil is at
most one lattice edge per substep. The learned drive has its own radius.
The restoring frequency anchors the uniform mode; setting it to zero permits
uniform displacement drift under persistent forcing (and ballistic drift for
an undamped initial uniform velocity). Nonzero speed is necessary for spatial
wave transport. Setting speed to zero retains the local temporal oscillator.

Configuration checks enforce finite nonnegative coefficients, positive `dt`,
integer substeps, `damping*dt <= 1`, and
`dt^2*(8*speed^2 + frequency^2) + 2*damping*dt < 4`. This is a conservative
stability condition for the unforced linear solver's nonzero modes, **not** a
stability guarantee for the driven CTM/NLM/attention feedback system.

State is a pair of tensors passed explicitly between ticks, kept on the input
device/dtype, and reset to zero on every base-CTM forward call. Nothing is stored
on the module between examples, in its state dict, or in a detached cache.
Gradients flow through all wave substeps. Save the constructor configuration
alongside model weights; configuration is not encoded in tensor shapes alone.
This first version does not expose spatial options in the separate QAMNIST,
sorting or RL subclasses, whose existing dense behavior is retained.

## Reproducible generated-maze pilot

Only PyTorch and NumPy are needed for this runner (pytest for its tests). From
the repository root:

```bash
# A quick execution check; these updates are insufficient to assess learning.
python -m tasks.mazes.spatial_experiment \
  --steps 10 --seeds 0 --train_examples 32 --test_examples 16 \
  --batch_size 8 --iterations 8 --output logs/spatial-smoke

# Prespecified pilot: all three modes, three paired seeds, fixed data.
python -m tasks.mazes.spatial_experiment \
  --steps 1000 --seeds 0 1 2 --train_examples 512 --test_examples 128 \
  --cells 4 --horizon 64 --iterations 16 --batch_size 32 \
  --spatial_grid 8 8 --output logs/spatial-pilot

# A control for temporal filtering versus spatial wave transport.
python -m tasks.mazes.spatial_experiment \
  --steps 1000 --seeds 0 1 2 --wave_speed 0 \
  --output logs/spatial-zero-speed
```

Use a fresh output directory each time. `--device cuda` runs on CUDA if
available; for CUDA deterministic matrix multiplication, set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting. CPU is the default. The
runner always compares dense/spatial/wave. Grid and wave flags configure the
spatial branches.

The task is open-loop prediction of a route as up/right/down/left/STOP. DFS
mazes have random extra corridors (probability 0.15 per potential corridor)
and random distinct endpoints. Train/ID splits exclude identical wall layouts,
including rotations, using canonical rotation hashes. A 4x4 room graph is a
9x9 observed lattice; the size-generalization test uses 5x5 rooms / 11x11 lattice.
An additional test rotates the held-out ID mazes by 90 degrees. It is a paired
orientation test, not an independent test set or a claim that orientation was
absent during training. Larger-maze routes can be longer; report the supplied
length buckets as well as aggregate accuracy.

The observation has five channels: free space, start, goal, row coordinate,
column coordinate. These are input/world coordinates, distinct from the latent
neuronal topology. All modes use the actual CTM, an identity backbone, 16 input
features, 64 neurons, 5-tick NLM memory, and the same readout pairings. This
isolates the recurrent change without a ResNet pretraining confound. It is a
different, smaller benchmark than the existing image-maze task.

Training uses final-tick cross-entropy on the shortest route, including STOP,
ignores padding, never truncates routes, and clips gradients at 1.0. Each seed
shares the same non-synapse initial weights, neuron pairings, data, and minibatch
order across variants. Local and wave modes have identical trainable parameter
counts and initial local weights. Dense is compared at fixed neuronal width,
not fixed parameter count: log parameter counts and cost explicitly. Run both
an equal-update comparison and, subsequently, an equal-runtime comparison
before making efficiency claims. Three seeds are exploratory; expand seeds
and report paired differences and uncertainty before drawing conclusions.

Artifacts per run:

- `results.json`: full configuration, software versions, source revision and
  dirty-state indicator, source-file hashes, split hashes, losses, timings, parameter count,
  CUDA peak memory (if applicable), and all per-seed metrics.
- `datasets.npz`: exact train/evaluation arrays, to preserve the generated data.
- `<variant>-seed<seed>.pt`: weights, constructor kwargs, optimizer state, steps.
- `<variant>-seed<seed>-<split>.npz`: predicted routes at every tick, targets,
  shortest lengths and post-activation traces for the first evaluation batch.

Reload pilot weights with:

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model = ContinuousThoughtMachine(**checkpoint["model_kwargs"]).eval()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
```

## Diagnostics and interpretation

| Question | Measured outputs | Necessary caution/control |
| --- | --- | --- |
| Does an impulse propagate? | Per-neuron first-arrival tick, distance, squared-response distance, far-field energy fraction and L2 response over 16 ticks | Paired perturbed/unperturbed isolated synapses, zero attention, no NLM; threshold is reported. This is not whole-model signal speed. Dense neurons are assigned the same grid solely for comparison. |
| Does it solve or generalize? | Collision-free success with STOP at the goal, exact labeled route, token accuracy, successful-route length / shortest length, success by tick, first successful tick, route-length buckets | Valid alternate routes can succeed without matching the BFS label. First success at any tick is an oracle diagnostic, not an available stopping policy. Main score is final-tick valid success. |
| Does synchronization develop? | Neighbor/distant Pearson correlation, phase-locking value, global phase order, dominant frequency and spectral concentration; before and after training | DC is removed; silent sites are excluded. Compare against independent per-neuron time permutations and initial weights. Short, nonstationary traces make Hilbert-phase estimates unreliable. These statistics do not establish causality, spontaneous waves or symbolic abstraction. |
| What does it cost? | Trainable parameters, training seconds/update and optional peak CUDA allocation | The wave solver adds state/computation, not trainable parameters. Timings from a smoke run are not a benchmark. |

For evidence of computationally useful waves, first verify solver/locality
tests; then compare wave with local and zero-speed temporal controls under
paired seeds. Test whether a held-out success difference survives speed/gain
tuning chosen only on a validation split and comparable compute budgets.
Inspect arrival maps alongside phase gradients; high phase locking alone
can reflect common drive. The first pilot does not implement a causal
phase-scrambling intervention or learn synaptic weights through a local rule.
Stop attributing benefit to spatial propagation if the zero-speed control
explains it, or if gains disappear with compute/parameter/drive-scale controls.

## Existing image-maze task

Append to an existing CTM maze-training command with `d_model=256`:

```bash
--synapse_type spatial --spatial_grid 16 16 --spatial_radius 1
```

For explicit propagation, use `--synapse_type wave` and optionally the
`--wave_*` flags shown by `--help`. Both `tasks.mazes.train` and
`tasks.mazes.train_distributed` accept these options; the existing analysis
loader reconstructs the spatial configuration from checkpoint `args`.
Image data, distributed execution, mixed precision, CUDA and compilation
must be validated on the intended training hardware before a long run.

## Validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_spatial_synapses.py tests/test_spatial_experiment.py -q

# Also exercises legacy parity/QAMNIST/RL; requires the original task dependencies
# and downloads MNIST if it is absent.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/tests.py tests/test_spatial_synapses.py tests/test_spatial_experiment.py -q
```

See [spatial_validation.md](spatial_validation.md) for the checks actually run
and the small smoke-run results. No trained-performance improvement is claimed.
