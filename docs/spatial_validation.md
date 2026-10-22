# Validation record — 2026-10-22

Implementation base: `f659febd3019640366d1af47e836fae0d976311b`.
CPU execution; Python 3.12, PyTorch 2.14.1+cu130, NumPy 2.3.5. CUDA was unavailable.

## Executed checks

- Combined suite at the first completed implementation: **171 passed** (138
  existing regressions plus 33 new tests), in 49.15 seconds. This includes the
  original parity, QAMNIST and RL golden tests, without changing their expected
  values. MNIST was downloaded for the QAMNIST checks.
- After adding the maze checkpoint-format regression: **34 new tests passed**
  in 1.42 seconds. Thus all 138 existing tests and all 34 new tests were exercised.
- Direct comparison against `models/ctm.py` extracted from the base commit:
  default dense CTM state-dict keys/values and every tracked output were
  bit-for-bit identical for `synapse_depth=1` and `synapse_depth=3` with matched
  PyTorch/NumPy seeds.
- Both image-maze training CLIs loaded and displayed help with the new flags.
- The actual image-maze analysis loader restored the legacy `state_dict` format
  for dense CTM and the trainers' `model_state_dict` format for spatial/wave CTM,
  with no missing/unexpected keys and bit-for-bit identical predictions.
- Python compilation and `git diff --check` passed.

Tests cover exact recurrent Jacobian locality (corners, interiors, multiple
radii, rectangular grids, zero and circular boundaries), finite-speed impulse
propagation, substep equivalence, finite differences/autograd, long unforced
wave decay, zero-speed control, CTM output/tracking shapes, all-parameter
finite gradients, batch isolation, strict state-dict round trips, configuration
errors, silent/oscillatory diagnostic controls, disjoint/reproducible maze
splits, route validity, and training/checkpointing of all three variants.

## Actual smoke experiment

The following command was run twice with identical losses and predictive
metrics; timings naturally differed. The final run is retained in
[spatial_smoke_results.json](spatial_smoke_results.json), including exact
configuration, source hashes, per-neuron arrivals and all measured statistics.

```bash
python -m tasks.mazes.spatial_experiment \
  --steps 10 --seeds 0 --train_examples 32 --test_examples 16 \
  --batch_size 8 --iterations 8 --output logs/spatial-smoke-final
```

| Variant | Parameters | Final training-batch CE | ID success | Larger success | Rotated success |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 40,818 | 1.610831 | 0% | 0% | 0% |
| spatial | 33,650 | 1.610940 | 0% | 0% | 0% |
| wave | 33,650 | 1.610939 | 0% | 0% | 0% |

All three modes produced finite losses, completed backward/optimizer steps,
wrote checkpoints, and generated propagation/synchronization/generalization
reports. None solved the held-out mazes after ten updates. These are execution
checks, **not evidence of improved learning or generalization**. Eight-tick phase
estimates and one seed cannot establish emergent synchronization. Do not infer
model efficiency from subsecond training measurements.

## Not executed

The 1,000-update / three-seed pilot, a zero-speed training control, long image-maze
training, CUDA, AMP, compilation and multi-GPU/DDP training were not run.
The code's presence in those training entry points is not validation of those
hardware paths. Local or biologically plausible learning is not implemented.
See [the experiment protocol](spatial_synapses.md) for the next comparisons.
