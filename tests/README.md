# Tests

To execute all tests, run the following:

```bash
PYTHONPATH=. pytest tests/tests.py
```

## Golden Test
The golden test is designed to check if changes in the code have resulted in a difference in model behaviour. It takes a pre-defined input, runs the model, and compares the output to what is expected. If something changed in the model and the outputs are different, the test will fail.

The golden test can be ran with:
```bash
PYTHONPATH=. pytest tests/tests.py::test_golden_<task>
```

where `<task>` is one of `parity`, `qamnist`, `rl`.

## Spatial synapses

The download-free locality, wave propagation, gradient, checkpoint, diagnostic
and generated-maze smoke tests require only PyTorch, NumPy and pytest:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_spatial_synapses.py tests/test_spatial_experiment.py -q
```

See [the experiment guide](../docs/spatial_synapses.md). Run the existing
`tests/tests.py` explicitly as well: its filename is not discovered by pytest's
default `test_*.py` pattern.

The experiment tests also cover STOP/collision/alternate-route semantics,
token-weighted CE with padding, repeatable whole-training-set evaluation,
restoration of training mode and in-place-clamped decay parameters, intervention
cleanup on exceptions, checkpoint reload with controls, and identical training
weights/losses when evaluation cadence changes. Learning a dataset is checked
by the separate overfit experiment, not asserted in a fragile training unit test.
