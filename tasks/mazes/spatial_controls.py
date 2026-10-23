"""Temporary diagnostic interventions; the production CTM is unchanged.

These are mechanism tests, not equivalent architectures. Even `disconnected`
retains global observation attention, its broadcast, per-neuron temporal
processing and a dense output projection. It is not a fully local network.
"""

from contextlib import contextmanager
from dataclasses import replace

import torch


CONTROLS = ('none', 'no-recurrence', 'fixed-query', 'self-pairs', 'disconnected', 'zero-speed')


@contextmanager
def mechanism_control(model, control='none'):
    """Install reversible hooks; use the same context for training and reloads.

    no-recurrence: zero outgoing state entering synapses and reset wave state
      every tick. NLM history and synchronization accumulators still persist.
    fixed-query: replace the query by zero at every tick. MultiheadAttention's
      learned query bias remains, but cannot depend on recurrent state.
    self-pairs: replace output pair (i,j) by (i,i), retaining feature count.
    disconnected: combine the three interventions above.
    zero-speed: remove only the wave Laplacian; retain oscillator memory.
    """
    if control not in CONTROLS:
        raise ValueError(f'unknown control: {control}')
    if control == 'zero-speed' and model.synapse_type != 'wave':
        raise ValueError('zero-speed requires wave synapses')
    handles = []
    old_right = model.out_neuron_indices_right.clone()
    old_config = getattr(model.synapses, 'config', None)
    try:
        if control in ('no-recurrence', 'disconnected'):
            def remove_recurrence(module, inputs):
                x = inputs[0]
                x = torch.cat((x[:, :model.d_input], torch.zeros_like(x[:, model.d_input:])), -1)
                return (x,) if model.synapse_type == 'dense' else (x, None)
            handles.append(model.synapses.register_forward_pre_hook(remove_recurrence))
        if control in ('fixed-query', 'disconnected'):
            handles.append(model.q_proj.register_forward_hook(
                lambda module, inputs, output: torch.zeros_like(output)))
        if control in ('self-pairs', 'disconnected'):
            if model.neuron_select_type != 'random-pairing':
                raise ValueError('self-pairs requires random-pairing')
            model.out_neuron_indices_right.copy_(model.out_neuron_indices_left)
        if control == 'zero-speed':
            model.synapses.config = replace(old_config, wave_speed=0)
        yield
    finally:
        for handle in handles:
            handle.remove()
        model.out_neuron_indices_right.copy_(old_right)
        if old_config is not None:
            model.synapses.config = old_config


def parameter_audit(model):
    counts = {name: sum(p.numel() for p in module.parameters())
              for name, module in model.named_children()}
    total = sum(p.numel() for p in model.parameters())
    return dict(total=total, modules=counts, ungrouped=total-sum(counts.values()),
                readout_fraction=counts['output_projector']/total)
