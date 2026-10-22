"""Shared CLI/checkpoint configuration for the optional spatial CTM."""


def add_spatial_arguments(parser, *, include_synapse_type=True):
    if include_synapse_type:
        parser.add_argument('--synapse_type', choices=['dense', 'spatial', 'wave'], default='dense')
    parser.add_argument('--spatial_grid', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--spatial_radius', type=int, default=1)
    parser.add_argument('--spatial_boundary', choices=['zeros', 'circular'], default='zeros')
    parser.add_argument('--wave_speed', type=float, default=0.5)
    parser.add_argument('--wave_damping', type=float, default=0.2)
    parser.add_argument('--wave_frequency', type=float, default=0.5)
    parser.add_argument('--wave_dt', type=float, default=0.5)
    parser.add_argument('--wave_substeps', type=int, default=1)
    parser.add_argument('--wave_drive', type=float, default=1.0)


def spatial_model_kwargs(args):
    """Old checkpoints without these options still construct the dense CTM."""
    kind = getattr(args, 'synapse_type', 'dense')
    if kind == 'dense':
        return {'synapse_type': kind}
    grid = getattr(args, 'spatial_grid', None)
    if grid is None:
        raise ValueError('--spatial_grid HEIGHT WIDTH is required for spatial/wave synapses')
    config = dict(grid_shape=tuple(grid), radius=getattr(args, 'spatial_radius', 1),
                  boundary=getattr(args, 'spatial_boundary', 'zeros'))
    for key, default in [('wave_speed', 0.5), ('wave_damping', 0.2),
                         ('wave_frequency', 0.5), ('wave_dt', 0.5),
                         ('wave_substeps', 1), ('wave_drive', 1.0)]:
        config[key] = getattr(args, key, default)
    return {'synapse_type': kind, 'spatial_config': config}


def maze_checkpoint_weights(checkpoint):
    """Accept both training checkpoints and older released maze checkpoints."""
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint['state_dict']
