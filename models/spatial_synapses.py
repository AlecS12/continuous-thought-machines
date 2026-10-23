"""Local recurrent synapses on a neuronal sheet, with optional damped waves.

The sheet indexes latent neurons, not pixels in the observation. Attention is
an external, densely projected drive; locality is conditional on that drive.
There is deliberately no normalization across sites. Wave state is explicit,
so a module never retains a batch's activations between forward calls.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SpatialSynapseConfig:
    grid_shape: tuple[int, int]
    radius: int = 1
    boundary: str = "zeros"
    wave_speed: float = 0.5
    wave_damping: float = 0.2
    wave_frequency: float = 0.5
    wave_dt: float = 0.5
    wave_substeps: int = 1
    wave_drive: float = 1.0

    def __post_init__(self):
        if len(self.grid_shape) != 2 or any(
            not isinstance(n, Integral) or isinstance(n, bool) or n < 2
            for n in self.grid_shape
        ):
            raise ValueError("grid_shape must contain two integer dimensions >= 2")
        object.__setattr__(self, "grid_shape", tuple(self.grid_shape))
        if not isinstance(self.radius, Integral) or isinstance(self.radius, bool) or self.radius < 0:
            raise ValueError("radius must be a nonnegative integer")
        if self.radius >= min(self.grid_shape):
            raise ValueError("radius must be smaller than both grid dimensions")
        if self.boundary not in ("zeros", "circular"):
            raise ValueError("boundary must be 'zeros' or 'circular'")
        if not isinstance(self.wave_substeps, Integral) or isinstance(self.wave_substeps, bool) or self.wave_substeps < 1:
            raise ValueError("wave_substeps must be a positive integer")
        for name in ("wave_speed", "wave_damping", "wave_frequency", "wave_dt", "wave_drive"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.wave_dt == 0:
            raise ValueError("wave_dt must be positive")
        # Sufficient strict stability bound for the unforced nonzero modes:
        # eigenvalues of -L <= 8 for the unit-spacing, five-point stencil.
        # At zero restoring frequency, a uniform mode can drift.
        dt = self.wave_dt
        if self.wave_damping * dt > 1 or (
            dt**2 * (8 * self.wave_speed**2 + self.wave_frequency**2)
            + 2 * self.wave_damping * dt >= 4
        ):
            raise ValueError("wave parameters violate the explicit solver stability bound")


class SpatialSynapses(nn.Module):
    """Neuron-specific local weights and a pointwise gated, bounded drive.

    Input: [batch, d_input + d_model], ordered [attention, recurrent state].
    Output: (pre_activation [batch, d_model], state). In local mode state is
    None. In wave mode it is (displacement, velocity), each [batch, 1, H, W].
    Row-major neuron i occupies (i // W, i % W). Each site's weights are
    independent; storage and computation scale as O(D * (2*radius+1)**2).
    """

    def __init__(self, d_model, d_input, config, *, wave=False, dropout=0.0):
        super().__init__()
        if isinstance(config, dict):
            config = SpatialSynapseConfig(**config)
        if not isinstance(config, SpatialSynapseConfig):
            raise ValueError("spatial synapses require a SpatialSynapseConfig or dict")
        if math.prod(config.grid_shape) != d_model:
            raise ValueError("grid height * width must equal d_model")
        if not isinstance(d_input, Integral) or d_input < 1:
            raise ValueError("d_input must be a positive integer")
        self.config = config
        self.d_model = d_model
        self.d_input = d_input
        self.wave = wave
        self.dropout = nn.Dropout(dropout)
        kernel_elements = (2 * config.radius + 1)**2
        # [candidate/gate, target neuron, neighbor offset]
        self.local_weight = nn.Parameter(torch.empty(2, d_model, kernel_elements))
        nn.init.uniform_(self.local_weight, -1 / math.sqrt(kernel_elements), 1 / math.sqrt(kernel_elements))
        self.input_projection = nn.Linear(d_input, 2 * d_model)

    def _pad(self, field, radius):
        mode = "constant" if self.config.boundary == "zeros" else "circular"
        return F.pad(field, (radius,) * 4, mode=mode)

    def laplacian(self, field):
        """Five-point unit-spacing Laplacian (Dirichlet zero or periodic)."""
        padded = self._pad(field, 1)
        return (padded[..., 2:, 1:-1] + padded[..., :-2, 1:-1]
                + padded[..., 1:-1, 2:] + padded[..., 1:-1, :-2] - 4 * field)

    def local_drive(self, x):
        if x.ndim != 2 or x.shape[1] != self.d_input + self.d_model:
            raise ValueError("expected [batch, d_input + d_model] input")
        x = self.dropout(x)
        observation, recurrent = x.split((self.d_input, self.d_model), dim=-1)
        field = recurrent.reshape(-1, 1, *self.config.grid_shape)
        neighborhoods = F.unfold(self._pad(field, self.config.radius),
                                 kernel_size=2 * self.config.radius + 1)
        mixed = torch.einsum("bkn,cnk->bcn", neighborhoods, self.local_weight)
        mixed = mixed + self.input_projection(observation).reshape(-1, 2, self.d_model)
        return torch.tanh(mixed[:, 0]) * torch.sigmoid(mixed[:, 1])

    def propagate(self, drive, state=None):
        """Advance a driven damped wave by wave_substeps, holding drive fixed.

        v <- (1 - damping*dt)*v + dt*(speed**2*L(u) - frequency**2*u + gain*drive)
        u <- u + dt*v

        The restoring term controls the uniform mode on a periodic sheet.
        Stability of the coupled CTM/NLM/attention system is not guaranteed
        by the linear solver bound. Gradients through u and v are retained.
        """
        if drive.ndim != 2 or drive.shape[1] != self.d_model:
            raise ValueError("drive must have shape [batch, d_model]")
        drive = drive.reshape(-1, 1, *self.config.grid_shape)
        if state is None:
            displacement = torch.zeros_like(drive)
            velocity = torch.zeros_like(drive)
        else:
            if len(state) != 2 or any(
                item.shape != drive.shape or item.device != drive.device or item.dtype != drive.dtype
                for item in state
            ):
                raise ValueError("wave state must match drive batch, grid, device and dtype")
            displacement, velocity = state
        cfg = self.config
        for _ in range(cfg.wave_substeps):
            velocity = ((1 - cfg.wave_damping * cfg.wave_dt) * velocity
                        + cfg.wave_dt * (cfg.wave_speed**2 * self.laplacian(displacement)
                                         - cfg.wave_frequency**2 * displacement
                                         + cfg.wave_drive * drive))
            displacement = displacement + cfg.wave_dt * velocity
        return displacement.flatten(1), (displacement, velocity)

    def forward(self, x, state=None):
        drive = self.local_drive(x)
        if self.wave:
            return self.propagate(drive, state)
        if state is not None:
            raise ValueError("local-only synapses do not accept wave state")
        return drive, None
