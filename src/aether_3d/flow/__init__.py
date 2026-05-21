"""Lumina / Aether Flow Matching primitives.

Clean-room, fully retyped and re-documented implementation of the probability-path
+ velocity-field machinery originally present in both baseline repositories.

Public symbols:
    create_flow_transport, FlowTransport, FlowSampler
    InterpolationPath, LinearPath, GVPPath, VPPath
    PredictionTarget, LossWeighting
"""

from .path import (
    InterpolationPath,
    LinearPath,
    GVPPath,
    VPPath,
    get_path,
)
from .transport import (
    FlowTransport,
    FlowSampler,
    PredictionTarget,
    LossWeighting,
    create_flow_transport,
)
from .integrators import ode, sde
from .utils import expand_time_like_data, mean_flat

__all__ = [
    "InterpolationPath",
    "LinearPath",
    "GVPPath",
    "VPPath",
    "get_path",
    "FlowTransport",
    "FlowSampler",
    "PredictionTarget",
    "LossWeighting",
    "create_flow_transport",
    "ode",
    "sde",
    "expand_time_like_data",
    "mean_flat",
]
