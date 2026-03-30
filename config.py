"""
Configuration settings for the Underworld3 thermal convection rewrite.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Container for simulation parameters."""

    inner_radius: float = 1.22
    outer_radius: float = 2.22
    cell_size: float = 0.05

    temperature_inner: float = 1.0
    temperature_outer: float = 0.0

    viscosity: float = 1.0
    diffusivity: float = 1.0

    max_steps: int = 200
    output_interval: int = 10

    results_data_dir: str = "results/data"
    results_figure_dir: str = "results/figures"
