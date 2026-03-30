"""
Basic parameter checks for the Underworld3 convection rewrite.
"""

from config import ModelConfig


def run_basic_checks(config: ModelConfig) -> None:
    """Run simple validation checks."""
    assert config.outer_radius > config.inner_radius, (
        "Outer radius must be greater than inner radius."
    )
    assert config.cell_size > 0.0, "Cell size must be positive."
    assert config.viscosity > 0.0, "Viscosity must be positive."
    assert config.diffusivity > 0.0, "Diffusivity must be positive."
    assert config.max_steps > 0, "max_steps must be positive."
    assert config.output_interval > 0, "output_interval must be positive."
