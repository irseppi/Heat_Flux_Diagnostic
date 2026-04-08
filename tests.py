"""
Basic parameter checks for the annulus convection workflow.
"""

from config import ModelConfig


def run_basic_checks(config: ModelConfig) -> None:
    """Run simple validation checks."""
    assert config.outer_radius > config.inner_radius, "Outer radius must exceed inner radius."
    assert config.inner_radius > 0.0, "Inner radius must be positive."
    assert config.mesh_resolution > 0, "Mesh resolution must be positive."
    assert config.qdegree >= 2, "Quadrature degree should be at least 2."
    assert config.viscosity > 0.0, "Viscosity must be positive."
    assert config.diffusivity > 0.0, "Diffusivity must be positive."
    assert config.max_steps > 0, "max_steps must be positive."
    assert config.output_interval > 0, "output_interval must be positive."
    assert config.temperature_inner > config.temperature_outer, (
        "For the present setup, the inner boundary should be hotter than the outer boundary."
    )
    assert config.max_dt > 0.0, "max_dt must be positive."
    assert config.min_dt > 0.0, "min_dt must be positive."
    assert config.max_dt >= config.min_dt, "max_dt must be >= min_dt."
    assert config.timestep_safety > 0.0, "timestep_safety must be positive."
    assert config.perturbation_amplitude >= 0.0, "Perturbation amplitude must be non-negative."
    assert config.perturbation_wavenumber > 0, "Perturbation wavenumber must be positive."
    assert config.angular_bins >= 16, "Use at least 16 angular bins."
    assert config.radial_bins >= 20, "Use at least 20 radial bins."
    assert 0.0 < config.tail_fraction < 1.0, "tail_fraction must lie between 0 and 1."
    assert config.boundary_tolerance_factor > 0.0, "boundary_tolerance_factor must be positive."
