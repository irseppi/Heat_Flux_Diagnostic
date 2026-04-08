"""
Configuration settings for the Underworld3 annulus convection model.

"""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class EarthConfig:
    """Physical reference values for an Earth-like mantle analogue."""

    planet_name: str = "Earth"
    planet_radius_m: float = 6.3710084e6
    core_radius_m: float = 3.48e6
    gravity_m_s2: float = 9.8
    surface_temperature_K: float = 273.0
    cmb_temperature_K: float = 3600.0
    density_kg_m3: float = 3300.0
    thermal_expansivity_1_K: float = 2.0e-5
    thermal_diffusivity_m2_s: float = 1.1394e-6
    thermal_conductivity_W_mK: float = 4.7
    heat_capacity_J_kgK: float = 1250.0
    reference_viscosity_Pa_s: float = 1.02e21

    @property
    def mantle_thickness_m(self) -> float:
        return self.planet_radius_m - self.core_radius_m

    @property
    def delta_temperature_K(self) -> float:
        return self.cmb_temperature_K - self.surface_temperature_K


@dataclass
class ModelConfig:
    """Container for simulation and output parameters."""

    earth: EarthConfig = field(default_factory=EarthConfig)

    # Nondimensional annulus geometry
    inner_radius: float = 0.55
    outer_radius: float = 1.0
    mesh_resolution: int = 20
    qdegree: int = 3

    # Thermal boundary conditions
    temperature_inner: float = 1.0
    temperature_outer: float = 0.0

    # Baseline rheology / diffusivity
    viscosity: float = 1.0
    diffusivity: float = 1.0

    # Thermal perturbation
    perturbation_amplitude: float = 0.03
    perturbation_wavenumber: int = 4

    # Runtime
    max_steps: int = 1200
    output_interval: int = 25
    timestep_safety: float = 0.25
    max_dt: float = 5.0e-4
    min_dt: float = 1.0e-6

    # Diagnostics controls
    angular_bins: int = 72
    radial_bins: int = 100
    tail_fraction: float = 0.2
    boundary_tolerance_factor: float = 0.60

    # Output directories
    results_data_dir: str = str(PROJECT_ROOT / "results" / "data")
    results_figure_dir: str = str(PROJECT_ROOT / "results" / "figures")

    @property
    def shell_thickness(self) -> float:
        return self.outer_radius - self.inner_radius

    @property
    def cell_size(self) -> float:
        return self.shell_thickness / self.mesh_resolution
