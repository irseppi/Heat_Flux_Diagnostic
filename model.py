"""
Model setup utilities for annulus thermal convection in Underworld3.
"""

import sympy
import underworld3 as uw

from config import ModelConfig


def create_mesh(config: ModelConfig):
    """Create the annulus mesh."""
    return uw.meshing.Annulus(
        radiusOuter=config.outer_radius,
        radiusInner=config.inner_radius,
        cellSize=config.cell_size,
    )


def create_fields(mesh):
    """Create mesh variables for the model."""
    velocity = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    temperature_initial = uw.discretisation.MeshVariable("T0", mesh, 1, degree=3)
    return velocity, pressure, temperature, temperature_initial


def configure_stokes_solver(mesh, velocity, pressure, temperature, config: ModelConfig):
    """Configure the Stokes solver."""
    x_coord, y_coord = mesh.X
    radius = sympy.sqrt(x_coord**2 + y_coord**2)
    radial_unit = mesh.X / radius

    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = config.viscosity
    stokes.bodyforce = radial_unit * temperature.sym[0]

    return stokes


def configure_temperature_solver(mesh, velocity, temperature, config: ModelConfig):
    """Configure the advection-diffusion solver."""
    adv_diff = uw.systems.AdvDiffusion(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        solver_name="adv_diff",
    )
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = config.diffusivity
    adv_diff.add_dirichlet_bc(config.temperature_inner, "Lower")
    adv_diff.add_dirichlet_bc(config.temperature_outer, "Upper")
    return adv_diff
