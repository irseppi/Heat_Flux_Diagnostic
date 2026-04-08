"""
Model setup utilities for annulus thermal convection in Underworld3.

key note:
- the conductive reference state in an annulus is logarithmic in radius
  rather than linear in radius.
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
        qdegree=config.qdegree,
    )


def create_fields(mesh):
    """Create mesh variables for the model."""
    velocity = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=False)
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=3, continuous=True)
    temperature_initial = uw.discretisation.MeshVariable("T_init", mesh, 1, degree=3, continuous=True)
    return velocity, pressure, temperature, temperature_initial


def initialise_temperature(mesh, temperature, temperature_initial, config: ModelConfig):
    """
    Initialise temperature using the conductive annulus profile plus a
    boundary-compatible perturbation.

    Conductive profile:
        T_cond(r) = T_o + (T_i - T_o) * ln(r_o / r) / ln(r_o / r_i)

    Perturbation:
        A * sin(nθ) * sin(π r')
    which vanishes at both boundaries.
    """
    r, th = mesh.CoordinateSystem.R

    ri = config.inner_radius
    ro = config.outer_radius
    ti = config.temperature_inner
    to = config.temperature_outer

    r_prime = (r - ri) / (ro - ri)
    conductive = to + (ti - to) * sympy.log(ro / r) / sympy.log(ro / ri)

    perturbation = (
        config.perturbation_amplitude
        * (ti - to)
        * sympy.sin(config.perturbation_wavenumber * th)
        * sympy.sin(sympy.pi * r_prime)
    )

    init_t = conductive + perturbation

    with mesh.access(temperature, temperature_initial):
        values = uw.function.evaluate(init_t, temperature.coords).reshape((-1, 1))
        temperature.data[...] = values
        temperature_initial.data[...] = values


def configure_stokes_solver(mesh, velocity, pressure, temperature, config: ModelConfig):
    """
    Configure the Stokes solver:
    - constant viscosity
    - radial thermal buoyancy
    - penalized zero normal velocity on annulus boundaries
    """
    unit_rvec = mesh.CoordinateSystem.unit_e_0
    gamma_n = unit_rvec

    stokes = uw.systems.Stokes(
        mesh,
        velocityField=velocity,
        pressureField=pressure,
    )

    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = config.viscosity

    stokes.bodyforce = temperature.sym[0] * unit_rvec
    stokes.tolerance = 1.0e-4
    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

    penalty = 1.0e6 * config.viscosity
    stokes.add_natural_bc(penalty * gamma_n.dot(velocity.sym) * gamma_n, "Upper")
    stokes.add_natural_bc(penalty * gamma_n.dot(velocity.sym) * gamma_n, "Lower")

    return stokes


def configure_temperature_solver(mesh, velocity, temperature, config: ModelConfig):
    """Configure the advection-diffusion solver."""
    adv_diff = uw.systems.AdvDiffusion(
        mesh,
        u_Field=temperature,
        V_fn=velocity.sym,
        order=2,
    )

    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = config.diffusivity

    adv_diff.add_dirichlet_bc(config.temperature_inner, "Lower")
    adv_diff.add_dirichlet_bc(config.temperature_outer, "Upper")

    adv_diff.petsc_options.setValue("snes_rtol", 1.0e-3)
    adv_diff.petsc_options.setValue("ksp_rtol", 1.0e-4)

    return adv_diff
