"""
Diagnostics for annulus convection.

Key concepts:
- conductive boundary flux is used for boundary diagnostics
- conductive reference solution is annulus-correct
- Nusselt number is normalized by conductive flux at each radius
- integrated heat transport Q = 2π r <q_r>
- angular flux heterogeneity is binned and normalized
- radial mean temperature profile is exported
"""

import csv
import os

import numpy as np
import sympy
import underworld3 as uw

from config import ModelConfig


def compute_rayleigh_number(config: ModelConfig) -> float:
    """
    Earth-interpretation Rayleigh number:
        Ra = g * rho * alpha * DeltaT * H^3 / (eta * kappa)
    """
    earth = config.earth
    h = earth.mantle_thickness_m
    delta_t = earth.delta_temperature_K

    return (
        earth.gravity_m_s2
        * earth.density_kg_m3
        * earth.thermal_expansivity_1_K
        * delta_t
        * h**3
        / (earth.reference_viscosity_Pa_s * earth.thermal_diffusivity_m2_s)
    )


def compute_radial_heat_flux(mesh, temperature, velocity):
    """
    Compute radial conductive, advective, and total heat flux:
        q_cond = -∇T · e_r
        q_adv  = T (u · e_r)
        q_tot  = q_cond + q_adv

    For report-grade boundary diagnostics, use q_cond at the boundary.
    """
    x_coord, y_coord = mesh.X
    radius = sympy.sqrt(x_coord**2 + y_coord**2)
    radial_unit = mesh.X / radius

    temp_scalar = temperature.sym[0]
    grad_t = sympy.Matrix(
        [
            sympy.diff(temp_scalar, x_coord),
            sympy.diff(temp_scalar, y_coord),
        ]
    )

    conductive_flux = -(grad_t.dot(radial_unit))
    advective_flux = temp_scalar * (velocity.sym.dot(radial_unit))
    total_flux = conductive_flux + advective_flux

    return conductive_flux, advective_flux, total_flux


def boundary_tolerance(config: ModelConfig) -> float:
    """Boundary sampling tolerance tied to mesh spacing."""
    return config.boundary_tolerance_factor * config.cell_size


def evaluate_boundary_average(mesh, flux_expression, boundary_radius, tolerance):
    """Area-average a flux expression near a specified boundary radius."""
    coords = mesh.data
    radius = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    mask = np.abs(radius - boundary_radius) < tolerance

    if not np.any(mask):
        return np.nan

    values = uw.function.evaluate(flux_expression, coords[mask]).reshape(-1)
    return float(np.mean(values))


def extract_boundary_flux_vs_angle(mesh, flux_expression, boundary_radius, tolerance):
    """Extract flux near a boundary as a function of azimuth."""
    coords = mesh.data
    radius = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    mask = np.abs(radius - boundary_radius) < tolerance

    if not np.any(mask):
        return np.array([]), np.array([])

    boundary_coords = coords[mask]
    theta = np.mod(np.arctan2(boundary_coords[:, 1], boundary_coords[:, 0]), 2.0 * np.pi)
    values = uw.function.evaluate(flux_expression, boundary_coords).reshape(-1)

    order = np.argsort(theta)
    return theta[order], values[order]


def bin_angular_profile(theta, flux, n_bins):
    """Bin an angular profile into uniform azimuth bins."""
    if len(theta) == 0:
        return np.array([]), np.array([])

    edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (theta >= edges[i]) & (theta < edges[i + 1])
        if np.any(mask):
            binned[i] = np.mean(flux[mask])

    # Fill gaps by linear interpolation if needed
    valid = np.isfinite(binned)
    if np.count_nonzero(valid) >= 2:
        binned = np.interp(centres, centres[valid], binned[valid])
    elif np.count_nonzero(valid) == 1:
        binned[:] = binned[valid][0]

    return centres, binned


def compute_rms_velocity(velocity):
    """Compute RMS velocity from the current velocity field."""
    speed_sq = velocity.data[:, 0] ** 2 + velocity.data[:, 1] ** 2
    return float(np.sqrt(np.mean(speed_sq)))


def conductive_reference_flux(radius, config: ModelConfig) -> float:
    """
    Conductive annulus reference flux:
        T_cond(r) = (T_i - T_o) * ln(r_o / r) / ln(r_o / r_i) + T_o
        q_cond(r) = -dT/dr = (T_i - T_o) / (r ln(r_o / r_i))
    """
    delta_t = config.temperature_inner - config.temperature_outer
    return delta_t / (radius * np.log(config.outer_radius / config.inner_radius))


def integrated_heat_transport(avg_flux, radius):
    """
    Integrated outward heat transport per unit out-of-plane thickness:
        Q = 2π r <q_r>
    """
    return 2.0 * np.pi * radius * avg_flux


def compute_boundary_diagnostics(inner_flux, outer_flux, config: ModelConfig):
    """Compute annulus-correct Nusselt numbers and integrated transport."""
    q_cond_inner = conductive_reference_flux(config.inner_radius, config)
    q_cond_outer = conductive_reference_flux(config.outer_radius, config)

    nu_inner = inner_flux / q_cond_inner
    nu_outer = outer_flux / q_cond_outer

    q_int_inner = integrated_heat_transport(inner_flux, config.inner_radius)
    q_int_outer = integrated_heat_transport(outer_flux, config.outer_radius)

    return nu_inner, nu_outer, q_int_inner, q_int_outer


def normalise_profile(flux):
    """Return profile normalized by its mean."""
    mean_flux = np.nanmean(flux)
    if np.isclose(mean_flux, 0.0):
        return np.full_like(flux, np.nan)
    return flux / mean_flux


def compute_radial_temperature_profile(mesh, temperature, n_bins):
    """Compute azimuthally averaged temperature as a function of radius."""
    coords = temperature.coords
    temp = temperature.data[:, 0]
    radius = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

    edges = np.linspace(radius.min(), radius.max(), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    t_mean = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (radius >= edges[i]) & (radius < edges[i + 1])
        if np.any(mask):
            t_mean[i] = np.mean(temp[mask])

    valid = np.isfinite(t_mean)
    if np.count_nonzero(valid) >= 2:
        t_mean = np.interp(centres, centres[valid], t_mean[valid])

    return centres, t_mean


def conductive_temperature_profile(radius, config: ModelConfig):
    """Annulus conductive temperature profile."""
    ri = config.inner_radius
    ro = config.outer_radius
    ti = config.temperature_inner
    to = config.temperature_outer
    return to + (ti - to) * np.log(ro / radius) / np.log(ro / ri)


def summarise_tail(rows, tail_fraction):
    """
    Summarize the last part of the run to diagnose whether the system is
    approaching statistical steady state.
    """
    n = len(rows)
    n_tail = max(5, int(np.ceil(tail_fraction * n)))
    tail = rows[-n_tail:]

    def mean_std(key):
        values = np.array([row[key] for row in tail], dtype=float)
        return float(np.mean(values)), float(np.std(values))

    return {
        "n_tail": n_tail,
        "inner_flux_mean": mean_std("inner_flux")[0],
        "inner_flux_std": mean_std("inner_flux")[1],
        "outer_flux_mean": mean_std("outer_flux")[0],
        "outer_flux_std": mean_std("outer_flux")[1],
        "nu_inner_mean": mean_std("nu_inner")[0],
        "nu_inner_std": mean_std("nu_inner")[1],
        "nu_outer_mean": mean_std("nu_outer")[0],
        "nu_outer_std": mean_std("nu_outer")[1],
        "q_int_inner_mean": mean_std("q_int_inner")[0],
        "q_int_inner_std": mean_std("q_int_inner")[1],
        "q_int_outer_mean": mean_std("q_int_outer")[0],
        "q_int_outer_std": mean_std("q_int_outer")[1],
        "vrms_mean": mean_std("vrms")[0],
        "vrms_std": mean_std("vrms")[1],
    }


def save_transport_csv(output_path, rows):
    """Save time-dependent transport diagnostics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "step",
        "time",
        "dt",
        "inner_flux",
        "outer_flux",
        "nu_inner",
        "nu_outer",
        "q_int_inner",
        "q_int_outer",
        "vrms",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_boundary_flux_csv(output_path, theta, flux, flux_norm):
    """Save binned boundary heat-flux profile."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["theta_rad", "flux", "flux_normalised"])
        for angle, value, norm_value in zip(theta, flux, flux_norm):
            writer.writerow([angle, value, norm_value])


def save_radial_profile_csv(output_path, radius, temp_mean, temp_cond):
    """Save radial mean temperature profile."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["radius", "temperature_mean", "temperature_conductive"])
        for r_val, t_val, t_cond in zip(radius, temp_mean, temp_cond):
            writer.writerow([r_val, t_val, t_cond])


def save_summary_csv(output_path, config: ModelConfig, rayleigh_number: float, tail_stats: dict):
    """Save configuration metadata and tail-window statistics."""
    earth = config.earth
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = [
        ["planet", earth.planet_name],
        ["planet_radius_m", earth.planet_radius_m],
        ["core_radius_m", earth.core_radius_m],
        ["mantle_thickness_m", earth.mantle_thickness_m],
        ["gravity_m_s2", earth.gravity_m_s2],
        ["surface_temperature_K", earth.surface_temperature_K],
        ["cmb_temperature_K", earth.cmb_temperature_K],
        ["density_kg_m3", earth.density_kg_m3],
        ["thermal_expansivity_1_K", earth.thermal_expansivity_1_K],
        ["thermal_diffusivity_m2_s", earth.thermal_diffusivity_m2_s],
        ["thermal_conductivity_W_mK", earth.thermal_conductivity_W_mK],
        ["heat_capacity_J_kgK", earth.heat_capacity_J_kgK],
        ["reference_viscosity_Pa_s", earth.reference_viscosity_Pa_s],
        ["estimated_rayleigh_number", rayleigh_number],
        ["tail_fraction", config.tail_fraction],
    ]

    for key, value in tail_stats.items():
        rows.append([key, value])

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
