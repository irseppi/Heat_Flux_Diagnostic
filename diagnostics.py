"""
Diagnostics for radial heat flux in annulus convection.
"""

import csv
import os

import numpy as np
import sympy
import underworld3 as uw


def compute_radial_heat_flux(mesh, temperature, velocity):
    """Compute conductive, advective, and total radial heat flux."""
    x_coord, y_coord = mesh.X
    radius = sympy.sqrt(x_coord**2 + y_coord**2)
    radial_unit = mesh.X / radius

    grad_t = temperature.sym.gradient()
    conductive_flux = -(grad_t.dot(radial_unit))
    advective_flux = temperature.sym[0] * (velocity.sym.dot(radial_unit))
    total_flux = conductive_flux + advective_flux

    return conductive_flux, advective_flux, total_flux


def evaluate_boundary_average(mesh, flux_expression, boundary_radius, tolerance=0.03):
    """Average a flux expression near a specified boundary radius."""
    coords = mesh.data
    radius = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    mask = np.abs(radius - boundary_radius) < tolerance

    if not np.any(mask):
        return np.nan

    values = uw.function.evaluate(flux_expression, coords[mask])
    return float(np.mean(values))


def save_diagnostics_csv(output_path, rows):
    """Save diagnostic rows to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time", "dt", "inner_flux", "outer_flux"])
        writer.writerows(rows)
