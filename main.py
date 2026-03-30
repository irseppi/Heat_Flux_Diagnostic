"""
Run the refactored Underworld3 annulus convection workflow.

Inputs
------
Parameters are defined in config.py.

Outputs
-------
- CSV diagnostics in results/data
- plots in results/figures

Usage
-----
python main.py
"""

import os

from config import ModelConfig
from diagnostics import (
    compute_radial_heat_flux,
    evaluate_boundary_average,
    save_diagnostics_csv,
)
from model import (
    configure_stokes_solver,
    configure_temperature_solver,
    create_fields,
    create_mesh,
)
from plotting import plot_flux_timeseries
from tests import run_basic_checks


def main() -> None:
    """Run the annulus convection workflow."""
    config = ModelConfig()
    run_basic_checks(config)

    os.makedirs(config.results_data_dir, exist_ok=True)
    os.makedirs(config.results_figure_dir, exist_ok=True)

    mesh = create_mesh(config)
    velocity, pressure, temperature, temperature_initial = create_fields(mesh)

    stokes = configure_stokes_solver(
        mesh,
        velocity,
        pressure,
        temperature,
        config,
    )
    adv_diff = configure_temperature_solver(
        mesh,
        velocity,
        temperature,
        config,
    )

    conductive_flux, advective_flux, total_flux = compute_radial_heat_flux(
        mesh,
        temperature,
        velocity,
    )

    elapsed_time = 0.0
    rows = []

    for step in range(config.max_steps):
        stokes.solve(zero_init_guess=False)
        dt = 5.0 * stokes.estimate_dt()
        adv_diff.solve(timestep=dt)
        elapsed_time += dt

        inner_flux = evaluate_boundary_average(
            mesh,
            total_flux,
            config.inner_radius,
        )
        outer_flux = evaluate_boundary_average(
            mesh,
            total_flux,
            config.outer_radius,
        )

        rows.append([step, elapsed_time, dt, inner_flux, outer_flux])

        if step % config.output_interval == 0:
            print(f"Step {step:04d} | time = {elapsed_time:.4e}")

    csv_path = os.path.join(config.results_data_dir, "heat_flux_history.csv")
    save_diagnostics_csv(csv_path, rows)

    figure_path = os.path.join(
        config.results_figure_dir,
        "heat_flux_timeseries.png",
    )
    plot_flux_timeseries(rows, save_path=figure_path)


if __name__ == "__main__":
    main()
