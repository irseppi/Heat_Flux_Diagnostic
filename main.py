"""
Run the annulus convection workflow with physically consistent diagnostics.

"""

import os

from config import ModelConfig
from diagnostics import (
    bin_angular_profile,
    boundary_tolerance,
    compute_boundary_diagnostics,
    compute_radial_heat_flux,
    compute_radial_temperature_profile,
    compute_rayleigh_number,
    compute_rms_velocity,
    conductive_temperature_profile,
    evaluate_boundary_average,
    extract_boundary_flux_vs_angle,
    normalise_profile,
    save_boundary_flux_csv,
    save_radial_profile_csv,
    save_summary_csv,
    save_transport_csv,
    summarise_tail,
)
from model import (
    configure_stokes_solver,
    configure_temperature_solver,
    create_fields,
    create_mesh,
    initialise_temperature,
)
from plotting import (
    plot_integrated_heat_transport,
    plot_normalized_boundary_flux,
    plot_radial_temperature_profile,
    plot_temperature_velocity_field,
    plot_transport_evolution,
)
from tests import run_basic_checks


def main() -> None:
    """Run the annulus convection workflow."""
    config = ModelConfig()
    run_basic_checks(config)

    os.makedirs(config.results_data_dir, exist_ok=True)
    os.makedirs(config.results_figure_dir, exist_ok=True)

    rayleigh_number = compute_rayleigh_number(config)
    print(f"Planet analogue: {config.earth.planet_name}")
    print(f"Estimated Rayleigh number: {rayleigh_number:.3e}")

    mesh = create_mesh(config)
    velocity, pressure, temperature, temperature_initial = create_fields(mesh)

    initialise_temperature(mesh, temperature, temperature_initial, config)

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

    tol = boundary_tolerance(config)
    elapsed_time = 0.0
    rows = []

    for step in range(config.max_steps):
        stokes.solve(zero_init_guess=True)

        dt_est = adv_diff.estimate_dt()
        dt = min(config.max_dt, max(config.min_dt, config.timestep_safety * dt_est))

        adv_diff.solve(timestep=dt, zero_init_guess=False)
        elapsed_time += dt

        # Use conductive boundary flux for physically consistent boundary diagnostics
        inner_flux = evaluate_boundary_average(
            mesh, conductive_flux, config.inner_radius, tol
        )
        outer_flux = evaluate_boundary_average(
            mesh, conductive_flux, config.outer_radius, tol
        )
        vrms = compute_rms_velocity(velocity)

        nu_inner, nu_outer, q_int_inner, q_int_outer = compute_boundary_diagnostics(
            inner_flux,
            outer_flux,
            config,
        )

        rows.append(
            {
                "step": step,
                "time": elapsed_time,
                "dt": dt,
                "inner_flux": inner_flux,
                "outer_flux": outer_flux,
                "nu_inner": nu_inner,
                "nu_outer": nu_outer,
                "q_int_inner": q_int_inner,
                "q_int_outer": q_int_outer,
                "vrms": vrms,
            }
        )

        if step % config.output_interval == 0:
            print(
                f"Step {step:04d} | time = {elapsed_time:.4e} | "
                f"dt = {dt:.4e} | vrms = {vrms:.4e} | "
                f"Nu_i = {nu_inner:.4f} | Nu_o = {nu_outer:.4f}"
            )

    # Tail-window statistics help diagnose whether the run is still transient
    tail_stats = summarise_tail(rows, config.tail_fraction)

    print("\nTail-window diagnostic summary")
    print(f"  Samples in tail: {tail_stats['n_tail']}")
    print(
        f"  Inner flux  = {tail_stats['inner_flux_mean']:.6e} ± "
        f"{tail_stats['inner_flux_std']:.2e}"
    )
    print(
        f"  Outer flux  = {tail_stats['outer_flux_mean']:.6e} ± "
        f"{tail_stats['outer_flux_std']:.2e}"
    )
    print(
        f"  Nu_inner    = {tail_stats['nu_inner_mean']:.6e} ± "
        f"{tail_stats['nu_inner_std']:.2e}"
    )
    print(
        f"  Nu_outer    = {tail_stats['nu_outer_mean']:.6e} ± "
        f"{tail_stats['nu_outer_std']:.2e}"
    )
    print(
        f"  Q_inner     = {tail_stats['q_int_inner_mean']:.6e} ± "
        f"{tail_stats['q_int_inner_std']:.2e}"
    )
    print(
        f"  Q_outer     = {tail_stats['q_int_outer_mean']:.6e} ± "
        f"{tail_stats['q_int_outer_std']:.2e}"
    )
    print(
        f"  VRMS        = {tail_stats['vrms_mean']:.6e} ± "
        f"{tail_stats['vrms_std']:.2e}"
    )

    save_transport_csv(
        os.path.join(config.results_data_dir, "transport_history.csv"),
        rows,
    )

    save_summary_csv(
        os.path.join(config.results_data_dir, "summary_metrics.csv"),
        config,
        rayleigh_number,
        tail_stats,
    )

    # Boundary heterogeneity profiles
    theta_inner_raw, flux_inner_raw = extract_boundary_flux_vs_angle(
        mesh,
        conductive_flux,
        config.inner_radius,
        tol,
    )
    theta_outer_raw, flux_outer_raw = extract_boundary_flux_vs_angle(
        mesh,
        conductive_flux,
        config.outer_radius,
        tol,
    )

    theta_inner, flux_inner = bin_angular_profile(
        theta_inner_raw,
        flux_inner_raw,
        config.angular_bins,
    )
    theta_outer, flux_outer = bin_angular_profile(
        theta_outer_raw,
        flux_outer_raw,
        config.angular_bins,
    )

    flux_inner_norm = normalise_profile(flux_inner)
    flux_outer_norm = normalise_profile(flux_outer)

    save_boundary_flux_csv(
        os.path.join(config.results_data_dir, "inner_flux_vs_angle.csv"),
        theta_inner,
        flux_inner,
        flux_inner_norm,
    )
    save_boundary_flux_csv(
        os.path.join(config.results_data_dir, "outer_flux_vs_angle.csv"),
        theta_outer,
        flux_outer,
        flux_outer_norm,
    )

    # Radial mean temperature profile
    radius, temp_mean = compute_radial_temperature_profile(
        mesh,
        temperature,
        config.radial_bins,
    )
    temp_cond = conductive_temperature_profile(radius, config)

    save_radial_profile_csv(
        os.path.join(config.results_data_dir, "radial_temperature_profile.csv"),
        radius,
        temp_mean,
        temp_cond,
    )

    # Final figure set: only the five strongest figures
    plot_temperature_velocity_field(
        temperature,
        velocity,
        save_path=os.path.join(config.results_figure_dir, "figure_1_temperature_velocity.png"),
    )
    plot_transport_evolution(
        rows,
        tail_fraction=config.tail_fraction,
        save_path=os.path.join(config.results_figure_dir, "figure_2_transport_evolution.png"),
    )
    plot_integrated_heat_transport(
        rows,
        tail_fraction=config.tail_fraction,
        save_path=os.path.join(config.results_figure_dir, "figure_3_heat_budget.png"),
    )
    plot_normalized_boundary_flux(
        theta_inner,
        flux_inner_norm,
        theta_outer,
        flux_outer_norm,
        save_path=os.path.join(config.results_figure_dir, "figure_4_boundary_heterogeneity.png"),
    )
    plot_radial_temperature_profile(
        radius,
        temp_mean,
        temp_cond,
        save_path=os.path.join(config.results_figure_dir, "figure_5_radial_temperature_profile.png"),
    )


if __name__ == "__main__":
    main()
