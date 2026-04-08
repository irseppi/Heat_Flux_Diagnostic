"""
Plotting utilities for the annulus convection workflow.

figures:
1. temperature + velocity
2. transport evolution
3. integrated heat transport
4. normalized boundary heterogeneity
5. radial mean temperature profile
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(save_path):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


def plot_transport_evolution(rows, tail_fraction=0.2, save_path=None):
    """Three-panel transport evolution figure."""
    time = np.array([row["time"] for row in rows])
    q_inner = np.array([row["inner_flux"] for row in rows])
    q_outer = np.array([row["outer_flux"] for row in rows])
    nu_inner = np.array([row["nu_inner"] for row in rows])
    nu_outer = np.array([row["nu_outer"] for row in rows])
    vrms = np.array([row["vrms"] for row in rows])

    n_tail = max(5, int(np.ceil(tail_fraction * len(rows))))
    t0 = time[-n_tail]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(time, q_inner, label="Inner boundary")
    axes[0].plot(time, q_outer, label="Outer boundary")
    axes[0].axvspan(t0, time[-1], alpha=0.15)
    axes[0].set_ylabel("Conductive radial heat flux")
    axes[0].set_title("Transport evolution")
    axes[0].legend()

    axes[1].plot(time, nu_inner, label="Inner boundary Nu")
    axes[1].plot(time, nu_outer, label="Outer boundary Nu")
    axes[1].axvspan(t0, time[-1], alpha=0.15)
    axes[1].set_ylabel("Nusselt number")
    axes[1].legend()

    axes[2].plot(time, vrms)
    axes[2].axvspan(t0, time[-1], alpha=0.15)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("RMS velocity")

    plt.tight_layout()

    _ensure_parent(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_integrated_heat_transport(rows, tail_fraction=0.2, save_path=None):
    """Integrated boundary heat transport figure."""
    time = np.array([row["time"] for row in rows])
    q_int_inner = np.array([row["q_int_inner"] for row in rows])
    q_int_outer = np.array([row["q_int_outer"] for row in rows])
    residual = q_int_outer - q_int_inner

    n_tail = max(5, int(np.ceil(tail_fraction * len(rows))))
    t0 = time[-n_tail]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(time, q_int_inner, label=r"$Q_i = 2\pi r_i \langle q_i \rangle$")
    axes[0].plot(time, q_int_outer, label=r"$Q_o = 2\pi r_o \langle q_o \rangle$")
    axes[0].axvspan(t0, time[-1], alpha=0.15)
    axes[0].set_ylabel("Integrated heat transport")
    axes[0].set_title("Boundary heat budget")
    axes[0].legend()

    axes[1].plot(time, residual)
    axes[1].axvspan(t0, time[-1], alpha=0.15)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$Q_o - Q_i$")

    plt.tight_layout()

    _ensure_parent(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_normalized_boundary_flux(theta_inner, flux_inner_norm, theta_outer, flux_outer_norm, save_path=None):
    """Plot normalized boundary heterogeneity on the same axes."""
    plt.figure(figsize=(8, 4.5))
    plt.plot(theta_inner, flux_inner_norm, label="Inner boundary")
    plt.plot(theta_outer, flux_outer_norm, label="Outer boundary")
    plt.xlabel("Azimuth angle (rad)")
    plt.ylabel(r"$q(\theta) / \langle q \rangle$")
    plt.title("Boundary heat-flux heterogeneity")
    plt.legend()
    plt.tight_layout()

    _ensure_parent(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_radial_temperature_profile(radius, temp_mean, temp_cond, save_path=None):
    """Plot radial mean temperature against conductive reference profile."""
    plt.figure(figsize=(6.5, 5))
    plt.plot(radius, temp_mean, label="Simulation mean")
    plt.plot(radius, temp_cond, label="Conductive reference")
    plt.xlabel("Radius")
    plt.ylabel("Temperature")
    plt.title("Radial mean temperature profile")
    plt.legend()
    plt.tight_layout()

    _ensure_parent(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_temperature_velocity_field(temperature, velocity, save_path=None):
    """
    Plot temperature with sparsified velocity vectors.
    """
    tcoords = temperature.coords
    tvals = temperature.data[:, 0]

    vcoords = velocity.coords
    u = velocity.data[:, 0]
    v = velocity.data[:, 1]

    plt.figure(figsize=(7, 7))
    sc = plt.scatter(tcoords[:, 0], tcoords[:, 1], c=tvals, s=7)

    stride = max(1, len(vcoords) // 180)
    plt.quiver(
        vcoords[::stride, 0],
        vcoords[::stride, 1],
        u[::stride],
        v[::stride],
    )

    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Temperature and velocity field")
    plt.colorbar(sc, label="Temperature")
    plt.tight_layout()

    _ensure_parent(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()
