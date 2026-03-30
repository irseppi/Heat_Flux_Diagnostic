"""
Plotting utilities for the Underworld3 convection rewrite.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_flux_timeseries(rows, save_path=None):
    """Plot boundary heat flux through time."""
    time = np.array([row[1] for row in rows])
    inner_flux = np.array([row[3] for row in rows])
    outer_flux = np.array([row[4] for row in rows])

    plt.figure(figsize=(7, 4))
    plt.plot(time, inner_flux, label="Inner boundary")
    plt.plot(time, outer_flux, label="Outer boundary")
    plt.xlabel("Time")
    plt.ylabel("Average radial heat flux")
    plt.title("Boundary heat flux evolution")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.close()
