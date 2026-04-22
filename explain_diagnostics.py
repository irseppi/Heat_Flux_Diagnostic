from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
OUTPUT_FIGURE = Path("quicklooks") / "diagnostics_quicklook.png"


class Colors:
    """ANSI color codes for simple terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header() -> None:
    """Print the script title."""
    print(
        f"{Colors.HEADER}{Colors.BOLD}"
        f"Heat Flux Diagnostic - Diagnostic Explainer"
        f"{Colors.END}\n"
    )


def explain_terms() -> None:
    """Print plain-language explanations of common diagnostics."""
    print(
        f"{Colors.OKCYAN}{Colors.BOLD}"
        f"What these diagnostics mean:"
        f"{Colors.END}\n"
    )

    explanations = {
        "time": (
            "The simulation clock. It shows how far the model has evolved."
        ),
        "step": (
            "The timestep number. Each step advances the model by a small "
            "amount."
        ),
        "dt": (
            "The timestep size. Smaller values usually mean the solver is "
            "taking more cautious steps through simulated time."
        ),
        "vrms": (
            "Root-mean-square velocity. This gives a simple measure of how "
            "strongly the material is moving. Larger values usually mean "
            "more vigorous convection."
        ),
        "Nu_i": (
            "Inner-boundary Nusselt number or heat-flux measure. It helps "
            "describe how efficiently heat is transferred across the inner "
            "boundary compared with pure conduction."
        ),
        "Nu_o": (
            "Outer-boundary Nusselt number or heat-flux measure. It tracks "
            "how efficiently heat is transported across the outer boundary."
        ),
        "heat_flux": (
            "Heat flux describes how much thermal energy is moving through "
            "a boundary or region. It is often more informative than "
            "temperature alone when studying transport."
        ),
    }

    for name, text in explanations.items():
        print(f"{Colors.BOLD}{name}{Colors.END}: {text}\n")

    print(
        f"{Colors.OKBLUE}{Colors.BOLD}"
        f"How to interpret changes:"
        f"{Colors.END}\n"
    )
    print("- Rising vrms often means convection is strengthening.")
    print(
        "- Nearly steady vrms can suggest that the system is moving toward "
        "a more stable pattern."
    )
    print(
        "- Large differences between inner and outer heat-flux measures can "
        "mean the model is still adjusting."
    )
    print(
        "- Very irregular changes may indicate a transient phase or a run "
        "that needs closer inspection.\n"
    )


def find_candidate_files(results_dir: Path) -> List[Path]:
    """Return possible diagnostics files from the results directory."""
    if not results_dir.exists():
        return []

    candidates: List[Path] = []

    for pattern in ("*.csv", "*.txt", "*.dat"):
        candidates.extend(results_dir.rglob(pattern))

    filtered = [
        path
        for path in candidates
        if path.is_file()
        and path.name.lower() not in {"readme.txt", "notes.txt"}
    ]

    filtered.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return filtered


def clean_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    dataframe = dataframe.copy()
    dataframe.columns = [str(col).strip() for col in dataframe.columns]
    return dataframe


def try_read_table(file_path: Path) -> Optional[pd.DataFrame]:
    """Try a few simple readers and return a dataframe if one works."""
    readers = [
        lambda path: pd.read_csv(path),
        lambda path: pd.read_csv(path, sep=r"\s+", engine="python"),
        lambda path: pd.read_table(path),
        lambda path: pd.read_table(path, sep=r"\s+", engine="python"),
    ]

    for reader in readers:
        try:
            dataframe = reader(file_path)
        except Exception:
            continue

        if dataframe is not None and not dataframe.empty:
            if len(dataframe.columns) >= 2:
                return clean_columns(dataframe)

    return None


def find_column(dataframe: pd.DataFrame, options: List[str]) -> Optional[str]:
    """Find a matching column name using exact or partial matching."""
    lower_map = {col.lower(): col for col in dataframe.columns}

    for option in options:
        if option.lower() in lower_map:
            return lower_map[option.lower()]

    for col in dataframe.columns:
        col_lower = col.lower()
        for option in options:
            if option.lower() in col_lower:
                return col

    return None


def detect_plot_columns(
    dataframe: pd.DataFrame,
) -> Tuple[Optional[str], List[str]]:
    """Detect likely time and diagnostic columns for plotting."""
    time_col = find_column(
        dataframe,
        ["time", "model_time", "sim_time", "t"],
    )
    vrms_col = find_column(
        dataframe,
        ["vrms", "v_rms", "velocity_rms"],
    )
    nui_col = find_column(
        dataframe,
        ["Nu_i", "nui", "nu_inner", "inner_nusselt"],
    )
    nuo_col = find_column(
        dataframe,
        ["Nu_o", "nuo", "nu_outer", "outer_nusselt"],
    )
    heat_flux_col = find_column(
        dataframe,
        ["heat_flux", "flux", "q"],
    )

    y_columns = [
        col for col in [vrms_col, nui_col, nuo_col, heat_flux_col]
        if col is not None
    ]

    return time_col, y_columns


def summarise_data(
    dataframe: pd.DataFrame,
    time_col: Optional[str],
    y_columns: List[str],
) -> None:
    """Print a short summary of the available diagnostics."""
    print(
        f"{Colors.OKCYAN}{Colors.BOLD}"
        f"Quick interpretation of available data:"
        f"{Colors.END}\n"
    )

    if time_col is not None:
        try:
            start_time = dataframe[time_col].iloc[0]
            end_time = dataframe[time_col].iloc[-1]
            print(f"- Time range found: {start_time} to {end_time}")
        except Exception:
            print(
                "- A time-like column was found, but its range could not be "
                "read cleanly."
            )
    else:
        print(
            "- No obvious time column was found, so a time-series plot may "
            "not be possible."
        )

    if not y_columns:
        print(
            "- No standard diagnostic columns like vrms, Nu_i, or Nu_o were "
            "detected.\n"
        )
        return

    for col in y_columns:
        try:
            series = pd.to_numeric(
                dataframe[col],
                errors="coerce",
            ).dropna()

            if len(series) < 2:
                print(f"- {col}: not enough numeric data to interpret.")
                continue

            first_value = series.iloc[0]
            last_value = series.iloc[-1]
            change = last_value - first_value

            if abs(change) < 1.0e-12:
                trend = "is nearly unchanged"
            elif change > 0:
                trend = "increases overall"
            else:
                trend = "decreases overall"

            print(
                f"- {col}: starts at {first_value:.4g}, ends at "
                f"{last_value:.4g}, and {trend}."
            )
        except Exception:
            print(f"- {col}: found, but could not summarize numerically.")

    print("")


def make_time_series_plot(
    dataframe: pd.DataFrame,
    time_col: Optional[str],
    y_columns: List[str],
    output_path: Path,
) -> bool:
    """Create a time-series plot if suitable columns are available."""
    if time_col is None or not y_columns:
        return False

    plot_df = dataframe.copy()
    plot_df[time_col] = pd.to_numeric(
        plot_df[time_col],
        errors="coerce",
    )

    for col in y_columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df.dropna(subset=[time_col])

    if plot_df.empty:
        return False

    plt.figure(figsize=(10, 6))

    for col in y_columns:
        valid_data = plot_df[[time_col, col]].dropna()
        if not valid_data.empty:
            plt.plot(valid_data[time_col], valid_data[col], label=col)

    plt.xlabel(time_col)
    plt.ylabel("Diagnostic value")
    plt.title("Diagnostic Summary Through Time")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    return True


def select_summary_metrics(dataframe: pd.DataFrame) -> pd.Series:
    """Select relevant numeric metrics from a summary-style file."""
    first_row = dataframe.iloc[0]
    numeric_values = pd.to_numeric(first_row, errors="coerce").dropna()

    preferred_order = [
        "estimated_rayleigh_number",
        "reference_viscosity_Pa*s",
        "thermal_diffusivity_m2/s",
        "thermal_conductivity_W/mK",
        "thermal_expansivity_1/K",
        "gravity_m/s2",
        "density_kg/m3",
        "surface_temperature_K",
        "cmb_temperature_K",
        "planet_radius_m",
        "core_radius_m",
        "mantle_thickness_m",
        "heat_capacity_J/kgK",
    ]

    selected = []
    for name in preferred_order:
        if name in numeric_values.index:
            selected.append(name)

    if selected:
        return numeric_values[selected]

    return numeric_values


def make_summary_plot(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> bool:
    """Create a summary figure from a one-row parameter table."""
    if dataframe.empty:
        return False

    metric_series = select_summary_metrics(dataframe)

    if metric_series.empty:
        return False

    labels = [name.replace("_", " ") for name in metric_series.index]
    values = metric_series.values

    figure, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )

    ax1.barh(labels, values)
    ax1.set_xscale("log")
    ax1.set_xlabel("Value (log scale)")
    ax1.set_title("Run Summary Parameters")

    planet_name = "Unknown"
    if "planet" in dataframe.columns:
        planet_name = str(dataframe["planet"].iloc[0])

    interpretation_lines = [
        f"Planet analogue: {planet_name}",
        "",
        "Why this figure is useful:",
        "- It gives a quick overview of the",
        "  main physical and model inputs.",
        "- It helps explain the setup",
        "  without reading the raw CSV.",
        "- It acts as a lightweight",
        "  post-processing and QC summary.",
    ]

    ax2.axis("off")
    ax2.text(
        0.0,
        1.0,
        "\n".join(interpretation_lines),
        va="top",
        fontsize=10,
    )

    figure.suptitle("Heat Flux Diagnostic - Quick Run Overview")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return True


def describe_summary_file(dataframe: pd.DataFrame) -> None:
    """Print a simple explanation for summary-style files."""
    print(
        f"{Colors.OKCYAN}{Colors.BOLD}"
        f"Summary-style file detected:"
        f"{Colors.END}\n"
    )
    print(
        "- This file looks like a run summary or parameter table rather than "
        "a time-evolving diagnostics file."
    )
    print(
        "- A summary figure will be created from the numeric values that are "
        "available."
    )
    print(
        "- This makes it easier to explain the model setup and compare runs "
        "without opening the raw output file.\n"
    )


def main() -> None:
    """Run the diagnostic explainer and optional quick plot."""
    print_header()
    explain_terms()

    print(
        f"{Colors.OKBLUE}Looking for diagnostic files in "
        f"'{RESULTS_DIR}'...{Colors.END}"
    )

    files = find_candidate_files(RESULTS_DIR)

    if not files:
        print(
            f"{Colors.WARNING}No candidate results files were found yet."
            f"{Colors.END}"
        )
        print(
            "This is fine. The explanation section above still helps "
            "beginners understand the diagnostics."
        )
        return

    for file_path in files:
        dataframe = try_read_table(file_path)

        if dataframe is None:
            continue

        print(f"\n{Colors.OKGREEN}Using data file:{Colors.END} {file_path}")
        print(
            f"{Colors.OKGREEN}Detected columns:{Colors.END} "
            f"{list(dataframe.columns)}\n"
        )

        time_col, y_columns = detect_plot_columns(dataframe)
        summarise_data(dataframe, time_col, y_columns)

        plot_created = make_time_series_plot(
            dataframe,
            time_col,
            y_columns,
            OUTPUT_FIGURE,
        )

        if plot_created:
            print(
                f"{Colors.OKGREEN}{Colors.BOLD}"
                f"Time-series plot saved successfully:"
                f"{Colors.END} {OUTPUT_FIGURE}"
            )
            return

        describe_summary_file(dataframe)

        summary_plot_created = make_summary_plot(
            dataframe,
            OUTPUT_FIGURE,
        )

        if summary_plot_created:
            print(
                f"{Colors.OKGREEN}{Colors.BOLD}"
                f"Summary figure saved successfully:"
                f"{Colors.END} {OUTPUT_FIGURE}"
            )
        else:
            print(
                f"{Colors.WARNING}A readable table was found, but no figure "
                f"could be generated from it.{Colors.END}"
            )
            print(
                "You may need to save either time-series diagnostics or a "
                "numeric summary table in CSV form."
            )

        return

    print(
        f"{Colors.WARNING}Files were found in results/, but none could be "
        f"read as a simple diagnostics table.{Colors.END}"
    )
    print(
        "If you later save diagnostics as CSV with columns like time, vrms, "
        "Nu_i, and Nu_o, this script should work directly."
    )


if __name__ == "__main__":
    main()
