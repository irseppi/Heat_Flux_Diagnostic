# Heat_Flux_Diagnostic

A modular Underworld3 workflow for thermal convection in annulus geometry with heat-flux diagnostics for simplified planetary interior studies.

---

## Overview

This project refactors an Underworld3 thermal convection model into a cleaner, more modular, and more maintainable workflow for studying heat transport in a simplified shell-like planetary interior.

The model solves coupled Stokes flow and thermal advection-diffusion in annulus geometry. In addition to improving code organization, this refactor adds heat-flux diagnostics so that model behaviour can be evaluated using physically meaningful transport metrics rather than temperature visualization alone.

This project is intended as a simplified computational framework for studying convective heat transport. It is not a fully realistic Earth model. Instead, it provides a reusable starting point for future geodynamic and planetary interior applications.

---

## Scientific Motivation

Temperature plots are useful, but they do not fully describe how efficiently a convecting system transports heat.

A key scientific extension in this workflow is the addition of radial heat-flux diagnostics. These diagnostics help quantify thermal transport across model boundaries and track how that transport changes through time.

This makes it possible to:

- evaluate heat transport efficiency
- compare model behaviour under different parameter choices
- assess whether the system is approaching steady-state behaviour
- support future Earth and comparative planetary interior studies

---

## Project Scope

This project focuses on a simplified convection problem in annulus geometry. The goal is not maximum physical realism, but a code structure that is easier to read, test, validate, and extend.

The workflow is designed to support future additions such as:

- temperature-dependent viscosity
- internal heating
- additional transport diagnostics
- more planet-specific parameter choices
- comparative studies across terrestrial bodies

Because the model is configuration-driven, parameters such as gravity, thermal expansivity, viscosity, thermal diffusivity, and boundary temperatures can be modified more easily than in a monolithic script.

---

## What This Refactor Improves

The original workflow functioned as a largely monolithic script, which made extension, testing, interpretation, and reuse more difficult.

This refactor improves:

- readability
- maintainability
- modularity
- reproducibility
- parameter management
- diagnostic output handling

The project is being rewritten toward a more structured, documented, and PEP 8-oriented design.

---

## Repository Structure

The workflow is organized into separate modules with distinct responsibilities:

- `config.py` — model parameters and runtime configuration
- `model.py` — mesh generation, fields, and solver setup
- `diagnostics.py` — heat-flux and related diagnostic calculations
- `plotting.py` — plotting and visualization utilities
- `tests.py` — lightweight validation and assertion-based checks
- `main.py` — primary execution entry point
- `run_tests.py`- runs the project's built-in checks and clearly shows whether everything is working or if any test has failed.
  
A `ModelConfig` dataclass is used to group model parameters into a single structured object, reducing ambiguity in function interfaces and improving parameter handling.

---

## Selected Project Tasks

### Project Task 1
Refactor the original convection script into a modular, documented, and maintainable workflow. This includes improved code organization, clearer file structure, separation of responsibilities, and reusable functions and modules.

### Project Task 2
Add structured parameter handling and heat-flux diagnostics to the workflow. This includes the implementation of a `ModelConfig` dataclass, configuration-driven input through `config.py`, and diagnostic calculations collected in `diagnostics.py`.

---

## Additional Technical Improvements

Beyond the two selected project tasks, the current refactor also includes:

- lightweight validation checks in `tests.py`
- structured and reproducible diagnostic output
- cleaner repository organization for development and extension
- feature-branch-based development workflow
- improved separation between setup, diagnostics, plotting, and execution

---

## Installation and Environment Setup

This project is designed to be run using the **Underworld3 environment**.

### Recommended Repository Layout

Store both repositories in the same parent directory:

```text
REPOSITORIES/
├── underworld3
└── Heat_Flux_Diagnostic

>> This sibling layout keeps execution simple and avoids path confusion.

1. Clone Underworld3
cd /path/to/REPOSITORIES
git clone https://github.com/underworldcode/underworld3.git
cd underworld3
2. Set Up Underworld3

Run:

./uw setup

Underworld3 uses pixi to manage its environments. The ./uw setup command installs and configures the UW3 environment and dependencies.

When prompted, choose the runtime environment.

3. About the UW3 Conda Environment

You may already have, or may choose to create, a local conda environment such as:

conda create -n uw3 python=3.11
conda activate uw3

This can be useful as a local shell environment, but for this project it is not the primary environment manager.

The important point is:

conda can provide a local Python environment
Underworld3 itself is managed through pixi
the most reliable way to run this project is still through:
pixi run -e runtime python ../Heat_Flux_Diagnostic/main.py

So even if you activate a conda environment called uw3, you should still use the UW3 pixi runtime command when running this repository.

4. Clone This Project

Return to the parent directory and clone this repository:

cd ..
git clone https://github.com/ss2098/Heat_Flux_Diagnostic.git

Your directory layout should now look like:

REPOSITORIES/
├── underworld3
└── Heat_Flux_Diagnostic
How to Run

This project should be run from inside the underworld3 repository using the Underworld3 runtime environment.

Recommended Command
cd /path/to/REPOSITORIES/underworld3
pixi run -e runtime python ../Heat_Flux_Diagnostic/main.py
What This Command Means
pixi run runs a command inside a pixi-managed environment
-e runtime selects the Underworld3 environment named runtime
python ../Heat_Flux_Diagnostic/main.py launches this project using the correct Python, Underworld3, PETSc, and dependency stack

This is the intended execution path for this repository.

## Important Note

Do not assume that python main.py will work from an arbitrary environment.

The safest and recommended approach is always:

pixi run -e runtime python ../Heat_Flux_Diagnostic/main.py
Optional Underworld3 Checks

From inside the underworld3 directory, these commands may be useful:

./uw doctor
./uw build
./uw test

These help verify that the Underworld3 environment is set up correctly before running this project.

Expected Runtime Output

When the model runs successfully, terminal output may include:

planet analogue
estimated Rayleigh number
timestep number
simulation time
timestep size
vrms
inner and outer Nusselt numbers

Example-style output:

Planet analogue: Earth
Estimated Rayleigh number: ...
Step 0000 | time = ... | dt = ... | vrms = ...
Step 0025 | time = ... | dt = ... | vrms = ... | Nu_i = ... | Nu_o = ...
Heat-Flux Diagnostics

A major scientific addition in this refactor is the implementation of heat-flux diagnostics.

These diagnostics are intended to measure how thermal energy is transported across the model boundaries and how that transport evolves through time.

Heat flux is useful because it provides a more direct measure of transport efficiency than temperature visualization alone. In particular, boundary-averaged radial heat flux can be used to:

evaluate thermal transport behaviour
compare model runs under different parameter choices
assess whether the system is approaching steady state
support future planetary interior interpretation
Current Capabilities

The current workflow includes:

modular model setup
structured parameter handling
annulus-domain thermal convection workflow
heat-flux diagnostics
runtime logging of key diagnostic quantities
plotting and visualization utilities
lightweight internal validation checks

The code has been run successfully in the Underworld3 runtime environment during development.

Troubleshooting
Underworld3 does not import

From inside the underworld3 repository, try:

./uw doctor
./uw build

If needed, rerun:

./uw setup
Wrong Python or wrong environment

Always run the project through the Underworld3 runtime environment:

pixi run -e runtime python ../Heat_Flux_Diagnostic/main.py
API mismatch or solver argument errors

If local Underworld3 source changes or versions are out of sync, rebuild Underworld3:

./uw build

Then rerun the project.

Path problems

If the repositories are not side-by-side, replace the relative path with the full path to main.py.

Example:

pixi run -e runtime python /full/path/to/Heat_Flux_Diagnostic/main.py
Model Use and Extensions

The refactored workflow is designed as a reusable foundation for future geodynamic and planetary interior studies.

In its current form, it can be used to analyze simplified thermal convection and heat transport in a shell-like domain.

With further development, this framework could be extended toward more realistic mantle-style applications by adding features such as:

temperature-dependent viscosity
internal heating
additional transport diagnostics
more advanced rheologies
planet-specific parameter sets

Because the workflow is configuration-driven, it is also suitable for comparative studies of other terrestrial bodies through parameter changes alone.

Summary

This project is a modular refactor of an Underworld3 thermal convection workflow with a particular emphasis on:

maintainability
structured parameter handling
modular code design
heat-flux diagnostics
reproducible execution through the Underworld3 runtime environment

Its main purpose is to provide a clean and extensible framework for studying convective heat transport in simplified planetary interior settings while laying the groundwork for future Earth and comparative planetary applications.
