# Refactoring an Underworld3 Thermal Convection Script with Heat-Flux Diagnostics

## Project Summary
This project refactors an Underworld3 thermal convection model into a modular workflow for studying heat transport in a simplified planetary interior. The model solves coupled Stokes flow and thermal advection-diffusion in annulus geometry, and the refactor improves maintainability by separating configuration, model setup, diagnostics, plotting, testing, and execution into distinct components. A key scientific extension of the workflow is the addition of heat-flux diagnostics, which help quantify how efficiently heat is transported through the convecting system.

## What
This project refactors an existing Underworld3 thermal convection workflow into a cleaner, more modular, and more maintainable codebase. The original model represents a simplified shell-like planetary interior using annulus geometry and solves coupled Stokes flow and temperature advection-diffusion. The refactor separates configuration, model construction, diagnostics, plotting, and execution into distinct components, while also introducing heat-flux diagnostics and structured output handling.

## How
The workflow has been reorganized into a multi-file Python project with clearly separated responsibilities:

- `config.py` defines model parameters and runtime configuration.
- `model.py` builds the computational mesh, field variables, and solver setup.
- `diagnostics.py` contains heat-flux and related diagnostic calculations.
- `plotting.py` provides visualization utilities for model output.
- `tests.py` includes lightweight validation checks.
- `main.py` serves as the main execution entry point.

A `ModelConfig` dataclass is used to group model parameters into a single structured object, improving parameter management and reducing ambiguity in function interfaces. The codebase is being rewritten toward a more PEP 8-compliant, documented, and modular design.

## Scientific Scope
This project focuses on one important interior process: convective heat transport. It is not intended to be a fully realistic Earth model. Instead, it provides a simplified computational framework for studying how temperature differences drive flow and how that flow transports heat through an interior shell. This reduced scope keeps the model interpretable and makes it easier to validate, extend, and adapt for future Earth and planetary interior studies.

## Why
The original workflow functioned as a monolithic script, which made extension, testing, interpretation, and reuse more difficult. This refactor improves readability, maintainability, and reproducibility, while also making it easier to add scientifically useful diagnostics.

In particular, heat-flux diagnostics are important because temperature plots alone do not fully describe thermal transport. Measuring radial heat flux provides a more physically meaningful way to evaluate how efficiently heat is transferred through the convecting system and whether the model is approaching steady-state behavior.

## Assignment Option
This project follows **Option #2: Re-write Existing Code**.

## Selected Project Tasks

### Project Task #1
Refactor the original convection script into a modular, documented, and maintainable workflow. This includes improved code organization, clearer file structure, PEP 8-oriented formatting, and separation of distinct responsibilities into reusable functions and modules.

### Project Task #2
Add structured parameter handling and heat-flux diagnostics to the workflow. This includes the implementation of a `ModelConfig` dataclass, configuration-driven parameter input through `config.py`, and diagnostic calculations collected in `diagnostics.py`.

## Additional Technical Improvements
The current refactor also includes several supplementary improvements beyond the two selected project tasks:

- validation checks in `tests.py`
- reproducible diagnostic output through CSV export
- cleaner repository organization for development and extension
- feature-branch-based development workflow

## Repository Structure

- `config.py` — model parameters and runtime configuration
- `model.py` — mesh generation, fields, and solver setup
- `diagnostics.py` — heat-flux diagnostic calculations
- `plotting.py` — plotting and visualization utilities
- `tests.py` — lightweight validation and assertion-based checks
- `main.py` — primary execution script

## Model Use and Extensions
The refactored workflow is designed as a reusable foundation for future geodynamic and planetary interior studies. In its current form, it can be used to analyze simplified thermal convection and heat transport in a shell-like domain. With further development, the same framework could be extended toward more realistic mantle-style applications by adding features such as temperature-dependent viscosity, internal heating, additional transport diagnostics, or more planet-specific parameter choices.

Because the workflow is configuration-driven, it can also be adapted for comparative studies of other terrestrial bodies by changing model parameters such as gravity, thermal expansivity, viscosity, thermal diffusivity, or boundary temperatures.

## How to Run

```bash
python main.py
```
## to run in uw3 environment (inside the underworld3) --
pixi run -e runtime python /Users/saurabhshukla/REPOSITORIES/Heat_Flux_Diagnostic/main.py

## This is the intended entry point for the refactored workflow.

## Heat-Flux Diagnostics

A major scientific addition in this refactor is the implementation of heat-flux diagnostics. These diagnostics are intended to measure how thermal energy is transported across the model boundaries and how that transport evolves through time.

Heat flux is useful because it provides a more direct measure of transport efficiency than temperature visualization alone. In particular, boundary-averaged radial heat flux can be used to:

- evaluate thermal transport behavior
- compare model runs under different parameter choices
- assess whether the system is approaching steady state
- support future Earth and planetary interior interpretation

## Current Status

The project structure, code refactor, configuration handling, diagnostics design, and documentation have been developed. Full end-to-end runtime validation is still in progress.

## Environment Limitation

At present, full execution is blocked by a PETSc shared-library linkage issue in the local Underworld3 environment. The compiled Underworld3 extension module depends on libpetsc.3.24.dylib, but that dynamic library is not currently being resolved from the active Python environment during import. This indicates a runtime dependency mismatch between the local Underworld3 build and the available PETSc installation, rather than a logic error in the refactored convection workflow itself.

## Next Steps

The next step is to resolve the PETSc runtime linkage mismatch in the local Underworld3 environment and then perform full end-to-end execution of the refactored workflow. Once import stability is restored, the remaining work will include validation of the Stokes solve, advection-diffusion update, boundary-condition enforcement, and heat-flux diagnostics, followed by generation of reproducible output files and finalized comparison plots. To improve reproducibility and reduce setup ambiguity, the repository will also be extended with documented conda environment instructions for uw3, a dedicated Python/Jupyter kernel configuration using the same uw3 environment, and clearer environment initialization steps for consistent execution across systems.

## Summary

This project is a modular refactor of an Underworld3 thermal convection workflow with a particular emphasis on maintainability, structured parameter handling, and heat-flux diagnostics. Its main purpose is to provide a clean and extensible framework for studying convective heat transport in simplified planetary interior settings, while also laying the groundwork for future Earth and comparative planetary applications.