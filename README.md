# Refactoring an Underworld3 Thermal Convection Script with Heat-Flux Diagnostics

## Project Overview
This project rewrites a thermal convection workflow in Underworld3 into a cleaner, more modular structure with improved documentation, parameter handling, and heat-flux diagnostics.

## Assignment Option
This project follows **Option #2: Re-write Existing Code**.

## Goals
- improve code organization and readability
- introduce a parameter input system
- add heat-flux diagnostics
- save results in a reproducible format

## Task 1
- PEP 8 formatting
- modular functions
- documentation through docstrings and README
- version control with small commits
- accessible file structure and run instructions

## Task 2
- implemented a 'ModelConfig' dataclass to bundle parameters

## Task 3
- parameter input system through 'config.py'
- testing through assert checks in 'tests.py'
- state saving through CSV diagnostics
- branching development through a feature branch

## Repository Structure
- 'config.py' : model parameters
- 'model.py' : mesh and solver setup
- 'diagnostics.py' : heat-flux calculations
- 'plotting.py' : plotting utilities
- 'tests.py' : validation checks
- 'main.py' : main execution script

## How to Run
'''bash
python main.py

# Note 
1. This project is currently under active development. The codebase has been restructured into a cleaner modular workflow, and heat-flux diagnostics and parameter handling have been added. Full execution and validation are still in progress.

2. The present development environment has an Underworld3/PETSc shared-library linking issue that prevents full execution of the model at this stage. This issue affects runtime validation, but the code refactor, project structure, diagnostics design, and documentation have already been developed.

# Environment issue (not with the code , but with the imports)
At the current stage, full execution is blocked by a PETSc shared-library linkage issue in the local Underworld3 environment. The compiled Underworld3 extension module depends on libpetsc.3.24.dylib, but that dynamic library is not being resolved from the active Python environment during import. This indicates a runtime dependency mismatch between the local Underworld3 build and the available PETSc installation, rather than a logic error in the convection code itself.

# What's next then ??? 
Resolve the PETSc runtime linkage mismatch in the local Underworld3 environment, then perform end-to-end execution of the refactored convection workflow. After import stability is restored, validate the Stokes solve, advection–diffusion update, boundary-condition enforcement, and radial heat-flux diagnostics, then generate reproducible output files and comparison plots for the finalized example run.