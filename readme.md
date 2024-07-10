# Solving the Multiple Traveling Salsemen Problem
This code base contains a multitude of solvers and implementations relating to the MTSP.

## Netowrk Flow/ MILP
This is currenlty the only active solver.
This solver use gurobi to solve a Network flow implementation of the MTSP.

# Setup
This project uses both python and rust. This means much of the setup is handled behind the scenes in order to ensure propor communcation betweenthe languages.

## Rust and Python
The python version must of the form 3.12.XX.
The rustup verison must be >= 1.27.1.

## Virtual Environment
Setup a python virtual environment.
```bash
python -m venv .venv
```

## Rust Crates
There are 5 required crates that can all be downloaded from crates.io: nalgebra (0.33), plotters (0.3.5), plotters-backend (0.3.5), rand (0.8.5), and pyo3 (0.22.0). The first four are common amongst all rust projects and are self explanatory. The last crate, pyo3, allows for communication between python and rust.

## Python Packages
There are 3 required python packages (numpy, gurobipy, and matplotlib). The user need not worry about installing these because it is taken care of within the setup function. This is because Rust sometimes loses track of where the packages are within the virtual environment. Meaning it is good to ensure Rust knows where they are before running each time.


# Documentation
The code base is heavilly documented in both rust and python.

## Rust
If one desires, the rust documentation can be compilled into a common form using rustdoc and cargo. Fist go into the Implementation folder
```bash
cd Implementation
```
then run
```bash
cargo doc
```

## Python