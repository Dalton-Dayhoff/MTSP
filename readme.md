# Solving the Multiple Traveling Salesmen Problem
This code base contains a multitude of solvers and implementations relating to the MTSP.

## Network Flow/ MILP
This is currently the only active solver.
This solver use gurobi to solve a Network flow implementation of the MTSP.

# Setup
This project uses both python and rust. This means much of the setup is handled behind the scenes in order to ensure proper communication between the languages.

## Rust and Python
The python version must of the form 3.XX.
The rustup version must be >= 1.2X.X

## Python Packages
This code base uses gurobipy for the MILP solver. Using gurobipy to solve any problem that is complicated requires a license. To install this license, first ensure gurobipy is installed using
```bash
pip install gurobipy
```
Then go to https://support.gurobi.com/hc/en-us/articles/360059842732-How-do-I-set-up-a-license-without-installing-the-full-Gurobi-package and download the correct license tools for the operating system. Then using a license from the gurobi optimization website (https://portal.gurobi.com/iam/licenses/request), run the following command from the unzipped license tools folder.
```bash
grbgetkey <license id here>
```
There are 4 required python packages (numpy, scipy, gurobipy, and matplotlib). The user need not worry about installing numpy, scipy, or matplotlib because it is taken care of within the setup function. This is because Rust sometimes loses track of where the packages are within the virtual environment. Meaning it is good to ensure Rust knows where they are before running each time.

## Rust Crates
There are 6 required crates: nalgebra (0.33), plotters (0.3.5), plotters-backend (0.3.5), rand (0.8.5), toml (0.8.4), and pyo3 (0.22.0). The first five are common amongst all rust projects and are self explanatory. The last crate, pyo3, allows for communication between python and rust. One of the more useful traits of rust is that when building the code with cargo, it automatically checks for the required crates and installs any that do not exist.ca

## Virtual Environment
Setup a python virtual environment.
```bash
python -m venv .venv
```

# Running the code
There are currently two major sections of code to run. The milp solver and the k-means clustering section.

## milp
To generate the problem specified in the networkFlow toml and solve it using gurobipy.
```bash
cargo run -- milp run
```
To test gurobipy using the framework and the bounds specified in the networkFlow toml.
```bash
cargo run -- milp test
```

## k-means clustering
To run k-means clustering on the problem specified within the genetic toml (this alludes to the future genetic algorithm implementation).
```bash
cargo run -- cluster
```
# Documentation
The code base is documented in both rust and python.

## Rust
If one desires, the rust documentation can be compiled into a common form using rustdoc and cargo. Fist go into the Implementation folder
```bash
cd Implementation
```
then run
```bash
cargo doc
```
To open the documentation, modify this command
```bash
cargo doc --open
```

## Python