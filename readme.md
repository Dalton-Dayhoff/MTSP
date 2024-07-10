# Solving the Multiple Traveling Salsemen Problem
This code base contains a multitude of solvers and implementations relating to the MTSP.

## Netowrk Flow/ MILP
This is currenlty the only active solver.
This solver use gurobi to solve a Network flow implementation of the MTSP.

# Setup
The setup for a rust project is relatively simple because cargo automatically downloads the required packages upon building. However, this code base calls python from rust so there are a few setup steps.

## Rust and Python
The python version must of the form 3.12.XX.
The rustup verison must be >= 1.27.1.

## Virtual Environment
Setup a python virtual environment.
```bash
python -m venv .venv
```

## Python Packages
There are 3 required python packages (numpy, gurobipy, and matplotlib). The user need not worry about installing these because it is taken care of within the setup function. This is because Rust sometimes loses track of where the packages are within the virtual environment. Meaning it is good to ensure Rust knows where they are before running each time.