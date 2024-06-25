""" This file is to test out gurobi
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.sparse import lil_matrix

if __name__ == "__main__":
    m = gp.Model("NewModel")
    x = m.addVars(2, lb= [2, 1], ub=[4,4]) #type: ignore

    # Create linear constraint
    A = lil_matrix((1,2), dtype='float')
    A[0,0] = 1.
    A[0,1] = -1.
    z = gp.MVar.fromlist(x.select())
    m.addMConstr(A=A, x=z, b= np.array([0]), sense="=", name="linear")

    # Create objective
    Q = lil_matrix((2,2), dtype='float')
    Q[0,0] = 1.
    c = np.array([1., 2.])
    m.setMObjective(Q=Q, c=c, constant=0., xQ_L=z, xQ_R=z, xc=z, sense=GRB.MAXIMIZE)

    # Solve
    m.optimize()

    # Output the value of x
    print("Get's here")
