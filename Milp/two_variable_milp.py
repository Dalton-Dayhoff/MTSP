import gurobipy as gp
import numpy as np
from scipy.sparse import lil_matrix


if __name__ == "__main__":
    m = gp.Model("NewModel")
    x = m.addVars(2, lb=[-np.inf, 0])

    # Linear Constraints
    A = lil_matrix((3,2), dtype='float')
    A[0,0] = 4
    A[0,1] = -2
    A[1,0] = -2
    A[1,1] = -2
    A[2,0] = 0.5
    A[2,1] = -1
    A = A.toarray()
    b = np.array([0, -30, -3.75])
    z = gp.MVar.fromlist(x.select())
    m.addMConstr(A=A, x=None, b=b, sense='>=', name='Linear')

    # Objective function
    c = lil_matrix((1,2), dtype='float')
    c[0,0] = -2
    c[0,1] = -3
    c = c.toarray()
    c = c[0]
    m.setMObjective(Q=None, c=c, constant=0, xQ_L=z, xQ_R=z, xc=z, sense=gp.GRB.MINIMIZE)

    # Solve
    m.optimize()

    print("Gets Here")


