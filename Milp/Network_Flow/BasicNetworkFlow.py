import gurobipy as gp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import copy

def get_data():
    file = open("Milp/Network_Flow/Basic_Data.txt", 'r')
    content = file.read()
    content = content.strip(',')
    lines = content.splitlines()
    values = []
    for (i, line) in enumerate(lines):
        data = line.split()     
        if i == 0:
            data = [float(val) for val in data]
        else:
            data = [int(val) for val in data]
        values.append(data)
    costs = values[0]
    incidence_mat = values[1:] # this is currently transposed
    return costs, incidence_mat

if __name__ == "__main__":
    costs, incidence_mat = get_data()
    num_edges = len(costs)
    num_nodes = len(incidence_mat[0])
    b = np.zeros(num_nodes)
    # There are two agents in this problem, sources are 1 and sinks are -1
    b[0:2] = [1,1]
    b[-3:-1] = [-1,-1]
    # Transpose the incidence matrix
    incidence_mat = np.array(incidence_mat).T
    
    # Create the problem
    m = gp.Model("NewModel")
    x = m.addVars(num_edges, lb=0, ub=1)

    z = gp.MVar.fromlist(x.select())
    m.addMConstr(incidence_mat, z, "=", b)

    m.setMObjective(Q=None, c=np.array(costs), constant=0, xQ_L=z, xQ_R=z, xc=z, sense=gp.GRB.MINIMIZE)

    m.optimize()

    print(f"The total cost is {m.ObjVal}")
    print("Usage")
    visualize_inc = copy.deepcopy(incidence_mat)
    visualize_cost = copy.deepcopy(incidence_mat)
    for (j, row) in enumerate(incidence_mat):
        for (i, v) in enumerate(m.getVars()):
            if v.x != 0 and row[i] != 0:
                visualize_inc[j][i] = 1
            else:
                visualize_inc[j][i] = 0
            if row[i] > 0:
                visualize_cost[j][i] = costs[i]
            elif row[i] < 0:
                visualize_cost[j][i] = -costs[i]
            else:
                visualize_cost[j][i] = 0
    plt.subplot(2,1,1)
    plt.imshow(visualize_inc)
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.imshow(visualize_cost)
    plt.colorbar()
    plt.show()
    