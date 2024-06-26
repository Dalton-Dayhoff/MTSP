from typing import List
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import copy
from dataclasses import dataclass

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
    incidence_mat = values[1:-2] # this is currently transposed
    other_data = values[-1]
    return costs, incidence_mat, other_data

class Variables:
    num_tasks: int
    num_agents: int
    num_rows: int
    num_nodes: int
    num_edges: int
    task_y: List[float]
    max_task_y: float
    depot_y: List[float]
    depot_x_start: float
    depot_x_end: float
    agent_start_nodes: List[int]
    agent_end_nodes: List[int]

    def __init__(self, num_tasks, num_agents, num_rows, num_nodes, num_edges):
        self.num_tasks = num_tasks
        self.num_agents = num_agents
        self.num_rows = num_rows
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.task_y = []
        base_y = 0.1
        for task_ind in range(num_tasks):
            self.task_y.append((task_ind+1)*base_y)
        self.max_task_y = num_tasks*base_y
        agent_base_y = self.max_task_y/(num_agents + 1)
        self.depot_y = []
        for agent_ind in  range(num_agents):
            self.depot_y.append((agent_ind+1)*agent_base_y)
        self.depot_x_start = 0.0
        self.depot_x_end = 0.2*(num_rows + 1)
        self.agent_start_nodes = list(range(num_agents))
        self.agent_end_nodes = list(range(num_nodes))[-num_agents:]


    def get_x(self, row: int) -> float:
        return 0.2*row
    
    def get_task_number(self, node: int) -> int:
        return node % self.num_tasks
    
    def get_pos_of_node(self, node: int) -> (float, float): # type: ignore
        if node in self.agent_start_nodes:
            return (self.depot_x_start, self.depot_y[node])
        elif node in self.agent_end_nodes:
            return (self.depot_x_end, self.depot_y[node - (self.num_rows*self.num_tasks + self.num_agents)])
        else:
            task = self.get_task_number(node-2) + 1
            row = (node - 2)//self.num_tasks
            x_pos = self.get_x(row + 1)
            y_pos = self.task_y[task - 1]
            return (x_pos, y_pos)
        


def visualize(x, inc_mat, vars: Variables):
    figure, axes = plt.subplots()

    axes.set_ylim(0, vars.max_task_y + 0.1)
    axes.set_xlim(0, vars.depot_x_end)
    
    # Create Tasks
    for i in range(vars.num_rows):
        for j in range(vars.num_tasks):
            y_pos = vars.task_y[j]
            task = plt.Circle(( vars.get_x(i+1) , y_pos ), 0.03, color = 'gray')
            axes.add_artist( task )

    # Create ending depots
    for agent in range(vars.num_agents):
        agent = plt.Circle((vars.depot_x_end, vars.depot_y[agent]), 0.03, color = 'r')
        axes.add_artist(agent)
    
    # Create Lines
    lines = []
    for (i, x_i) in enumerate(x):
        from_node = np.where(inc_mat[:,i]==1)[0]
        to_node = np.where(inc_mat[:,i]==-1)[0]
        if len(to_node)==0 or len(from_node)==0:
            continue
        from_node = from_node[0]
        to_node = to_node[0]
        x_f, y_f = vars.get_pos_of_node(from_node)
        x_t, y_t = vars.get_pos_of_node(to_node)
        if x_i == 0:
            color = 'gray'
            linewidth = 1
            linestyle = '--'
        else:
            color = 'b'
            linewidth = 4
            linestyle = '-'
            from_circle = plt.Circle(( x_f , y_f ), 0.03 , color = 'r')
            axes.add_artist(from_circle)
        line = Line2D([x_f, x_t], [y_f, y_t], color=color, linewidth=linewidth,linestyle=linestyle)
        axes.add_line(line)
    plt.show()

def solve_basic(costs, inc_mat, vars: Variables):
    b = np.zeros(vars.num_nodes)
    # There are two agents in this problem, sources are 1 and sinks are -1
    b[0:vars.num_agents] = np.ones(vars.num_agents)
    b[-vars.num_agents:] = np.ones(vars.num_agents)*-1
    # Transpose the incidence matrix
    inc_mat = np.array(inc_mat).T
    
    # Create the problem
    m = gp.Model("NewModel")
    x = m.addVars(vars.num_edges, lb=0, ub=1)

    z = gp.MVar.fromlist(x.select())
    m.addMConstr(inc_mat, z, "=", b)

    m.setMObjective(Q=None, c=np.array(costs), constant=0, xQ_L=z, xQ_R=z, xc=z, sense=gp.GRB.MINIMIZE)

    m.optimize()

    print(f"The total cost is {m.ObjVal}")
    print("Usage")
    x_i = [v.x for v in m.getVars()]

    visualize(x_i, inc_mat, vars)

def solve_all_constraints(costs, inc_mat, vars: Variables):
    b = np.zeros(vars.num_nodes)
    # There are two agents in this problem, sources are 1 and sinks are -1
    b[0:2] = [1,1]
    b[-2:] = [-1,-1]
    # Transpose the incidence matrix
    inc_mat = np.array(inc_mat).T
    
    # Create the problem
    m = gp.Model("NewModel")
    x = m.addVars(num_edges, vtype=GRB.BINARY)

    # Constraints
    z = gp.MVar.fromlist(x.select())
    m.addMConstr(inc_mat, z, "=", b)

    ## Create groups
    ### Edge Groups
    edge_groups = []
    for (node_index, node_inc) in enumerate(inc_mat):
        edge_groups.append([])
        for (edge_index, edge) in enumerate(node_inc):
            if edge == -1: # Create groups of edges that go into a given node
                edge_groups[node_index].append(edge_index)
    ### Combine edge groups into task groups
    node_edge_groups = [[] for _ in range(vars.num_tasks)]
    for (node_index, group) in enumerate(edge_groups[vars.num_agents:-vars.num_agents]):
        task_number = node_index % vars.num_tasks
        node_edge_groups[task_number].extend(group)
    ## Create matrix
    edge_group_mat = np.zeros_like(inc_mat)
    for (node_index, group) in enumerate(edge_groups):
        if node_index < vars.num_agents:
            continue
        elif node_index >= vars.num_nodes - vars.num_agents:
            for edge in group:
                edge_group_mat[node_index, edge] = 1
        else:
            task_number = (node_index - vars.num_agents) % vars.num_tasks
            for edge in node_edge_groups[task_number]:
                edge_group_mat[node_index, edge] = 1

    ## Group Constraint
    b_edge_groups = np.ones(len(edge_group_mat))
    m.addMConstr(edge_group_mat, z, "<=", b_edge_groups)

    # Objective
    m.setMObjective(Q=None, c=np.array(costs), constant=0, xQ_L=z, xQ_R=z, xc=z, sense=gp.GRB.MINIMIZE)

    m.optimize()

    print(f"The total cost is {m.ObjVal}")
    print("Usage")
    x_i = [v.x for v in m.getVars()]

    visualize(x_i, inc_mat, vars)

if __name__ == "__main__":
    costs, incidence_mat, other_data = get_data()
    num_tasks = other_data[0]
    num_agents= other_data[1]
    num_rows = other_data[2]
    num_nodes = other_data[3]
    num_edges = len(costs)
    vars = Variables(num_tasks, num_agents, num_rows, num_nodes, num_edges)
    solve_basic(costs, incidence_mat, vars)
    solve_all_constraints(costs, incidence_mat, vars)
