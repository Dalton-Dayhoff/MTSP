from typing import List, Tuple
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

class Variables:
    '''
    Class holding all variables needed throughout

    Attributes:
            Problem Definition Variables
        num_tasks: number of tasks
        num_agents: number of agents
        num_rows: number of rows
        num_nodes; number of nodes
        num_edges: number of edges
            Visualiztion Variables
        task_y: List of y coordinates on the graph for the tasks
        max_task_y: the largest y coordinate value of the tasks, used to set the size of the grpah
        depot_x_start: the x coordinate of the starting depots
        depot_x_end: the x coordinate of the ending depots
        agent_start_nodes: list of node indecies for starting depots
        agent_end_nodes: list of node indecies for the ending depots

    '''
    # Problem Definition Variables
    num_tasks: int
    num_agents: int
    num_rows: int
    num_nodes: int
    num_edges: int
    agent_start_nodes: List[int]
    agent_end_nodes: List[int]

    # Visualization Variables
    task_y: List[float]
    max_task_y: float
    depot_y: List[float]
    depot_x_start: float
    depot_x_end: float

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
            return (self.depot_x_end, self.depot_y[node - self.num_nodes + self.num_agents])
        else:
            task = self.get_task_number(node-2) + 1
            row = (node - self.num_agents)//self.num_tasks
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
            task = plt.Circle(( vars.get_x(i+1) , y_pos ), 0.03, color = 'black')
            axes.add_artist( task )

    # Create ending depots
    for agent in range(vars.num_agents):
        agent = plt.Circle((vars.depot_x_end, vars.depot_y[agent]), 0.03, color = 'r')
        axes.add_artist(agent)
    
    # Create Lines
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
            color = 'grey'
            linewidth = 1
            linestyle = '--'
            alpha = 0.1
        else:
            color = 'b'
            linewidth = 4
            linestyle = '-'
            alpha = 1
            from_circle = plt.Circle(( x_f , y_f ), 0.03 , color = 'r')
            axes.add_artist(from_circle)
        line = Line2D([x_f, x_t], [y_f, y_t], color=color, linewidth=linewidth,linestyle=linestyle, alpha = alpha)
        axes.add_line(line)


def solve_basic(costs, inc_mat, vars: Variables) -> Tuple[float]:
    b = np.zeros(vars.num_nodes)
    # There are two agents in this problem, sources are 1 and sinks are -1
    b[0:vars.num_agents] = np.ones(vars.num_agents)
    b[-vars.num_agents:] = np.ones(vars.num_agents)*-1
    # Transpose the incidence matrix
    inc_mat = np.array(inc_mat).T
    
    # Create the problem
    m = gp.Model("Network Flow")
    x = m.addVars(vars.num_edges, lb=0, ub=1)

    z = gp.MVar.fromlist(x.select())
    m.addMConstr(inc_mat, z, "=", b)

    m.setMObjective(Q=None, c=np.array(costs), constant=0, xQ_L=z, xQ_R=z, xc=z, sense=gp.GRB.MINIMIZE)

    m.optimize()

    value = m.ObjVal
    time = m.Runtime

    print(f"The total cost is {m.ObjVal}")
    print("Usage")
    x_i = [v.x for v in m.getVars()]
    

    visualize(x_i, inc_mat, vars)
    plt.show()
    return value, time

def solve_all_constraints(costs, distances, inc_mat, vars: Variables) -> Tuple[float]:
    b = np.zeros(vars.num_nodes)
    # There are two agents in this problem, sources are 1 and sinks are -1
    b[0:vars.num_agents] = np.ones(vars.num_agents)
    b[-vars.num_agents:] = np.ones(vars.num_agents)*-1
    # Transpose the incidence matrix
    inc_mat = np.array(inc_mat).T
    
    # Create the problem
    m = gp.Model("NewModel")
    m.setParam("OutputFlag", 0)
    x = m.addVars(vars.num_edges, vtype=GRB.BINARY)
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
    ### Combine the edge groups variable so that all edges connected to a task are together
    node_edge_groups = [[] for _ in range(vars.num_tasks)]
    for (node_index, group) in enumerate(edge_groups[vars.num_agents:-vars.num_agents]):
        task_number = node_index % vars.num_tasks
        node_edge_groups[task_number].extend(group)
    ## Create matrix
    edge_group_mat = np.zeros_like(inc_mat)
    for (node_index, group) in enumerate(edge_groups):
        if node_index in vars.agent_start_nodes:
            # There are no incoming edges at the starting depots
            continue
        elif node_index in vars.agent_end_nodes:
            # Add every connected edge to the ending depot
            for edge in group:
                edge_group_mat[node_index, edge] = 1
        else:
            # Add all edges connecting to a given task
            task_number = (node_index - vars.num_agents) % vars.num_tasks
            for edge in node_edge_groups[task_number]:
                edge_group_mat[node_index, edge] = 1

    ## Group Constraint
    b_edge_groups = np.ones(len(edge_group_mat))
    m.addMConstr(edge_group_mat, z, "<=", b_edge_groups)

    # Objective
    m.setMObjective(Q=None, c=np.array(costs), constant=0, xQ_L=z, xQ_R=z, xc=z, sense=gp.GRB.MINIMIZE)

    m.optimize()
    
    value = m.ObjVal
    time = m.Runtime

    # Get actual score
    edge_usage = [v.x for v in m.getVars()]
    cost = 0
    for i, (edge) in enumerate(edge_usage):
        if edge > 0:
            cost += distances[i]

    print(f"Optimal Value: {value}")
    # if value > 100000:
    #     visualize(x_i, inc_mat, vars)
    #     plt.show()

    return value, time, cost