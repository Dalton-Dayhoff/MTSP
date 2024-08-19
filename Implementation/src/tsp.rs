use rand;
use plotters::prelude::*;
use core::{f64, task};
use std::{io, fs, time::Duration, path::Path};
use toml::Value;
use initialization::{k_clustering, k_clustering_no_agents};
use std::time::Instant;

use crate::initialization;

#[derive(Debug)]
/// A task, aka a city
pub(crate)struct Task {
    /// All tasks must have a location
    pub(crate)location: (f64, f64),
}

impl Task {
    /// Calculates the distance between the current task and any other point
    /// 
    /// let dist  = task.calc_distance(other_task.location)
    /// let dist  = task.calc_distance(agent.start)
    pub(crate)fn calc_distance(&self, next_task: (f64,f64)) -> f64{
        let x_dist: f64 = self.location.0 - next_task.0;
        let y_dist: f64 = self.location.1 - next_task.1;
        ((x_dist.powf(2.0) + y_dist.powf(2.0)) as f64).sqrt()
    }   
}
/// Allows a task to be cloned
impl Clone for Task{
    fn clone(&self) -> Self {
        Self { location: self.location.clone() }
    }
}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.location == other.location
    }
}

#[derive(Debug)]
/// An agent collects tasks
pub(crate)struct Agent {
    /// The depot is the starting and ending locaiton for the agent
    pub(crate)depot_location: (f64, f64),
    /// This is used by planners when trying to create a path through the tasks
    pub(crate)current: (f64, f64),
    /// A tour is the ordered list of tasks the agent collects
    pub(crate)tour: Vec<Task>,
    }
impl Agent {
    /// Calculates the distance between the current task and any other point
    /// 
    /// let dist  = agent.calc_distance(task.location)
    /// let dist  = agent.calc_distance(other_agent.start)
    pub(crate)fn calc_distance(&self, next_task: (f64,f64)) -> f64{
        let x_dist: f64 = self.depot_location.0 - next_task.0;
        let y_dist: f64 = self.depot_location.1 - next_task.1;
        ((x_dist.powf(2.0) + y_dist.powf(2.0)) as f64).sqrt()
    }   
}

/// Allows an agent to be cloned
impl Clone for Agent{
    fn clone(&self) -> Self {
        Self {
            depot_location: self.depot_location.clone(),
            current: self.current.clone(), 
            tour: self.tour.clone(),
        }
    }
}
#[derive(Debug, Clone)]
/// A multiple traveling salesmen problem
pub(crate)struct Tsp {
    /// List of all tasks
    pub(crate) tasks: Vec<Task>,
    /// List of all agents
    pub(crate) agents: Vec<Agent>,
    /// Used in generation to place all the tasks and depots
    pub(crate) world_size: (f64, f64),
    /// List of the each agents tours
    /// A tour is represented by the list of the lengths of all connections
    pub(crate) tours: Vec<Vec<f64>>,
    /// The list of total tour lengths for each agent
    pub(crate) total_distances: Vec<f64>,
    // Optional list of tasks that are clustrered together
    groups: Option<Vec<Vec<Task>>>,
    //
    num_backbone_per_group: Option<usize>,
    //
    current_backbone_group: Option<usize>
}

impl Tsp {

    fn get_task_index(&self, node_index: usize) -> usize {
        let num_tasks = self.tasks.len();
        let num_agents = self.agents.len();
        let mut num_backbone = 0;
        if !self.current_backbone_group.is_none(){
            num_backbone = self.num_backbone_per_group.unwrap() * self.current_backbone_group.unwrap();
        }
        let mut task_number = node_index - num_agents - num_backbone;
        task_number %= num_tasks;
        return task_number;
    }

    fn in_same_group(&self, to_node: usize, from_node: usize) -> bool{
        let to_task_ind = self.get_task_index(to_node);
        let from_task_ind = self.get_task_index(from_node);
        for tasks in self.groups.as_ref().unwrap(){
            let to_bool = tasks.contains(&self.tasks[to_task_ind]);
            let from_bool = tasks.contains(&self.tasks[from_task_ind]);
            match (to_bool, from_bool) {
                (true, true) => return true,
                (false, true) => return false,
                (true, false) => return false,
                (false, false) => continue
            }
        }
        return false;
    }
    
    fn get_backbone_node_index(&self, from_node: usize) -> usize{
        let from_task_ind = self.get_task_index(from_node);
        for (i, tasks) in self.groups.as_ref().unwrap().iter().enumerate(){
            if tasks.contains(&self.tasks[from_task_ind]){
                return i;
            }
        }
        return 0;
    }

    /// Finds the total tour length of each tour
    /// 
    /// Tsp.total_distances = self.calc_all_distance();
    pub(crate) fn calc_all_distance(&self) -> Vec<f64> {
        let mut distances: Vec<f64> = Vec::new();
        for i in 0..self.tours.len(){
            let tour = &self.tours[i];
            let start_dist = self.agents[i].tour[0].calc_distance(self.agents[i].depot_location);
            let end_dist = self.agents[i].tour[tour.len() - 1].calc_distance(self.agents[i].depot_location);
            let mid_dist: f64 = tour.iter().sum();
            distances.push(mid_dist + start_dist + end_dist);
        }
        distances
    }

    /// Creates a representation of tours by finding the distances of each connection in the tour
    /// 
    /// Tsp.tours = self.calc_tours();
    pub(crate) fn calc_tours(&self) -> Vec<Vec<f64>> {
        let mut tours: Vec<Vec<f64>> = Vec::new();
        let mut tour: Vec<f64> = Vec::new();
        for agent in &self.agents{
            for i in 0..(agent.tour.len()-1){
                tour.push(agent.tour[i].calc_distance(agent.tour[i+1].location));
            }
            tours.push(tour);
            tour = [].to_vec();
        }
        tours
    }

    /// Draws the TSP or mTSP
    /// 
    /// let _ = Tsp.draw_solution("MTSP".to_string());
    fn draw_solution(&self, name: String) -> Result<(), Box<dyn std::error::Error>>{
        // Create name of images
        let name = "Images/".to_owned() + &name + ".svg";

        
        // Create background
        let root = SVGBackend::new(&name, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        // Create plotting area
        let mut chart = ChartBuilder::on(&root)
            .caption("Scatter Plot", ("sans-serif", 40).into_font())
            .build_cartesian_2d(0.0..self.world_size.0, 0.0..self.world_size.1)?;

        // Get all colors to be used
        let num_colors = self.agents.len();
        let mut colors = Vec::with_capacity(num_colors);
        for i in 0..num_colors {
            let red = (i * 33) % 256;
            let green = (i * 57) % 256;
            let blue = (i * 43) % 256;
            colors.push(RGBColor(red as u8, green as u8, blue as u8));
    }
        // Draw tasks
        chart.draw_series(
            self.tasks.iter().map(|task| Circle::new((task.location.0, task.location.1), 5, RGBColor(0, 0, 255))),
    )?;

        // Draw agents
        chart.draw_series(
            self.agents.iter().zip(colors.iter()).map(|(agent, color)| {
                TriangleMarker::new((agent.depot_location.0, agent.depot_location.1), 5, color.clone())
            }),
    )?;
        // Create another representation of tours
        // Each tour is a list of each location the agent visits
        let mut all_tours: Vec<Vec<(f64,f64)>> = Vec::new();
        for agent in &self.agents{
            let mut tour: Vec<(f64,f64)> = agent.tour.iter().map(|task| (task.location.0, task.location.1)).collect();
            tour.insert(0, agent.depot_location);
            tour.push(agent.depot_location);
            all_tours.push(tour)
        }

        // Plot connections for each tour
        for i in 0..all_tours.len(){
            let tour = all_tours[i].clone();
            let color = colors[i].clone();
            match chart.draw_series(LineSeries::new(tour, color)) {
                Ok(_) => (),
                Err(err) => eprintln!("Error drawing series: {}", err),
            }
        }
        Ok(())
    }

    /// Creates the network flow formulation of the mTSP
    ///     sparse matrix representation
    /// 
    /// * 'columns_of_graph' - The total number of columns
    /// * 'early_ender' - The extra multiplyer to adjust the costs of an agents route ending earlier in the graph
    ///     This means an agent collects less tasks than the other agent(s)
    pub fn to_incidence_backbone(
        mut self,
        columns_of_graph: usize,
        early_ender: usize
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>, Vec<i64>, Vec<usize>){
        let nodes_in_graph = self.agents.len()*2 + (columns_of_graph - 2)*self.tasks.len();
        // Define edges
        let mut edge_costs: Vec<f64>  = Vec::new();
        let mut distances: Vec<f64> = Vec::new();
        let mut rows: Vec<usize> = Vec::new();
        let mut columns: Vec<usize> = Vec::new();
        let mut values: Vec<i64> = Vec::new();
        let mut edge: usize = 0;

        // Create groups
        self.groups = Some(k_clustering_no_agents(&self));
        self.num_backbone_per_group = Some(self.groups.clone().unwrap().len());
        let group_lengths: Vec<usize> = self.groups.clone().unwrap().iter().map(|task_group| task_group.len()).collect();
        let max_group_size = group_lengths.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        // first set of edges is from salesmen 1 to all the cities
        let mut to_node = self.agents.len();
        let mut from_node = 0;
        for agent in &self.agents{
            for task in &self.tasks{
                // Get cost
                let task_location = task.location;
                let dist = agent.calc_distance(task_location);
                edge_costs.push(dist);
                distances.push(dist);

                // Incidence matrix
                columns.push(edge);
                columns.push(edge);
                rows.push(to_node);
                rows.push(from_node);
                values.push(-1);
                values.push(1);

                edge += 1;
                to_node += 1;
            }
            from_node += 1;
            to_node = self.agents.len();
        }
        to_node = self.agents.len() + self.tasks.len();
        let mut backbone_group = 1;
        let mut backbone_nodes_idx_start = to_node;
        to_node += self.num_backbone_per_group.unwrap();
        // This iterates over all the edges that directly connect cities together and cities the ending depots
        for i in 0..columns_of_graph - 2{
            for task in &self.tasks{
                for next_task in &self.tasks{
                    if i == columns_of_graph - 3{
                        break;
                    }
                    if i != (backbone_group - 1) * max_group_size && !self.in_same_group(to_node, from_node){
                        // This checks to see if there exists a backbone node here
                        continue;
                    }
                    else if !self.in_same_group(to_node, from_node){
                        let backbone_index = self.get_backbone_node_index(from_node);
                        let to_node_backbone = backbone_nodes_idx_start + backbone_index;
                        let dist = 0.0;
                        edge_costs.push(dist);
                        distances.push(dist);
                        columns.push(edge);
                        columns.push(edge);
                        rows.push(to_node_backbone);
                        rows.push(from_node);
                        values.push(-1);
                        values.push(1);
                    } else{
                        // Cost
                        let dist = task.calc_distance(next_task.location);
                        edge_costs.push(dist);
                        distances.push(dist);
    
                        // Incidence 
                        // Each column has two values, 1 and -1
                        columns.push(edge);
                        columns.push(edge);
                        rows.push(to_node);
                        rows.push(from_node);
                        values.push(-1);
                        values.push(1);
                    }
                    to_node += 1;
                    edge += 1;
                }
                to_node = self.agents.len() + (i+1)*self.tasks.len() + backbone_group*self.num_backbone_per_group.unwrap();
                from_node += 1;
            }
            if i > backbone_group*max_group_size{
                backbone_group += 1;
            }
            if i+1 == (backbone_group - 1)*max_group_size{
                let new_backbone_nodes_idx_start = to_node + 1;
                // Connect the backbone nodes
                for j in backbone_nodes_idx_start..(backbone_nodes_idx_start + self.num_backbone_per_group.unwrap()){
                    for k in new_backbone_nodes_idx_start..(new_backbone_nodes_idx_start + self.num_backbone_per_group.unwrap()) {
                        let dist = 0.0;
                        edge_costs.push(dist);
                        distances.push(dist);

                        columns.push(edge);
                        columns.push(edge);
                        let from_backbone = self.get_backbone_node_index(j);
                        let to_backbone = self.get_backbone_node_index(i);
                        if from_backbone == to_backbone{
                            continue;
                        }
                        rows.push(j);
                        rows.push(k);
                        values.push(-1);
                        values.push(1);
                        edge += 1;
                    }
                    // Connect previous backbone nodes to their groups
                    to_node = self.agents.len() + (i+2)*self.tasks.len() + backbone_group*self.num_backbone_per_group.unwrap();
                    for _ in &self.tasks{
                        let backbone_idx = self.get_backbone_node_index(to_node);
                        let backbone_node_idx = j - backbone_nodes_idx_start;
                        if backbone_idx == backbone_node_idx{
                            let dist = 0.0;
                            edge_costs.push(dist);
                            distances.push(dist);
                            
                            columns.push(edge);
                            columns.push(edge);
                            rows.push(j);
                            rows.push(to_node);
                            values.push(-1);
                            values.push(1);
                            edge += 1;
                        }
                        to_node += 1;
                    }
                }
            }
            // Set the node the edge is going to
            to_node = self.agents.len() + (i+2)*self.tasks.len() + backbone_group*self.num_backbone_per_group.unwrap();
        }
        let other_data = vec![self.tasks.len(), self.agents.len(), columns_of_graph - 2, nodes_in_graph, edge_costs.len()];  
        return (distances, edge_costs, rows, columns, values, other_data);
    }

    /// Creates the network flow formulation of the mTSP
    ///     sparse matrix representation
    /// 
    /// * 'columns_of_graph' - The total number of columns
    /// * 'early_ender' - The extra multiplyer to adjust the costs of an agents route ending earlier in the graph
    ///     This means an agent collects less tasks than the other agent(s)
    pub fn to_incidence_simplified(
        self,
        columns_of_graph: usize,
        early_ender: usize
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>, Vec<i64>, Vec<usize>){
        let nodes_in_graph = self.agents.len()*2 + (columns_of_graph - 2)*self.tasks.len();
        // Define edges
        let mut edge_costs: Vec<f64>  = Vec::new();
        let mut distances: Vec<f64> = Vec::new();
        let mut rows: Vec<usize> = Vec::new();
        let mut columns: Vec<usize> = Vec::new();
        let mut values: Vec<i64> = Vec::new();
        let mut edge: usize = 0;

        // first set of edges is from salesmen 1 to all the cities
        let mut to_node = self.agents.len();
        let mut from_node = 0;
        for agent in &self.agents{
            for task in &self.tasks{
                // Get cost
                let task_location = task.location;
                let dist = agent.calc_distance(task_location);
                edge_costs.push(dist);
                distances.push(dist);

                // Incidence matrix
                columns.push(edge);
                columns.push(edge);
                rows.push(to_node);
                rows.push(from_node);
                values.push(-1);
                values.push(1);

                edge += 1;
                to_node += 1;
            }
            from_node += 1;
            to_node = self.agents.len();
        }
        to_node = self.agents.len() + self.tasks.len();
        // This iterates over all the edges that directly connect cities together and cities the ending depots
        for i in 0..columns_of_graph - 2{
            for task in &self.tasks{
                for next_task in &self.tasks{
                    if i == columns_of_graph - 3{
                        break;
                    }
                    // Cost
                    let dist = task.calc_distance(next_task.location);
                    edge_costs.push(dist);
                    distances.push(dist);

                    // Incidence 
                    columns.push(edge);
                    columns.push(edge);
                    rows.push(to_node);
                    rows.push(from_node);
                    values.push(-1);
                    values.push(1);
                
                    edge += 1;
                    to_node += 1;
                }
                // Connect to end nodes
                // Set the node the edge is going to
                to_node = self.agents.len() + (columns_of_graph - 2)*self.tasks.len();
                for agent in &self.agents{
                    // Cost
                    let dist = task.calc_distance(agent.depot_location);
                    let multiplyer = if i >= 10 {1.0} else {(10 - i) as f64};
                    edge_costs.push((multiplyer * early_ender as f64 * (columns_of_graph - i) as f64) *dist);
                    distances.push(dist);
                    // Incidence 
                    columns.push(edge);
                    columns.push(edge);
                    rows.push(to_node);
                    rows.push(from_node);
                    values.push(-1);
                    values.push(1);
                
                    edge += 1;
                    to_node += 1;
                }

                // Set the nodes the edge is connected to
                to_node = self.agents.len() + (i+1)*self.tasks.len();
                from_node += 1;
            }
            // Set the node the edge is going to
            to_node = self.agents.len() + (i+2)*self.tasks.len();
        }
        let other_data = vec![self.tasks.len(), self.agents.len(), columns_of_graph - 2, nodes_in_graph, edge_costs.len()];  
        return (distances, edge_costs, rows, columns, values, other_data);
    }

    /// Creates the network flow formulation of the mTSP
    /// 
    /// * 'columns_of_graph' - The total number of columns
    /// * 'early_ender' - The extra multiplyer to adjust the costs of an agents route ending earlier in the graph
    ///     This means an agent collects less tasks than the other agent(s)
    pub fn _to_incidence(
        self, 
        columns_of_graph: usize, 
        early_ender: usize
    ) -> (Vec<f64>, Vec<f64>, Vec<Vec<i32>>, Vec<usize>){
        let nodes_in_graph = self.agents.len()*2 + (columns_of_graph - 2)*self.tasks.len();
        // Define edges
        let mut edge_costs: Vec<f64>  = Vec::new();
        let mut distances: Vec<f64> = Vec::new();
        let mut incidence_mat: Vec<Vec<i32>> = Vec::new();

        // first set of edges is from salesmen 1 to all the cities
        let mut to_node = self.agents.len();
        let mut from_node = 0;
        for agent in &self.agents{
            for task in &self.tasks{
                // Get cost
                let task_location = task.location;
                let dist = agent.calc_distance(task_location);
                edge_costs.push(dist);
                distances.push(dist);

                // Incidence matrix
                _update_incidence(&nodes_in_graph, &to_node, &from_node, &mut incidence_mat);
                to_node += 1;
            }
            from_node += 1;
            to_node = self.agents.len();
        }
        to_node = self.agents.len() + self.tasks.len();
        // This iterates over all the edges that directly connect cities together and cities the ending depots
        for i in 0..columns_of_graph - 2{
            for task in &self.tasks{
                for next_task in &self.tasks{
                    if i == columns_of_graph - 3{
                        break;
                    }
                    // Cost
                    let dist = task.calc_distance(next_task.location);
                    edge_costs.push(dist);
                    distances.push(dist);

                    // Incidence 
                    _update_incidence(&nodes_in_graph, &to_node, &from_node, &mut incidence_mat);
                    to_node += 1;
                }
                // Connect to end nodes
                // Set the node the edge is going to
                to_node = self.agents.len() + (columns_of_graph - 2)*self.tasks.len();
                for agent in &self.agents{
                    // Cost
                    let dist = task.calc_distance(agent.depot_location);
                    let multiplyer = if i >= 10 {1.0} else {(10 - i) as f64};
                    edge_costs.push((multiplyer * early_ender as f64 * (columns_of_graph - i) as f64) *dist);
                    distances.push(dist);
                    // Incidence 
                    _update_incidence(&nodes_in_graph, &to_node, &from_node, &mut incidence_mat);
                    to_node += 1;
                }

                // Set the nodes the edge is connected to
                to_node = self.agents.len() + (i+1)*self.tasks.len();
                from_node += 1;
            }
            // Set the node the edge is going to
            to_node = self.agents.len() + (i+2)*self.tasks.len();
        }
        let other_data = vec![self.tasks.len(), self.agents.len(), columns_of_graph - 2, nodes_in_graph, edge_costs.len()];  
        return (distances, edge_costs, incidence_mat, other_data);
    }
}

/// This function adds an edge to the incidence matrix
/// 
/// * 'num_nodes' - The total number of nodes in the graph
/// * 'to_node' - The node the edge is going to
/// * 'from_node' - The node the edge is coming from
/// * 'inc_mat' - The current incidence matrix
fn _update_incidence(num_nodes: &usize, to_node: &usize, from_node: &usize, inc_mat: &mut Vec<Vec<i32>>) {
    let mut new_col = vec![0; *num_nodes ];
    new_col[*to_node] = -1;
    new_col[*from_node] = 1;
    inc_mat.push(new_col);
}

/// Create a simple TSP
/// 
/// Used mostly for debugging/testing
fn _create_specific_tsp() -> Tsp{
    let tasks = vec![
        Task{ location: (0.5, 1.0)}, 
        Task{ location: (4.5, 3.0)},
        Task{ location: (2.0, 4.5)},
        Task{ location: (1.5, 0.5)},
        Task{ location: (4.0, 0.5)},
        Task{ location: (2.0, 3.0)},
        Task{ location: (2.5, 4.5)},
    ];
    let agents = vec![Agent{
        depot_location: (0.5, 0.5),
        current: (0.5, 0.5),
        tour: tasks.clone()
    }];
    let w_size = (6.0, 6.0);
    let mut problem = Tsp{
        tasks: tasks,
        agents: agents,
        world_size: w_size,
        tours: Vec::new(),
        total_distances: Vec::new(),
        groups: None
    };
    problem.tours = problem.calc_tours();
    problem.total_distances = problem.calc_all_distance();
    problem
}

/// Create a simple MTSP
/// 
/// Used mostly for debugging/testing
fn _create_specific_mtsp() -> Tsp{
    let mut tasks1 = vec![
        Task{ location: (0.5, 1.0)}, 
        Task{ location: (4.5, 3.0)},
        Task{ location: (2.0, 4.5)},
        Task{ location: (1.5, 0.5)},
        Task{ location: (4.0, 0.5)},
        Task{ location: (2.0, 3.0)},
        Task{ location: (2.5, 4.5)},
    ];
    let tasks2 = vec![
        Task{ location: (1.0, 0.5)}, 
        Task{ location: (3.0, 4.5)},
        Task{ location: (4.5, 2.0)},
        Task{ location: (0.5, 1.5)},
        Task{ location: (0.5, 4.0)},
        Task{ location: (3.0, 2.0)},
        Task{ location: (4.5, 2.5)},
    ];
    let agents = vec![
        Agent{
            depot_location: (0.5, 0.5),
            current: (0.5, 0.5),
            tour: tasks1.clone()
    },
        Agent{
            depot_location: (4.5, 4.5),
            current: (4.5, 4.5),
            tour: tasks2.clone()
    }];
    let w_size = (6.0, 6.0);
    tasks1.extend(tasks2);
    let mut problem = Tsp{
        tasks: tasks1,
        agents: agents,
        world_size: w_size,
        tours: Vec::new(),
        total_distances: Vec::new(),
        groups: None
    };
    problem.tours = problem.calc_tours();
    problem.total_distances = problem.calc_all_distance();
    problem
    
}

/// Create a MTSP from input
/// 
/// * 'num_agents' - The number of agents
/// * 'num_tasks' - The number of tasks
/// * 'world_size' - The total size of the world the agents and tasks live in
pub(crate)fn create_random_mtsp(num_agents: usize, num_tasks: usize, world_size: (f64, f64)) -> Tsp{
    let mut tasks = vec![];
    for _ in 0..num_tasks{
        tasks.push(Task{ location: (world_size.0*rand::random::<f64>(), world_size.1*rand::random::<f64>())});
    }
    let task_lists: Vec<Vec<Task>> = tasks.chunks(num_tasks/num_agents).map(|s| s.into()).collect();
    let mut agents = vec![];
    for i in 0..num_agents{
        let pos = (world_size.0*rand::random::<f64>(), world_size.1*rand::random::<f64>());
        agents.push(Agent{
            depot_location: pos,
            current: pos,
            tour: task_lists[i].clone()
        })
    }
    let mut problem = Tsp{
        tasks: tasks,
        agents: agents,
        world_size: world_size,
        tours: Vec::new(),
        total_distances: Vec::new(),
        groups: None
    };
    problem.tours = problem.calc_tours();
    problem.total_distances = problem.calc_all_distance();
    problem
}

/// Uses previously defined functions to create a simple network flow MTSP implementation
/// 
/// Used mostly for debugging/testing
fn _create_basic_network_flow()-> io::Result<(Vec<usize>, Vec<f64>, Vec<f64>, Vec<Vec<i32>>)> {
    let prob = _create_specific_mtsp();
    let (distances, costs, mat, other_data) = prob._to_incidence(9, 2);
    Ok((other_data, costs, distances,  mat))
}

/// Uses previously defined functions to create a network flow MTSP implementation
/// /// * 'num_agents' - The number of agents
/// * 'num_tasks' - The number of tasks
/// * 'world_size' - The total size of the world the agents and tasks live in
/// * 'num_columns' - The total number of columns in the problem
/// * 'cost_multiplyer' - The extra multiplyer for augmenting the network flow edge costs
pub(crate)fn create_random_network_flow(
    num_agents: usize, 
    num_tasks: usize, 
    world_size: (f64, f64), 
    mut num_columns: usize, 
    cost_multiplyer: usize
) -> io::Result<(Vec<usize>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>, Vec<i64>, Duration, Duration)>{
    let mut time = Instant::now();
    let prob = create_random_mtsp(num_agents, num_tasks, world_size);
    let creation_time = time.elapsed();
    if num_columns < num_tasks + 1{
        num_columns = num_tasks + 1;
    }
    time = Instant::now();
    let (distances, costs, rows, cols, values, other_data) = prob.to_incidence_simplified(num_columns, cost_multiplyer);

    let incidence_time = time.elapsed();
    Ok((other_data, costs, distances,  rows, cols, values, creation_time, incidence_time))
    
}

pub(crate)fn read_toml_and_run()-> Result<(), Box<dyn std::error::Error>>{
    let toml_content = fs::read_to_string("Variables/genetic.toml")?;

    let val: Value = toml_content.parse()?;
    let mtsp_vars = val.get("mTSP").unwrap();
    let num_agents:usize = mtsp_vars.get("num_agents").unwrap().as_integer().unwrap() as usize;
    let num_tasks = mtsp_vars.get("num_tasks").unwrap().as_integer().unwrap() as usize;
    let world_size_array = mtsp_vars.get("world_size").unwrap().as_array().unwrap();
    let world_size: Vec<f64> = world_size_array.iter()
        .filter_map(|val| val.as_float())
        .collect();
    let problem = create_random_mtsp(num_agents, num_tasks, (world_size[0], world_size[1]));
    let mut problems = k_clustering_no_agents(&problem);
    let mut full_problems: Vec<Tsp> = Vec::new();
    for tasklist in problems{
        full_problems.push(Tsp { 
            tasks: tasklist, 
            agents: vec![Agent {depot_location: (0.0, 0.0), current: (0.0, 0.0), tour: Vec::new()}], 
            world_size: problem.world_size, 
            tours: Vec::new(), 
            total_distances: Vec::new(),
            groups: None 
        });
    }
    for entry in std::fs::read_dir("Images/").unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    for (i, tsp_prob) in full_problems.iter().enumerate(){
        let label = format!("Agent {}", i);
        let _ = tsp_prob.draw_solution(label);
    }
    // let _ = problem.draw_solution("original_mtsp".to_string());
    //     let problems = k_clustering(problem);
    //     for i in 0..problems.len(){
    //         let label = format!("Agent {}", i);
    //         let _ = problems[i].draw_solution(label);
    //     }
    Ok(())
}