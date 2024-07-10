use rand;
use plotters::prelude::*;
use std::io;

#[derive(Debug)]
pub(crate)struct Task {
    pub(crate)location: (f64, f64),
}

impl Task {
    pub(crate)fn calc_distance(&self, next_task: (f64,f64)) -> f64{
        let x_dist: f64 = self.location.0 - next_task.0;
        let y_dist: f64 = self.location.1 - next_task.1;
        ((x_dist.powf(2.0) + y_dist.powf(2.0)) as f64).sqrt()
    }   
}
impl Clone for Task{
    fn clone(&self) -> Self {
        Self { location: self.location.clone() }
    }
}

#[derive(Debug)]
pub(crate)struct Agent {
    pub(crate)start: (f64, f64),
    pub(crate)current: (f64, f64),
    pub(crate)tour: Vec<Task>
}
impl Agent {
    pub(crate)fn calc_distance(&self, next_task: (f64,f64)) -> f64{
        let x_dist: f64 = self.start.0 - next_task.0;
        let y_dist: f64 = self.start.1 - next_task.1;
        ((x_dist.powf(2.0) + y_dist.powf(2.0)) as f64).sqrt()
    }   
}

impl Clone for Agent{
    fn clone(&self) -> Self {
        Self {start: self.start.clone(), current: self.current.clone(), tour: self.tour.clone() }
    }
}
#[derive(Debug)]
pub(crate)struct Tsp {
    pub(crate) tasks: Vec<Task>,
    pub(crate) agents: Vec<Agent>,
    pub(crate) world_size: (f64, f64),
    pub(crate) tours: Vec<Vec<f64>>,
    pub(crate) total_distances: Vec<f64>
}

impl Tsp {
    pub(crate) fn calc_all_distance(&self) -> Vec<f64> {
        let mut distances: Vec<f64> = Vec::new();
        for i in 0..self.tours.len(){
            let tour = &self.tours[i];
            let start_dist = self.agents[i].tour[0].calc_distance(self.agents[i].start);
            let end_dist = self.agents[i].tour[tour.len() - 1].calc_distance(self.agents[i].start);
            let mid_dist: f64 = tour.iter().sum();
            distances.push(mid_dist + start_dist + end_dist);
        }
        distances
    }

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

    pub(crate)fn draw_solution(&self, name: String) -> Result<(), Box<dyn std::error::Error>>{
        let name = "Images/".to_owned() + &name + ".svg";
        let root = SVGBackend::new(&name, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Scatter Plot", ("sans-serif", 40).into_font())
            .build_cartesian_2d(0.0..self.world_size.0, 0.0..self.world_size.1)?;
        let num_colors = self.agents.len();
        let mut colors = Vec::with_capacity(num_colors);
        for i in 0..num_colors {
            let red = (i * 33) % 256;
            let green = (i * 57) % 256;
            let blue = (i * 43) % 256;
            colors.push(RGBColor(red as u8, green as u8, blue as u8));
    }

        chart.draw_series(
            self.tasks.iter().map(|task| Circle::new((task.location.0, task.location.1), 5, RGBColor(0, 0, 255))),
    )?;
        chart.draw_series(
            self.agents.iter().zip(colors.iter()).map(|(agent, color)| {
                TriangleMarker::new((agent.start.0, agent.start.1), 5, color.clone())
            }),
    )?;
        let mut all_tours: Vec<Vec<(f64,f64)>> = Vec::new();
        for agent in &self.agents{
            let mut tour: Vec<(f64,f64)> = agent.tour.iter().map(|task| (task.location.0, task.location.1)).collect();
            tour.insert(0, agent.start);
            tour.push(agent.start);
            all_tours.push(tour)
        }
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

    pub fn to_incidence(self, rows_of_graph: usize, early_ender: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<i32>>, Vec<usize>){
        let nodes_in_graph = self.agents.len()*2 + (rows_of_graph - 2)*self.tasks.len();
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
                update_incidence(&nodes_in_graph, &to_node, &from_node, &mut incidence_mat);
                to_node += 1;
            }
            from_node += 1;
            to_node = self.agents.len();
        }
        to_node = self.agents.len() + self.tasks.len();
        // This iterates over all the edges that directly connect cities together
        for i in 0..rows_of_graph - 2{
            for task in &self.tasks{
                for next_task in &self.tasks{
                    if i == rows_of_graph - 3{
                        break;
                    }
                    // Cost
                    let dist = task.calc_distance(next_task.location);
                    edge_costs.push(dist);
                    distances.push(dist);

                    // Incidence 
                    update_incidence(&nodes_in_graph, &to_node, &from_node, &mut incidence_mat);
                    to_node += 1;
                }
                // Connect to end nodes
                // Set the node the edge is going to
                to_node = self.agents.len() + (rows_of_graph - 2)*self.tasks.len();
                for agent in &self.agents{
                    // Cost
                    let dist = task.calc_distance(agent.start);
                    let multiplyer = if i >= 10 {1.0} else {(10 - i) as f64};
                    edge_costs.push((multiplyer * early_ender as f64 * (rows_of_graph - i) as f64) *dist);
                    distances.push(dist);
                    // Incidence 
                    update_incidence(&nodes_in_graph, &to_node, &from_node, &mut incidence_mat);
                    to_node += 1;
                }

                // Set the nodes the edge is connected to
                to_node = self.agents.len() + (i+1)*self.tasks.len();
                from_node += 1;
            }
            // Set the node the edge is going to
            to_node = self.agents.len() + (i+2)*self.tasks.len();
        }
        let other_data = vec![self.tasks.len(), self.agents.len(), rows_of_graph - 2, nodes_in_graph, edge_costs.len()];  
        return (distances, edge_costs, incidence_mat, other_data);
    }
}

fn update_incidence(num_nodes: &usize, to_node: &usize, from_node: &usize, inc_mat: &mut Vec<Vec<i32>>) {
    let mut new_col = vec![0; *num_nodes ];
    new_col[*to_node] = -1;
    new_col[*from_node] = 1;
    inc_mat.push(new_col);
}

pub(crate)fn create_specific_tsp() -> Tsp{
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
        start: (0.5, 0.5),
        current: (0.5, 0.5),
        tour: tasks.clone()
    }];
    let w_size = (6.0, 6.0);
    let mut problem = Tsp{
        tasks: tasks,
        agents: agents,
        world_size: w_size,
        tours: Vec::new(),
        total_distances: Vec::new()
    };
    problem.tours = problem.calc_tours();
    problem.total_distances = problem.calc_all_distance();
    problem
}

pub(crate)fn create_specific_mtsp() -> Tsp{
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
            start: (0.5, 0.5),
            current: (0.5, 0.5),
            tour: tasks1.clone()
    },
        Agent{
            start: (4.5, 4.5),
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
        total_distances: Vec::new()
    };
    problem.tours = problem.calc_tours();
    problem.total_distances = problem.calc_all_distance();
    problem
    
}


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
            start: pos,
            current: pos,
            tour: task_lists[i].clone()
        })
    }
    let mut problem = Tsp{
        tasks: tasks,
        agents: agents,
        world_size: world_size,
        tours: Vec::new(),
        total_distances: Vec::new()
    };
    problem.tours = problem.calc_tours();
    problem.total_distances = problem.calc_all_distance();
    problem
}

pub(crate) fn _create_basic_network_flow()-> io::Result<(Vec<usize>, Vec<f64>, Vec<Vec<i32>>)> {
    let prob = create_specific_mtsp();
    let (distances, costs, mat, other_data) = prob.to_incidence(9, 2);
    Ok((other_data, costs, mat))
}


pub(crate)fn create_random_network_flow(num_agents: usize, num_tasks: usize, world_size: (f64, f64), mut num_rows: usize, cost_multiplyer: usize) -> io::Result<(Vec<usize>, Vec<f64>, Vec<f64>, Vec<Vec<i32>>)>{
    let prob = create_random_mtsp(num_agents, num_tasks, world_size);
    if num_rows < num_tasks{
        num_rows = num_tasks + 1;
    }
    let (distances, costs, mat, other_data) = prob.to_incidence(num_rows, cost_multiplyer);
    Ok((other_data, costs, distances,  mat))
    
}