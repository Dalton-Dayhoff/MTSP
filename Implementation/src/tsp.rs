use std::f32::INFINITY;
use std::collections::HashMap;
use rand;
use plotters::prelude::*;

const NUMBER_AGENTS: usize = 1;
const NUMBER_TASKS: usize = 7;

#[derive(Debug)]
struct Task {
    location: (f64, f64),
}

impl Task {
    fn calc_distance(&self, next_task: (f64,f64)) -> f64{
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
struct Agent {
    start: (f64, f64),
    current: (f64, f64),
    tour: Vec<Task>
}
impl Agent {
    fn calc_distance_from_start(&self, next_task: (f64,f64)) -> f64{
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
pub struct Tsp {
    tasks: Vec<Task>,
    agents: Vec<Agent>,
    world_size: (f64, f64),
    tours: Vec<Vec<f64>>,
    total_distances: Vec<f64>
}

impl Tsp {
    fn calc_all_distance(&self) -> Vec<f64> {
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

    fn calc_tours(&self) -> Vec<Vec<f64>> {
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

    pub fn draw_solution(&self, name: String) -> Result<(), Box<dyn std::error::Error>>{
        let name = name + ".svg";
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
            let label_string = format!("Agent{}", &i);
            match chart.draw_series(LineSeries::new(tour, color)) {
                Ok(_) => (),
                Err(err) => eprintln!("Error drawing series: {}", err),
            }
        }
        
        Ok(())
    }
}



pub fn k_clustering(problem: Tsp) -> Vec<Tsp>{
    let number_of_clusters = problem.agents.len();
    // Initilize centroid to be agents
    // This was done to push the centroids towards the agents when the agents were not part of centroid calculation
    // Could play with this for different results
    let mut centroids: Vec<(f64, f64)> = problem.agents.iter().map(|agent| agent.start).collect();

    let mut diff = 1.0;
    let mut i = 0;
    struct Cluster{
        task_list: Vec<Task>,
        agent: Agent
    }
    impl Clone for Cluster {
        fn clone(&self) -> Self {
        Self { task_list: self.task_list.clone(), agent: self.agent.clone() }
    }
    }
    let mut seperated_task_list: Vec<Cluster> = Vec::new();
    for j in 0..number_of_clusters{
        seperated_task_list.push(Cluster{task_list: Vec::new(), agent: problem.agents[j].clone()});
    }
    let initialize_task_list = seperated_task_list.clone();
    while diff != 0.0 && i < problem.agents.len()*10{
        seperated_task_list = initialize_task_list.clone();
        // Assign each task to it's closest centroid
        for task in &problem.tasks{
            let mut distances = vec![];
            for centroid in centroids.clone(){
                distances.push(task.calc_distance(centroid));
            }
            let (min_ind, _) = distances.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
            seperated_task_list[min_ind].task_list.push(task.clone());
        }
        // Push groups towards closer size
        // Comment this for loop out for only dense data
        for i in 0..6*number_of_clusters/3{
            let mut min_indices: Vec<usize> = (0..seperated_task_list.len()).collect();
            min_indices.sort_by_key(|&index| seperated_task_list[index].task_list.len());
            let (max_ind, _) = seperated_task_list.iter().enumerate().max_by_key(|(_, v)| v.task_list.len()).unwrap();
            // Find task in largest task list closest to original centroid.
            let mut closest_task_ind = 0;
            let mut closest_list_ind = 0;
            let mut dist: f64 = INFINITY.into();
            for j in 0..seperated_task_list[max_ind].task_list.len(){
                for k in 0..2{
                    let min_ind = min_indices[k];
                    let comp = seperated_task_list[max_ind].task_list[j].calc_distance(centroids[min_ind]);
                    if comp < dist{
                        closest_task_ind = j;
                        closest_list_ind = k;
                        dist = comp;
                    }
                }
            }
            let moved_task = seperated_task_list[max_ind].task_list.remove(closest_task_ind);
            seperated_task_list[closest_list_ind].task_list.push(moved_task);
        }
        
        // Find centroids of groups, without using the agents
        let mut new_centroids = Vec::new();
        for cluster in seperated_task_list.clone(){
            let (sum_x, sum_y): (f64, f64) = cluster
                .task_list
                .iter()
                .map(|task| task.location)
                .fold((0.0, 0.0), |(acc_x, acc_y), (x, y)| (acc_x + x, acc_y + y));
            let new_centroid = (sum_x/(cluster.task_list.len() as f64), sum_y/(cluster.task_list.len() as f64));
            new_centroids.push(new_centroid);
        }
        // Assign agents to closest centroids
        let mut possible_centroid_indecies =Vec::from_iter(0..number_of_clusters);
        for agent in &problem.agents{
            let mut min_dist = INFINITY as f64;
            let mut index_of_possible_centroids = 0;
            for j in &possible_centroid_indecies{
                let centroid = new_centroids[*j];
                let distance = agent.calc_distance_from_start(centroid);
                    if distance < min_dist{
                        min_dist = distance;
                        index_of_possible_centroids = *j as usize;

                     }
            }
            seperated_task_list[index_of_possible_centroids].agent = agent.clone();
            possible_centroid_indecies.retain(|&x| x != index_of_possible_centroids);
        }

        // Find new centroids including the agents
        new_centroids = vec![];
        for cluster in seperated_task_list.clone(){
            let (mut sum_x, mut sum_y): (f64, f64) = cluster
                .task_list
                .iter()
                .map(|task| task.location)
                .fold((0.0, 0.0), |(acc_x, acc_y), (x, y)| (acc_x + x, acc_y + y));
            sum_x += cluster.agent.start.0;
            sum_y += cluster.agent.start.1;
            let new_centroid = (sum_x/((cluster.task_list.len()+ 1) as f64), sum_y/((cluster.task_list.len() + 1) as f64));
            new_centroids.push(new_centroid);
        }

        // Compare new centroids with previous iteration
        diff = 0.0;
        for j in 0..centroids.len(){
            println!("Old Centroid: {:?}", centroids[j]);
            println!("New Centroid: {:?}", new_centroids[j]);
            let dist = ((centroids[j].0 - new_centroids[j].0).powf(2.0) + (centroids[j].1 - new_centroids[j].1).powf(2.0)).sqrt();
            diff += dist;
        }
        println!("Distance from previous centroid: {}", &diff);
        centroids = new_centroids;
        i += 1; 
    }
    // Create the vector Traveling salesmen problems to solve individually
    let mut seperated_problems: Vec<Tsp> = vec![];
    for cluster in seperated_task_list{
        seperated_problems.push(Tsp{
            tasks: cluster.task_list, 
            agents: vec![cluster.agent], 
            world_size: problem.world_size, 
            tours: Vec::new(), 
            total_distances: Vec::new()})
    }
    for i in 0..seperated_problems.len(){
        println!("Problem {} has {} tasks", i, seperated_problems[i].tasks.len());
        println!("Problem {} has {:?} agent starting points", i, seperated_problems[i].agents.iter().map(|agent: &Agent| agent.start).collect::<Vec<(f64, f64)>>());
        seperated_problems[i].agents[0].tour = seperated_problems[i].tasks.clone();
        seperated_problems[i].tours = seperated_problems[i].calc_tours();
        seperated_problems[i].total_distances = seperated_problems[i].calc_all_distance();
        println!("The problem has total distance {:?}", seperated_problems[i].total_distances);
    }
    // println!("Seperated Problems: {:?}", seperated_problems);
    seperated_problems

}


pub fn create_specific_tsp() -> Tsp{
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

pub fn create_specific_mtsp() -> Tsp{
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


pub fn create_random_mtsp(num_agents: usize, num_tasks: usize, world_size: (f64, f64)) -> Tsp{
    let mut tasks = vec![];
    for i in 0..num_tasks{
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
