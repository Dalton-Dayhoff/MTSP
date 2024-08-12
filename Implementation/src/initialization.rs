use std::f64::INFINITY;

use rand::Rng;

use crate::tsp::*;


struct Cluster{
    task_list: Vec<Task>,
    agent: Option<Agent>,
    centroid: (f64, f64),
    centroid_sums: (f64, f64),
    std_dev: f64,
    dev_sum: f64
}
impl Cluster {
    fn add_task(&mut self, task: Task) {
        let pos = task.location;
        self.task_list.push(task);
        // Update metrics 
        self.centroid_sums.0 += pos.0;
        self.centroid_sums.1 += pos.1;
        self.centroid = (
            self.centroid_sums.0 / self.task_list.len() as f64, 
            self.centroid_sums.1 / self.task_list.len() as f64
        );
        let x_dif = (pos.0 - self.centroid.0).abs();
        let y_dif = (pos.1 - self.centroid.1).abs(); 
        let dist = (x_dif.powf(2.0) + y_dif.powf(2.0)).sqrt();
        self.dev_sum += dist;
        self.std_dev = self.dev_sum / self.task_list.len() as f64;
    }

    fn remove_task(&mut self, task_ind: usize) -> Task{
        let task = self.task_list.remove(task_ind);
        // Update metrics
        let pos = task.location;
        let x_dif = (pos.0 - self.centroid.0).abs();
        let y_dif = (pos.1 - self.centroid.1).abs(); 
        let dist = (x_dif.exp2() + y_dif.exp2()).sqrt();
        self.dev_sum -= dist;
        self.std_dev = self.dev_sum / self.task_list.len() as f64;
        self.centroid_sums.0 -= pos.0;
        self.centroid_sums.1 -= pos.1;
        self.centroid = (
            self.centroid_sums.0 / self.task_list.len() as f64, 
            self.centroid_sums.1 / self.task_list.len() as f64
        );
        return task;

    }
}
impl Clone for Cluster {
    fn clone(&self) -> Self {
        Self { 
            task_list: self.task_list.clone(), 
            agent: self.agent.clone(), 
            centroid: self.centroid.clone(),
            centroid_sums: self.centroid_sums.clone(),
            std_dev: self.std_dev.clone(),
            dev_sum: self.dev_sum.clone() 
        }
}
}

pub fn k_clustering(problem: Tsp) -> Vec<Vec<Task>>{
    let number_of_clusters = problem.agents.len();
    // Initilize centroid to be agents
    // This was done to push the centroids towards the agents when the agents were not part of centroid calculation
    // Could play with this for different results

    let mut centroids: Vec<(f64, f64)> = problem.agents.iter().map(|agent| agent.depot_location).collect();

    let mut diff = 1.0;
    let mut i = 0;

    let mut seperated_task_list: Vec<Cluster> = Vec::new();
    for j in 0..number_of_clusters{
        seperated_task_list.push(Cluster{
            task_list: Vec::new(), 
            agent: Some(problem.agents[j].clone()), 
            centroid: (0.0, 0.0),
            centroid_sums: (0.0, 0.0),
            std_dev: 0.0,
            dev_sum: 0.0
        });
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
            seperated_task_list[min_ind].add_task(task.clone());
        }
        // Push groups towards closer size
        // Comment this for loop out for only dense data
        for _ in 0..6*number_of_clusters/3{
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
            let moved_task = seperated_task_list[max_ind].remove_task(closest_task_ind);
            seperated_task_list[closest_list_ind].add_task(moved_task);
        }
        
        // Assign agents to closest centroids
        let mut possible_centroid_indecies =Vec::from_iter(0..number_of_clusters);
        for agent in &problem.agents{
            let mut min_dist = INFINITY as f64;
            let mut index_of_possible_centroids = 0;
            for j in &possible_centroid_indecies{
                let centroid = seperated_task_list[*j].centroid;
                let distance = agent.calc_distance(centroid);
                    if distance < min_dist{
                        min_dist = distance;
                        index_of_possible_centroids = *j as usize;

                     }
            }
            seperated_task_list[index_of_possible_centroids].agent = Some(agent.clone());
            possible_centroid_indecies.retain(|&x| x != index_of_possible_centroids);
        }

        // Find new centroids including the agents
        let mut new_centroids = vec![];
        for cluster in seperated_task_list.clone(){
            let (mut sum_x, mut sum_y): (f64, f64) = cluster
                .task_list
                .iter()
                .map(|task| task.location)
                .fold((0.0, 0.0), |(acc_x, acc_y), (x, y)| (acc_x + x, acc_y + y));
            let weighting_of_agent_location = 0.5;
            sum_x += weighting_of_agent_location*cluster.agent.clone().unwrap().depot_location.0;
            sum_y += weighting_of_agent_location*cluster.agent.unwrap().depot_location.1;
            let new_centroid = (sum_x/((cluster.task_list.len()+ 1) as f64), sum_y/((cluster.task_list.len() + 1) as f64));
            new_centroids.push(new_centroid);
        }

        // Compare new centroids with previous iteration
        diff = 0.0;
        for j in 0..centroids.len(){
            let dist = ((centroids[j].0 - new_centroids[j].0).powf(2.0) + (centroids[j].1 - new_centroids[j].1).powf(2.0)).sqrt();
            diff += dist;
        }
        centroids = new_centroids;
        i += 1; 
    }
    // Create the vector Traveling salesmen problems to solve individually
    let mut task_lists: Vec<Vec<Task>> = Vec::new();
    for cluster in seperated_task_list{
        task_lists.push(cluster.task_list);
    }
    task_lists

    // let mut seperated_problems: Vec<Tsp> = vec![];
    // for cluster in seperated_task_list{
    //     seperated_problems.push(Tsp{
    //         tasks: cluster.task_list, 
    //         agents: vec![cluster.agent], 
    //         world_size: problem.world_size, 
    //         tours: Vec::new(), 
    //         total_distances: Vec::new()})
    // }
    // for i in 0..seperated_problems.len(){
    //     seperated_problems[i].agents[0].tour = seperated_problems[i].tasks.clone();
    //     seperated_problems[i].tours = seperated_problems[i].calc_tours();
    //     seperated_problems[i].total_distances = seperated_problems[i].calc_all_distance();
    // }
    // seperated_problems
}

pub fn k_clustering_no_agents(problem: Tsp) -> Vec<(Vec<Task>, (f64, f64))>{
    let mut diff = 1.0;
    let mut i: usize = 0;
    let num_clusters = Vec::from_iter(2..3*problem.agents.len());
    let mut seperated_task_list: Vec<Cluster> = Vec::new();
    let mut best_split = seperated_task_list.clone();
    let mut best_score = std::f64::INFINITY;
    let mut rng = rand::thread_rng();
    // loop to find ideal number of clusters
    for number_of_clusters in num_clusters{
        // Initialize clusters
        seperated_task_list = Vec::new();
        for j in 0..number_of_clusters{
            let x = problem.world_size.0*rng.gen::<f64>();
            let y = problem.world_size.1*rng.gen::<f64>();
            seperated_task_list.push(Cluster{
                task_list: Vec::new(), 
                agent: None, 
                centroid: (x, y),
                centroid_sums: (0.0, 0.0),
                std_dev: 0.0,
                dev_sum: 0.0
            });
        }
        let mut initialize_task_list = seperated_task_list.clone();
        // Find best configuration for given number of clusters
        while diff != 0.0 && i < number_of_clusters*5{
            seperated_task_list = initialize_task_list.clone();
            // Assign each task to it's closest centroid
            for task in &problem.tasks{
                let mut distances = vec![];
                for centroid in seperated_task_list.iter().map(|cluster| cluster.centroid){
                    distances.push(task.calc_distance(centroid));
                }
                let (min_ind, _) = distances.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
                seperated_task_list[min_ind].add_task(task.clone());
            }
            // Push groups towards closer size
            // Comment this for loop out for only dense data
            for _ in 0..6*number_of_clusters/3{
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
                        let comp = seperated_task_list[max_ind].task_list[j].calc_distance(seperated_task_list[min_ind].centroid);
                        if comp < dist{
                            closest_task_ind = j;
                            closest_list_ind = k;
                            dist = comp;
                        }
                    }
                }
                let moved_task = seperated_task_list[max_ind].remove_task(closest_task_ind);
                seperated_task_list[closest_list_ind].add_task(moved_task);
            }
            // Check how good clusters are
            let mut centroid_distances: Vec<f64> = Vec::new();
            for j in 0..number_of_clusters{
                let mut distances = Vec::new();
                for k in 0..number_of_clusters{
                    if j == k{
                        continue;
                    }
                    let current_centroid = seperated_task_list[j].centroid;
                    let next_centroid = seperated_task_list[k].centroid;
                    let dist = ((current_centroid.0 - next_centroid.0).powf(2.0) + (current_centroid.1 - next_centroid.1).powf(2.0)).sqrt();
                    distances.push(dist)
                }
                centroid_distances.push(distances.iter().sum::<f64>()/(distances.len() as f64))
            }
            diff = 0.0;
            for (j, intercluster_distance )in centroid_distances.iter().enumerate(){
                diff += seperated_task_list[j].std_dev/intercluster_distance
            }
            diff /= number_of_clusters as f64;
            println!("Number of clusters: {}", seperated_task_list.len());
            println!("Score: {}", diff);
            if diff < best_score{
                best_split = seperated_task_list.clone();
                best_score = diff;
            }
            // make new centroids
            for j in 0..number_of_clusters{
                let x = problem.world_size.0*rng.gen::<f64>();
                let y = problem.world_size.1*rng.gen::<f64>();
                initialize_task_list[j].centroid = (x,y);
            }
        }
        i = 0
    }
    // Create the vector Traveling salesmen problems to solve individually
    let mut task_lists: Vec<(Vec<Task>, (f64, f64))> = Vec::new();
    for cluster in best_split{
        if cluster.task_list.len() < 1{
            continue;
        }
        task_lists.push((cluster.task_list, cluster.centroid));
    }
    task_lists
}