use std::f64::INFINITY;

use crate::tsp::*;

pub fn k_clustering(problem: Tsp) -> Vec<Tsp>{
    let number_of_clusters = problem.agents.len();
    // Initilize centroid to be agents
    // This was done to push the centroids towards the agents when the agents were not part of centroid calculation
    // Could play with this for different results
    let mut centroids: Vec<(f64, f64)> = problem.agents.iter().map(|agent| agent.depot_location).collect();

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
                let distance = agent.calc_distance(centroid);
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
            let weighting_of_agent_location = 0.5;
            sum_x += weighting_of_agent_location*cluster.agent.depot_location.0;
            sum_y += weighting_of_agent_location*cluster.agent.depot_location.1;
            let new_centroid = (sum_x/((cluster.task_list.len()+ 1) as f64), sum_y/((cluster.task_list.len() + 1) as f64));
            new_centroids.push(new_centroid);
        }

        // Compare new centroids with previous iteration
        diff = 0.0;
        for j in 0..centroids.len(){
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
        seperated_problems[i].agents[0].tour = seperated_problems[i].tasks.clone();
        seperated_problems[i].tours = seperated_problems[i].calc_tours();
        seperated_problems[i].total_distances = seperated_problems[i].calc_all_distance();
    }
    seperated_problems

}