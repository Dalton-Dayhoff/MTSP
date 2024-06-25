
use initilization::k_clustering;
use tsp::create_basic_network_flow;

mod tsp;
mod initilization;

fn main() {
    // let problem = tsp::create_random_mtsp(5, 30, (20.0, 20.0));
    // let _ = problem.draw_solution("original_mtsp".to_string());
    // let problems = k_clustering(problem);
    // for i in 0..problems.len(){
    //     let label = format!("Agent {}", i);
    //     let _ = problems[i].draw_solution(label);
    // }
    let result = create_basic_network_flow();
    match result {
        Ok(v) => println!("working with version: {v:?}"),
        Err(e) => println!("error parsing header: {e:?}"),
    }
}