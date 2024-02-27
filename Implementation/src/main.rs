use std::fmt::format;

use tsp::k_clustering;

mod tsp;

fn main() {
    let problem = tsp::create_random_mtsp(5, 30, (20.0, 20.0));
    let _ = problem.draw_solution("original_mtsp".to_string());
    let problems = k_clustering(problem);
    for i in 0..problems.len(){
        let label = format!("Agent {}", i);
        let _ = problems[i].draw_solution(label);
    }
    

}
