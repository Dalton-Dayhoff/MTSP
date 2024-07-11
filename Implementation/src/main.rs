use network_flow::{setup, read_toml};
use tsp::read_toml_and_run;

mod tsp;
mod initialization;
mod network_flow;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2{
        println!("No Solver Specified")
    }else if args[1].to_lowercase() == "milp"{
        setup().unwrap();
        if args.len() < 3{
            println!("MILP selected but no version specified");
        } else{
            read_toml(args[2].clone()).unwrap()
        }
    } else if args[1].to_lowercase() == "cluster"{
        read_toml_and_run().unwrap();
    }
    // let rust_or_python = 0;
    // // let (data, costs, inc_mat) = create_random_network_flow(3, 10, (10.0,10.0), 8, 2).unwrap();
    // // let result = create_basic_network_flow();
    // if rust_or_python == 0{
    //     // For python Data
    //     let _ = setup();
    //     test_network_flow(10, 40, 10, 6, 3, (10.0, 10.0), 2);
    //     // let result = get_results(data, costs, inc_mat).unwrap();
        
    // } else{
    //     // Rust Code
    //     let problem = tsp::create_random_mtsp(5, 30, (20.0, 20.0));
    //     let _ = problem.draw_solution("original_mtsp".to_string());
    //     let problems = k_clustering(problem);
    //     for i in 0..problems.len(){
    //         let label = format!("Agent {}", i);
    //         let _ = problems[i].draw_solution(label);
    //     }
    // }

    
}