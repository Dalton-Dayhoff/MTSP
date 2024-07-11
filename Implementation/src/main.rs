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
}