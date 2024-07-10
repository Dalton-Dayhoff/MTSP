
use std::process::Command;
use full_palette::DEEPPURPLE;
use plotters::prelude::*;
use pyo3::prelude::*;
use pyo3::prelude::PyResult;
use rand::Rng;

use crate::tsp::create_random_network_flow;


/// Setup the virtual environment
/// Make sure Rust has access to the required python packages
pub(crate) fn setup() -> PyResult<()> {
    Python::with_gil(|py| {
        // Ensure the correct python version
        assert!(py.version().starts_with("3.12"));
        
         // Get the active virtual environment path using virtualenv
        let virtual_env_path = std::env::var("VIRTUAL_ENV").unwrap_or_else(|_| {
            panic!("No virtual environment found. Ensure you have an active virtual environment.");
        });

        // Construct the path to the site-packages directory
        let venv_site_packages = format!("{}/lib/python3.12/site-packages", virtual_env_path);

        // Give Rust access to the Python packages in the virtual environment
        let sys = py.import_bound("sys").unwrap();
        sys.getattr("path").unwrap().call_method1("append", (venv_site_packages.clone(),)).unwrap();

        // Function to install a package to the virtual environment
        fn install_package(package: &str, python_interpreter: &str) {
            let output = Command::new(python_interpreter)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg(package)
                .output()
                .expect("Failed to execute pip install");

            if !output.status.success() {
                println!(
                    "Failed to install package: {}. Error: {}",
                    package,
                    String::from_utf8_lossy(&output.stderr)
                );
            } else {
                println!(
                    "Successfully installed package: {}. Output: {}",
                    package,
                    String::from_utf8_lossy(&output.stdout)
                );
            }
        }
        // Check installed packages using pip
        let python_interpreter = "../.venv/bin/python";
        let output = Command::new(python_interpreter)
            .arg("-m")
            .arg("pip")
            .arg("list")
            .output()
            .expect("Failed to execute pip list");
        if !output.status.success() {
            println!(
                "Failed to list installed packages. Error: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Install required packages if not installed
        let required_packages = vec!["gurobipy", "numpy", "matplotlib"];
        let installed_packages = String::from_utf8_lossy(&output.stdout);
        for package in required_packages {
            if !installed_packages.contains(package) {
                println!("Installing package: {}", package);
                install_package(package, &python_interpreter);
            }
        }

    });
    Ok(())
}

/// Calls the gurobi solver from python to solve the network flow formulation of the MTSP
/// Returns the augmented score, the runtime, and the true score
/// 
/// * 'data' - Contains the num_tasks, num_agents, num_columns, num_nodes, num_edges for the variables struct in python
/// * 'costs' - Augmented costs for network flow
/// * 'distances' - True costs
/// * 'inc_mat' - The incidence matrix for the problem
pub(crate) fn get_results(
    data: Vec<usize>, 
    costs: Vec<f64>, 
    distances: Vec<f64>, 
    inc_mat: Vec<Vec<i32>>
) -> PyResult<(f64, f64, f64)>{
    // Start Python
    pyo3::prepare_freethreaded_python();
    // Get the python file as a string
    let py_functions = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/Solvers/Network_Flow/NetworkFlow.py"
    ));
    let (score, time, dist) = Python::with_gil(|py| {
        // Set up variables class
        // Read the string python file and pull out variables class
        let py_module = PyModule::from_code_bound(
            py, 
            py_functions, 
            "NetworkFlow.py", 
            "Variables"
        )
        .unwrap();
        let variables_class = py_module.getattr("Variables").unwrap();
        let variables = variables_class.call1(
            (data[0],// num tasks
            data[1], // num agents
            data[2], // num columns
            data[3], // num nodes
            data[4] // num edges
            )
        ).unwrap();

        // Solve the given problem using MILP
        // Convert to python types
        let costs_py = costs.into_py(py);
        let inc_mat_py = inc_mat.into_py(py);
        let variables_py = variables.into_py(py);
        let distances_py = distances.into_py(py);
        // Get the function to solve
        let solver: Py<PyAny> = py_module.getattr("solve_all_constraints").unwrap().into();
        let result = solver.call1(py, (costs_py, distances_py, inc_mat_py, variables_py));
        result.unwrap().extract(py).unwrap()
    });
    
    Ok((score, time, dist))
}

/// Used to test the network flow implementation
/// Graphs scoring, runtime, number of tasks, and number of agents for each trial
/// 
/// * 'num_trials' - The number of tests to run
/// * 'max_tasks' - The maximum number of tasks in any one trial
/// * 'min_tasks' - The minimum number of tasks in any one trial
/// * 'max_agents' - The maximum number of agents in any one trial
/// * 'min_tasks' - The minimum number of agents in any one trial
/// * 'world_size' - The size of the world, for generation of the MTSP
/// * 'cost_multiplyer' - Extra multiplyer to augment costs for network flow
pub(crate) fn test_network_flow(
    num_trials: usize, 
    max_tasks: usize, 
    min_tasks: usize, 
    max_agents: usize, 
    min_agents: usize, 
    world_size: (f64, f64),
    cost_multiplyer: usize
) {
    // Initialize data storage
    let mut scores: Vec<f64> = Vec::new();
    let mut times: Vec<f64> = Vec::new();
    let mut true_scores: Vec<f64> = Vec::new();
    let mut tasks: Vec<f64> = Vec::new();
    let mut agents: Vec<f64> = Vec::new();
    // Generate data
    for i in 0..num_trials{
        println!("Trial {}", i + 1);
        let num_tasks = rand::thread_rng().gen_range(min_tasks..max_tasks);
        let num_agents = rand::thread_rng().gen_range(min_agents..max_agents);
        let (data, costs, distances,  inc_mat) = create_random_network_flow(
            num_agents.clone(), 
            num_tasks.clone(), 
            world_size, 
            num_tasks.clone(), 
            cost_multiplyer
        )
        .unwrap();
        let (score, runtime, dist) = get_results(data, costs,distances,  inc_mat).unwrap();
        tasks.push(num_tasks as f64);
        agents.push(num_agents as f64);
        scores.push(score);
        times.push(runtime);
        true_scores.push(dist);
    }
    let trial_numbers: Vec<f64> = (0..num_trials).map(|x| x as f64).collect();

    // Display data
    let root_area = BitMapBackend::new("images/Scores and RunTimes.png", (1000, 600))
        .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let areas = root_area.split_evenly((2, 2));
    // Create ordered pairs to plot the lines
    // Score series represents the augmented distnaces used by network flow to encourage less jumping to the end
    let _score_series: Vec<(f64, f64)> = trial_numbers.clone().into_iter().zip(scores.into_iter()).collect();
    let dist_series: Vec<(f64, f64)> = trial_numbers.clone().into_iter().zip(true_scores.into_iter()).collect();
    let time_series = trial_numbers.clone().into_iter().zip(times.into_iter()).collect();
    let task_series: Vec<(f64, f64)> = trial_numbers.clone().into_iter().zip(tasks.into_iter()).collect();
    let agent_series: Vec<(f64, f64)> = trial_numbers.into_iter().zip(agents.into_iter()).collect();
    let data_series = [dist_series, time_series, task_series, agent_series];
    // Setup the extra data for plotting
    let color = &DEEPPURPLE;
    let labels = ["Distance", "RunTimes", "Tasks", "Agents"];
    // Plot
    for (i, area) in areas.iter().enumerate(){
        let mut chart = ChartBuilder::on(area)
            .margin(50)
            .caption(format!("{}", labels[i]), ("sans-serif", 20).into_font())
            .x_label_area_size(5)
            .y_label_area_size(30)
            .build_cartesian_2d(0.0..num_trials as f64, 0.0..data_series[i].iter().map(|&(_, y)| y).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(10.0))
            .unwrap();

        chart.configure_mesh()
            .draw()
            .unwrap();

        chart.draw_series(LineSeries::new(
            data_series[i].clone(),
            color
        )).unwrap();
    }
    root_area.present().unwrap();
}