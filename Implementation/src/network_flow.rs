
use std::process::Command;
use full_palette::{DEEPORANGE_800, DEEPPURPLE, GREEN_900, PINK_400};
use plotters::prelude::*;
use pyo3::prelude::*;
use pyo3::prelude::PyResult;
use pyo3::types::PyList;
use rand::Rng;
use std::fs;
use toml::Value;

use crate::tsp::create_random_network_flow;


/// Setup the virtual environment
/// Make sure Rust has access to the required python packages
pub(crate) fn setup() -> PyResult<()> {
    Python::with_gil(|py| {
        // Ensure the correct python version
        
        let operating_system = std::env::consts::OS;
         // Get the active virtual environment path using virtual env
        let virtual_env_path = std::env::var("VIRTUAL_ENV").unwrap_or_else(|_| {
            panic!("No virtual environment found. Ensure you have an active virtual environment.");
        });

        // Construct the path to the site-packages directory
        let venv_site_packages;
        if operating_system == "windows"{
            venv_site_packages = format!("{}/lib/site-packages", virtual_env_path);
        } else {
            let version_info = py.version_info();
            let py_version = format!("python{}.{}", version_info.major, version_info.minor);
            venv_site_packages = format!("{}/lib/{}/site-packages", virtual_env_path, py_version);
        }


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
        
        let python_interpreter;
        if operating_system == "windows"{
            python_interpreter = format!("{}/Scripts/python", virtual_env_path);
        } else{
            python_interpreter = format!("{}/bin/python", virtual_env_path);
        }
        let output = Command::new(python_interpreter.clone())
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
        let required_packages = vec!["gurobipy", "numpy", "matplotlib", "scipy"];
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
    rows: Vec<usize>,
    cols: Vec<usize>,
    values: Vec<i64>
) -> PyResult<(f64, f64, f64, f64)>{
    // Start Python
    pyo3::prepare_freethreaded_python();
    // Get the python file as a string
    let py_functions = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/Solvers/Network_Flow/network_flow.py"
    ));
    let (score, time, dist, const_create) = Python::with_gil(|py| {
        // Set up variables class
        // Read the string python file and pull out variables class
        let py_module = PyModule::from_code_bound(
            py, 
            py_functions, 
            "network_flow.py", 
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
        let variables_py = variables.into_py(py);
        let distances_py = distances.into_py(py);
        let rows_py = rows.into_py(py);
        let cols_py = cols.into_py(py);
        let values_py = values.into_py(py);

        // Get the function to solve
        let solver: Py<PyAny> = py_module.getattr("solve_all_constraints").unwrap().into();
        let result = solver.call1(py, (costs_py, distances_py, rows_py, cols_py, values_py, variables_py));
        result.unwrap().extract(py).unwrap()
    });
    
    Ok((score, time, dist, const_create))
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
    let mut times: Vec<Vec<f64>> = Vec::new();
    let mut true_scores: Vec<f64> = Vec::new();
    let mut tasks: Vec<f64> = Vec::new();
    let mut agents: Vec<f64> = Vec::new();
    // Generate data
    for i in 0..num_trials{
        println!("Trial {}", i + 1);
        let num_tasks;
        if min_tasks < max_tasks{
            num_tasks = rand::thread_rng().gen_range(min_tasks..max_tasks);
        }else{
            num_tasks = min_tasks;
        }
        let num_agents;
        if min_agents < max_agents{
            num_agents = rand::thread_rng().gen_range(min_agents..max_agents);
        } else{
            num_agents = min_agents;
        }
        let (data, costs, distances,  rows, cols, values, creation_time, inc_time) = create_random_network_flow(
            num_agents.clone(), 
            num_tasks.clone(), 
            world_size, 
            num_tasks.clone(), 
            cost_multiplyer
        )
        .unwrap();
        let (score, runtime, dist, const_create) = get_results(data, costs,distances, rows, cols, values).unwrap();
        let trial_times = vec![runtime, creation_time.as_secs_f64(), inc_time.as_secs_f64() ,const_create];
        tasks.push(num_tasks as f64);
        agents.push(num_agents as f64);
        scores.push(score);
        times.push(trial_times);
        true_scores.push(dist);
    }
    let trial_numbers: Vec<f64> = (0..num_trials).map(|x| x as f64).collect();

    // Display data
    let root_area = BitMapBackend::new("Images/Scores and RunTimes.png", (1000, 600))
        .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let areas = root_area.split_evenly((2, 2));
    // Create ordered pairs to plot the lines
    // Score series represents the augmented distances used by network flow to encourage less jumping to the end
    let _score_series: Vec<(f64, f64)> = trial_numbers.clone().into_iter().zip(scores.into_iter()).collect();
    let dist_series: Vec<(f64, f64)> = trial_numbers.clone().into_iter().zip(true_scores.into_iter()).collect();
    let mut time_series: Vec<Vec<(f64, f64)>> = vec![vec![], vec![], vec![], vec![]];
    
    for (i, inner_vec) in times.iter().enumerate() {
        for (j, &time) in inner_vec.iter().enumerate() {
            time_series[j].push((trial_numbers[i], time));
        }
    }
    let task_series: Vec<(f64, f64)> = trial_numbers.clone().into_iter().zip(tasks.into_iter()).collect();
    let agent_series: Vec<(f64, f64)> = trial_numbers.into_iter().zip(agents.into_iter()).collect();
    let data_series = [dist_series, time_series[0].clone(), time_series[1].clone(), time_series[2].clone(), time_series[3].clone(), task_series, agent_series];
    // Setup the extra data for plotting
    let color = &DEEPPURPLE;
    let labels = ["Distance", "Times", "Tasks", "Agents"];
    let time_labels = ["Gurobi", "Problem", "Incidence", "Constraint"];
    let colors = vec![&CYAN, &PINK_400, &GREEN_900, &DEEPORANGE_800];
    // Plot
    let mut j = 0;
    for (k, label) in labels.iter().enumerate(){
        let area = &areas[k];
        let mut chart = ChartBuilder::on(area)
            .margin(50)
            .caption(format!("{}", label), ("sans-serif", 20).into_font())
            .x_label_area_size(5)
            .y_label_area_size(30)
            .build_cartesian_2d(0.0..(num_trials - 1) as f64, 0.0..data_series[j].iter().map(|&(_, y)| y).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(10.0))
            .unwrap();

        chart.configure_mesh()
            .draw()
            .unwrap();
        if j > 0 && j < 5{
            for (i, time_label) in time_labels.iter().enumerate(){
                let cur_color = colors[i];
                chart.draw_series(LineSeries::new(
                    data_series[j].clone(),
                    cur_color.clone()
                )).unwrap()
                .label(*time_label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], cur_color.clone()
            ));
                j += 1;
            }
            chart.configure_series_labels()
                .border_style(&BLACK)
                .position(SeriesLabelPosition::UpperLeft)
                .legend_area_size(5)
                .background_style(&WHITE.mix(0.8))
                .draw()
                .unwrap();
        } else{
            chart.draw_series(LineSeries::new(
                data_series[j].clone(),
                 color
            )).unwrap();
            j += 1;
        }
    }
    root_area.present().unwrap();
}

pub(crate)fn read_toml(version: String)-> Result<(), Box<dyn std::error::Error>>{
    let toml_content = fs::read_to_string("Variables/networkFlow.toml")?;

    let val: Value = toml_content.parse()?;
    if version == "run"{
        // Get MTSP generation variables
        let mtsp_vars = val.get("mTSP").unwrap();
        let num_agents:usize = mtsp_vars.get("num_agents").unwrap().as_integer().unwrap() as usize;
        let num_tasks = mtsp_vars.get("num_tasks").unwrap().as_integer().unwrap() as usize;
        let cost_multiplyer = mtsp_vars.get("cost_multiplyer").unwrap().as_integer().unwrap() as usize;
        let world_size_array = mtsp_vars.get("world_size").unwrap().as_array().unwrap();
        let world_size: Vec<f64> = world_size_array.iter()
            .filter_map(|val| val.as_float())
            .collect();

        // Get Network Flow variables
        let network_flow_vars = val.get("NetworkFlow").unwrap();
        let num_columns = network_flow_vars.get("num_columns").unwrap().as_integer().unwrap() as usize;
        

        // Create Problem
        let (data, costs, distances,  rows, cols, values, creation_time, inc_time) = 
            create_random_network_flow(
            num_agents.clone(), 
            num_tasks.clone(), 
            (world_size[0], world_size[1]), 
            num_columns.clone(), 
            cost_multiplyer
        ).unwrap();
        let (score, time, dist, constraints_creation) = get_results(data, costs, distances, rows, cols, values).unwrap();
        println!("Objective Score: {}", score);
        println!("Gurobipy Runtime: {}", time);
        println!("Total distance traveled: {}", dist);
        println!("Problem Creation: {}", creation_time.as_secs_f64());
        println!("Incidence Creation: {}", inc_time.as_secs_f64());
        println!("Constraint Creation: {}", constraints_creation);
    } else if version == "test" {
        // Get testing variables
        let test_vars = val.get("TestMilp").unwrap();
        let num_trials = test_vars.get("num_trials").unwrap().as_integer().unwrap() as usize;
        let max_tasks = test_vars.get("max_tasks").unwrap().as_integer().unwrap() as usize;
        let min_tasks = test_vars.get("min_tasks").unwrap().as_integer().unwrap() as usize;
        let max_agents = test_vars.get("max_agents").unwrap().as_integer().unwrap() as usize;
        let min_agents = test_vars.get("min_agents").unwrap().as_integer().unwrap() as usize;

        println!("Agents range: {}-{}", min_agents, max_agents);
        println!("Tasks range: {}-{}", min_tasks, max_tasks);

        // Get world size and cost multiplyer
        let mtsp_vars = val.get("mTSP").unwrap();
        let cost_multiplyer = mtsp_vars.get("cost_multiplyer").unwrap().as_integer().unwrap() as usize;
        let world_size_array = mtsp_vars.get("world_size").unwrap().as_array().unwrap();
        let world_size: Vec<f64> = world_size_array.iter()
            .filter_map(|val| val.as_float())
            .collect();
        test_network_flow(
            num_trials,
            max_tasks, 
            min_tasks, 
            max_agents, 
            min_agents, 
            (world_size[0], world_size[1]), 
            cost_multiplyer)
    } 

    Ok(())
}