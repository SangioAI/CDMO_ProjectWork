# CDMO_ProjectWork
Combinatorial Decision Making and Optimization project of the Artificial Intelligence course of study at the University of Bologna.

This project contains different approaches to solve the Multiple Courier Planning problem (MCP) using Constraint Programming (CP), SAT and Mixed-Integer Programming (MIP) strategies. The results of different optimization approaches exploiting different formulations and underling solvers are discussed in the pdf report.

## Docker Usage
Follow the step below to use Docker:
- Install Docker from here: https://www.docker.com/products/docker-desktop/
- Build the image of the docker: `docker build . -t cdmo_mcp`
- Run All solvers on all the instances: `docker run -it cdmo_mcp`
- Open an Interactive Bash in Docker: `docker run --rm -it cdmo_mcp /bin/bash`

> [!IMPORTANT]
> In the interactive Bash activate virtual envirorment python to run solvers :
> 
> `source venv1/bin/activate`
> 
> When you see `(venv1)` in the interactive bash, you are ready to go!

## Solver Usage
### CP Solvers
To run CP solver (`CP/solver.py`) accept the following arguments (listed by: `python CP/solver.py -h`):
```
usage: solver.py [-h] [-A] [-mzn MZN_FILE] [-pmzn PREP_MZN] [-heu] [-part] [-nosym] [-nored] [-g] [-s SOLVER] [-t TIMEOUT] [-n NUM_INSTANCE] [-o OUTPUT_DIR] [-oi OUTPUT_DIR_IMAGES]
                 [-i INPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -A, --run_all         Run all solvers and modalities at once
  -mzn MZN_FILE, --mzn_file MZN_FILE
                        Select mzn file to load
  -pmzn PREP_MZN, --prep_mzn PREP_MZN
                        Select pre-processing mzn file to load
  -heu, --heuristic     Use heuristics
  -part, --partition    Use solver-based Partitioner to find a better UpperBound in the pre-processing step
  -nosym, --no_symmetries
                        Use no symmetries breaking constraints
  -nored, --no_reduntants
                        Use no redundant constraints
  -g, --graph           Select whether to save an image of the solution graph
  -s SOLVER, --solver SOLVER
                        Select the solver
  -t TIMEOUT, --timeout TIMEOUT
                        Timeout in seconds
  -n NUM_INSTANCE, --num_instance NUM_INSTANCE
                        Select the instance that you want to solve, default = 0 solve all
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory where the output will be saved
  -oi OUTPUT_DIR_IMAGES, --output_dir_images OUTPUT_DIR_IMAGES
                        Directory where the output images for solutions will be saved
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Directory where the instance txt files can be found
```

To run CP solver you can either:
- run all configurations on all istances:
`python CP/solver.py -A -g -t 300`

- run all configurations on specific instances:
`python CP/solver.py -A -g -t 300 -n 12`

- run specific configurations on specific instances:
`python CP/solver.py -mzn "CP/mcp_3R.mzn" -part -heu -g -t 300 -n 12`

Available CP models are:
- `CP/mcp.mzn`: base model
- `CP/mcp_2R.mzn`: base model + Relax&Reconstruct
- `CP/mcp_3R.mzn`: base model + Relax&Reconstruct + Restart
- `CP/mcp_D.mzn`: base model with Decomposition

## SAT solver
To run SAT solver (`SAT/solver.py`) accept the following arguments (listed by: `python SAT/solver.py -h`):
```
usage: solver.py [-h] [-A] [-nosym] [-l] [-s] [-b] [-part] [-u] [-o OUTPUT_DIR] [-i INPUT_DIR] [-n NUM_INSTANCE] [-m]

optional arguments:
  -h, --help            show this help message and exit
  -A, --run_all         Run all solvers and modalities at once
  -nosym, --no_symmetries
                        Use no symmetries breaking constraints
  -l, --linear          Linear search
  -s, --std             Sequential model
  -b, --binary          Binary search
  -part, --partition    Use solver-based Partitioner to find a better UpperBound in the pre-processing step
  -u, --upper           Use simple and common upper bound
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory where the output will be saved
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Directory where the instance txt files can be found
  -n NUM_INSTANCE, --num_instance NUM_INSTANCE
                        Select the instance that you want to solve, default = 0 solve all
  -m, --main_models     Run just best models
```
## MIP solver
To run MIP solver (`MIP/solver.py`) accept the following arguments (listed by: `python MIP/solver.py -h`):
```
usage: solver.py [-h] [-A] [-n NUM_INSTANCE] [-m MODEL] [-i INPUT_DIR] [-o OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -A, --run_all         Run all solvers and modalities at once
  -n NUM_INSTANCE, --num_instance NUM_INSTANCE
                        Select the instance that you want to solve, default = 0 solve all
  -m MODEL, --model MODEL
                        Decide the model
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Directory where the instance txt files can be found
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory where the output will be saved
```
## Solution Checker Usage
To check if a solution is consistent with problem data use:
`python check_solution.py Instances/ Output/`
