# CDMO_ProjectWork
This project 

## Using Docker
Follow the step below to use Docker.
### Install Docker
Follow the installation step here: https://www.docker.com/products/docker-desktop/

### Build Docker command
Build the image of the docker:
`docker build . -f Dockerfile -t cdmo_mcp`

### Run Docker command
To directly run all the solvers on all the instances:
`docker run -it cdmo_mcp`

### Run Bash Docker command
To open an interactive shell:
`docker run --rm -it cdmo_examm /bin/bash`


## Python
Python in docker via venv

## CP
To run CP solver you can either:
- run all configurations on all istances:
`python solver.py -A -g -t 300`

- run all configurations on specific instances:
`python solver.py -A -g -t 300 -n 12`

- run specific configurations on specific instances:
`python solver.py -mzn "CP/mcp_3R.mzn" -heu -g -t 300 -n 12`

## Solution check
To check if a solution is consistent with problem data use:
`python check_solution.py Instances/ Output/`