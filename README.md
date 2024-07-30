# CDMO_ProjectWork

## Build Docker command
docker build . -f Dockerfile -t cdmo:exam

## Run Docker command
docker run --rm -v "$(pwd)":/exam -it cdmo:exam

## Commit Docker changes
docker commit -m "commit_message" CONTAINER_ID cdmo:exam

## Retrieve Docker running container Id
docker container ls

## Convert .dat files into .dzn files
python inst_converter.py
