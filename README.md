# CDMO_ProjectWork

## Build Docker command
docker build . -f Dockerfile -t cdmo:exam

## Run Docker command
docker run --rm -v "$(pwd)":/exam -it cdmo:exam

## Convert .dat files into .dzn files
python inst_converter.py