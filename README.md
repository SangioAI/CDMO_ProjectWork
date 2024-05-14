# CDMO_ProjectWork

## build command
docker build . -f Dockerfile -t cdmo:exam

## run command
docker run --rm -v "$(pwd)":/exam -it cdmo:exam