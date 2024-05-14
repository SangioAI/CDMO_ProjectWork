# CDMO_ProjectWork

## build command
docker build . -f Dockerfile -t cdmo:exam

## run command
docker run --rm -v "/Users/utenteadmin/CDMO_ProjectWork":/exam -it cdmo:exam