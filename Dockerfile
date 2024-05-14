FROM minizinc/minizinc:latest

WORKDIR exam/
#COPY . .

RUN apt-get update
RUN apt-get install -y z3 minizinc

# useless for exam
RUN apt-get install -y git
