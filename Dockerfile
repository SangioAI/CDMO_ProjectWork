FROM minizinc/minizinc:latest

WORKDIR ./CDMO_Sangiorgi_Fossa

COPY . .

RUN apt-get update
RUN apt-get install -y python3.10
RUN apt-get install -y python3-pip
RUN apt-get install libffi-dev
RUN apt-get install -y python3-venv
RUN python3 -m venv venv1
RUN venv1/bin/python3 -m pip install -r requirements.txt
RUN apt-get install -y z3
RUN venv1/bin/python3 -m amplpy.modules install highs gurobi  
RUN venv1/bin/python3 -m amplpy.modules activate 1215f5bf-778d-416e-be3d-d604b4b99873

CMD venv1/bin/python3 CP/solver.py -A -g && venv1/bin/python3 SAT/solver.py -A && venv1/bin/python3 MIP/solver.py -A