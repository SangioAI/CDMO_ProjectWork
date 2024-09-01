import numpy as np
import math
import sys
import os
import json
import argparse
from amplpy import AMPL, modules

from model import *


INSTANCES_PATH = "./Instances/"
OUTPUT_PATH = "./res/MIP/"

solvers = ["highs", "cbc", "gurobi", "cplex"]

'''
TO CHOOSE THE MODEL AND THE SOLVER type:   -m
1:
    ("highs", model)
2:
    ("cbc", model)
3:
    ("gurobi", model)
4:
    ("cplex", model)

'''



def run_model_on_instance(model, file, solver, symmetry_breaking=True):
 
    # extract data from .dat file
    with open(file) as f:
        m = int(next(f))
        n = int(next(f))
        l = [int(e) for e in next(f).split()]
        s = [int(e) for e in next(f).split()]
        D_matrix = np.genfromtxt(f, dtype=int)

    if symmetry_breaking:
        # sort the list of loads, keeping the permutation used for later
        L = [(l[i], i) for i in range(m)]
        L.sort(reverse=True)
        l, permutation = zip(*L)
        l = list(l)
        permutation = list(permutation)

    # flatten the adjacency matrix to a list in order to feed it to the model
    D = np.ravel(D_matrix).tolist()

    ampl = AMPL()

    # load model
    ampl.eval(model)

    # load parameters
    ampl.param["m"] = m
    ampl.param["n"] = n

    ampl.param["capacity"] = l
    ampl.param["size"] = s
    ampl.param["D"] = D


    # specify the solver to use and set timeout
    ampl.option["solver"] = solver
    if solver != "cplex":
        ampl.option[f"{solver}_options"] = "timelim=300"
    else:
        ampl.option[f"{solver}_options"] = "time=300"
    ampl.solve()

    solve_result = ampl.get_value("solve_result")
    time = math.floor(ampl.getValue('_total_solve_time'))

    if time >= 300:
        optimal = False
        time = 300
    else:
        optimal = True

    obj_value = int(round(ampl.get_objective('Obj_function').value(), 0))
    if solve_result == "infeasible":
        return {"time": time, "optimal": optimal, "obj": "UNSAT", "sol": []}

    elif obj_value == 0:    # No solution found, timeout
        return {"time": 300, "optimal": False, "obj": "N/A"}

    # solution
    df = ampl.get_variable("A").get_values().to_list()
    A = [[[-1 for j in range(n+1)] for k in range(n+1)] for i in range(m)]
    for i, j, k, value in df:
        i, j, k, value = int(i), int(j), int(k), int(round(value, 0))
        if symmetry_breaking:
            # also reorder couriers w.r.t permutation
            A[permutation[i-1]][j-1][k-1] = value
        else:
            A[i-1][j-1][k-1] = value

    # retrieve solution
    sol = []
    for i in range(m):
        route = []
        if A[i][n][n] == 1:
            pass
        else:
            v = A[i][n].index(1)
            while v != n:
                route.append(v+1)
                v = A[i][v].index(1)
        sol.append(route)

    return {"time": time, "optimal": optimal, "obj":obj_value, "sol": sol}


def run_mip(instance_file):

    # load solvers
    # modules.install(solvers)
    # modules.activate("1215f5bf-778d-416e-be3d-d604b4b99873")

    dictionary = {}

    for model_name, model in models:
        sym_break = False if "no_sym_break" in model_name else True
        solver = model_name.split('_')[0]

        # suppress solver output
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            model_dict = run_model_on_instance(model, instance_file, solver, symmetry_breaking=sym_break)
        except:
            sys.stdout = old_stdout
            print("There was an exception while running the model/retrieving solution")
            exit(1)
        finally:
            sys.stdout = old_stdout

        dictionary[model_name] = model_dict
        print(f"Finished running model {model_name}")


    return dictionary

parser = argparse.ArgumentParser()

parser.add_argument("-A", "--run_all", help="Run all solvers and modalities at once", action='store_true')   
parser.add_argument("-n", "--num_instance",help="Select the instance that you want to solve, default = 0 solve all",default=0, type= int)

parser.add_argument("-m", "--model", help="Decide the model", default=0, type= int)
parser.add_argument("-i", "--input_dir",help="Directory where the instance txt files can be found",default=INSTANCES_PATH, type= str)
parser.add_argument("-o", "--output_dir",help="Directory where the output will be saved", default=OUTPUT_PATH, type= str)
args = parser.parse_args()




if args.model == 1:
    models = [("highs", model)]
elif args.model == 2:
    models = [("cbc", model)]
elif args.model == 3:
    models = [("gurobi", model)]
elif args.model == 4:
    models = [("cplex", model)]
else:
    models = [ ("highs", model),
           #("highs_no_sym_break", model_no_sym_break),
           ("cbc", model),
           ("gurobi", model),
           #("gurobi_no_sym_break", model_no_sym_break),
           ("cplex", model)
          ]
    

fileNames = sorted([args.input_dir+dat for dat in os.listdir(args.input_dir) if dat.endswith("dat")])
outfileNames = sorted([args.output_dir+str(i+1).rjust(2,'0')+".json" for i in range(len(fileNames))])

if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 

if args.num_instance > 0:
    print(f"running instance: {args.num_instance}")
    fileName = fileNames[args.num_instance-1]
    outfileName = outfileNames[args.num_instance-1] 
    dic = run_mip(fileName)
    with open(outfileName, 'w') as file:
                json.dump(dic, file)
else:
    i = 1
    print(len(models))
    for fileName, outfileName in zip(fileNames, outfileNames):
        print(f"running instance: {i}")
        i += 1

        dic = run_mip(fileName)
        with open(outfileName, 'w') as file:
                json.dump(dic, file)
    
   














# from mip import *
# import json, time, math, sys


# def courier_tour(courier): # once that I have the sol I can reconstruct the tour
#     done = False
#     index = n
#     out =[]
#     while not done:
#         for j in range(n+1): # next step
#             if tour[courier][index][j].x: 
#                 if j == n: # if it reach n (final node) done
#                     done = True
#                     break
#                 out.append(j+1) # add the node to the journey
#                 break
#         index = j
#     return out

# def print_input_info():
#     print(f'Couriers: {m}')
#     print(f'Items: {n}')
#     print(f'Couriers Load: {load}')
#     print(f'Item\'s size: {size}')
#     print('Distance matrix:')
#     for i in range(n+1):
#         for j in range(n+1):
#             print(D[i][j], end = " ")
#         print()

# file_number = sys.argv[1]

# file = open("./Instances/inst"+file_number+".dat", "r")
# lines = file.readlines()
# file.close()
# m = int(lines[0].rstrip(''))
# n = int(lines[1].rstrip(''))
# load = [int(x) for x in lines[2].split(' ')]
# size = [int(x) for x in lines[3].split(' ')] 
# D = [[int(x) for x in line.rstrip().split(' ') ]for line in lines[4:]] # distance
# o = n


# start = time.time()
# # Matrix model
# model = Model(sense=MINIMIZE, solver_name=CBC)
# model.emphasis = 1 #feasibility
# model.threads = 8
# model.preprocess = 1 #enabled

# # tour[c,i,j] == 1 ---> courier c performed movement from i to j
# tour = model.add_var_tensor((m, n+1, n+1), "tour", var_type=BINARY)

# # variable for subtours elimination
# u = [model.add_var("u[%d]" % i, var_type=INTEGER, lb=0, ub=n) for i in range(n)]


# maxDist = model.add_var("maxDist", var_type=INTEGER)
# model.objective = maxDist

# # Constraints for matrix model

# # Each item is assigned to exactly one courier
# for j in range(n):
#                         # iterating by all nodes arc and courier each one appears once                                                 
#     model += xsum([tour[c][i][j] for c in range(m) for i in range(n+1)]) == 1

# # Each courier gets only one time to the same item
# for c in range(m):
#     for i in range(n+1):
#                         # all nodes that enter in j                exit from j
#         model += xsum([tour[c][i][j] for j in range(n+1)]) == xsum([tour[c][j][i] for j in range(n+1)]) 

# # C has to move 
# for c in range(m):
#     for i in range(n+1): #(n)
#         model += tour[c][i][i] == 0

# # Each courier cannot load more than his capacity
# for c in range(m):
#                     # multiply the size times 1 if it pass and has to be lower than loads
#     model += xsum([tour[c][i][j]*size[j] for i in range(n+1) for j in range(n)]) <= load[c]

# # Initial and final destinations are the same
# for c in range(m):
#     model += xsum([tour[c][n][j] for j in range(n)]) == 1
#     model += xsum([tour[c][j][n] for j in range(n)]) == 1

# # Miller-Tucker-Zemlin formulation for subtours elimination
# for i in range(n):
#     for j in range(n):
#         for c in range(m):           
#             model += u[j] - u[i] >= 1 - n*(1 - tour[c][i][j])

# # we have that u[i] < u[j] if there is the path from i to j
# # if there is from i to j n*0 ->  u[j] - u[i] >= 1 -> u[j] >= u[i] + 1   (uj bigger)


# distList = [xsum(D[i][j]*tour[c][i][j] for i in range(n+1) for j in range(n+1)) for c in range(m)]

# for c in range(m):
#     model += maxDist >= distList[c]


# time_limit = 300
# status = model.optimize(max_seconds=time_limit)

# elapsed_time = time.time() - start

# if status == OptimizationStatus.OPTIMAL:
#     for c in range(m):
#         print(f"Courier {c}")
#         print(f"TOUR: {courier_tour(c)}")
#     print('optimal solution cost {} found'.format(model.objective_value))
# elif status == OptimizationStatus.FEASIBLE:
#     elapsed_time = time_limit
#     print('NON optimal')
#     print('feasible solution cost {} found'.format(model.objective_value))
# else:
#     elapsed_time = time_limit
#     print(status)
#     print(model.objective_bound)


        
# output_dict = {
#     "cbc":
#     {
#         "time": math.floor(elapsed_time),
#         "optimal": status == OptimizationStatus.OPTIMAL,
#         "obj": round(model.objective_value) if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE else round(model.objective_bound),
#         "sol": [courier_tour(c) for c in range(m)] if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE else []
#     }
# }
# print