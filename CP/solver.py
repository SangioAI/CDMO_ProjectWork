import minizinc as mzn
import datetime as t
from utils import *
import time  as tm
import os
import json
import numpy as np

import argparse

PREP_MZN_FILE = "CP/bin_pack.mzn"
MZN_FILE = "CP/mcp.mzn"
MZN_2R_FILE = "CP/mcp_2R.mzn"
MZN_3R_FILE = "CP/mcp_3R.mzn"
MZN_D_FILE = "CP/mcp_D.mzn"
CP_SOLVERS = ["gecode", "chuffed"]
SYM_STRING = "_symbreak"
HEU_STRING = "_heuristics"
INSTANCES_PATH = "./Instances/"
OUTPUT_PATH = "./res/CP/"
OUTPUT_IMAGES_PATH = "./Output_Images/CP/"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-A", "--run_all", help="Run all solvers and modalities at once", action='store_true')
    parser.add_argument("-mzn", "--mzn_file", help="Select mzn file to load",default=MZN_FILE, type= str)
    parser.add_argument("-pmzn", "--prep_mzn", help="Select pre-processing mzn file to load",default=PREP_MZN_FILE, type= str)
    parser.add_argument("-heu", "--heuristic", help="Use heuristics", action='store_true')
    parser.add_argument("-part", "--partition", help="Use solver-based Partitioner to find a better UpperBound in the pre-processing step", action='store_true')
    parser.add_argument("-nosym", "--no_symmetries", help="Use no symmetries breaking constraints", action='store_true')
    parser.add_argument("-nored", "--no_reduntants", help="Use no redundant constraints", action='store_true')
    parser.add_argument("-g", "--graph", help="Select whether to visualizea solution graph", action='store_true')
    parser.add_argument("-s", "--solver", help="Select the solver", default='gecode', type= str)
    parser.add_argument("-t", "--timeout", help="Timeout in seconds", default=300, type= int)
    parser.add_argument("-n", "--num_instance",help="Select the instance that you want to solve, default = 0 solve all",default=0, type= int)
    parser.add_argument("-o", "--output_dir",help="Directory where the output will be saved", default=OUTPUT_PATH, type= str)
    parser.add_argument("-oi", "--output_dir_images",help="Directory where the output images for solutions will be saved", default=OUTPUT_IMAGES_PATH, type= str)
    parser.add_argument("-i", "--input_dir",help="Directory where the instance txt files can be found",default=INSTANCES_PATH, type= str)

    args = parser.parse_args()
    
    fileNames = sorted([args.input_dir+dat for dat in os.listdir(args.input_dir) if dat.endswith("dat")])
    outfileNames = sorted([args.output_dir+str(i+1).rjust(2,'0')+".json" for i in range(len(fileNames))])
    outImageNames = sorted([args.output_dir_images+str(i+1).rjust(2,'0')+".jpg" for i in range(len(fileNames)) if args.output_dir_images is not None])
    if args.num_instance > 0:
        fileNames = [fileNames[args.num_instance-1]]
        outfileNames = [outfileNames[args.num_instance-1]]
        outImageNames = [outImageNames[args.num_instance-1]]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir_images):
        os.makedirs(args.output_dir_images)

    json_outputs = []
    if args.run_all:
        j_out = solve(MZN_FILE, args.prep_mzn, fileNames, outfileNames, noRed=False, noSym=True, solvers=CP_SOLVERS, timeout=args.timeout, heuristics=False, graph=args.graph, partitionUB=args.partition, outImageNames=outImageNames, addName="")
        json_outputs += j_out 
        j_out = solve(MZN_FILE, args.prep_mzn, fileNames, outfileNames, noRed=False, noSym=False, solvers=CP_SOLVERS, timeout=args.timeout, heuristics=False, graph=args.graph, partitionUB=args.partition, outImageNames=outImageNames, addName="")
        for i in range(len(json_outputs)): appendJson(j_out[i], json_outputs[i]) 
        j_out = solve(MZN_FILE, args.prep_mzn, fileNames, outfileNames, noRed=False, noSym=False, solvers=CP_SOLVERS, timeout=args.timeout, heuristics=True, graph=args.graph, partitionUB=args.partition, outImageNames=outImageNames, addName="")
        for i in range(len(json_outputs)): appendJson(j_out[i], json_outputs[i])  
        # j_out = solve(MZN_3R_FILE, args.prep_mzn, fileNames, outfileNames, noRed=True, noSym=True, solvers=CP_SOLVERS, timeout=args.timeout, heuristics=False, graph=args.graph, partitionUB=args.partition, outImageNames=outImageNames, addName="_3R")
        # for i in range(len(json_outputs)): appendJson(j_out[i], json_outputs[i]) 
        # j_out = solve(MZN_3R_FILE, args.prep_mzn, fileNames, outfileNames, noRed=False, noSym=False, solvers=CP_SOLVERS, timeout=args.timeout, heuristics=False, graph=args.graph, partitionUB=args.partition, outImageNames=outImageNames, addName="_3R")
        # for i in range(len(json_outputs)): appendJson(j_out[i], json_outputs[i]) 
        j_out = solve(MZN_2R_FILE, args.prep_mzn, fileNames, outfileNames, noRed=False, noSym=False, solvers=CP_SOLVERS, timeout=args.timeout, heuristics=True, graph=args.graph, partitionUB=args.partition, outImageNames=outImageNames, addName="_2R")
        for i in range(len(json_outputs)): appendJson(j_out[i], json_outputs[i]) 
        j_out = solve(MZN_3R_FILE, args.prep_mzn, fileNames, outfileNames, noRed=False, noSym=False, solvers=CP_SOLVERS, timeout=args.timeout, heuristics=True, graph=args.graph, partitionUB=args.partition, outImageNames=outImageNames, addName="_3R")
        for i in range(len(json_outputs)): appendJson(j_out[i], json_outputs[i]) 
        j_out = solve(MZN_D_FILE, args.prep_mzn, fileNames, outfileNames, noRed=False, noSym=False, solvers=["chuffed"], timeout=args.timeout, heuristics=True, graph=args.graph, partitionUB=args.partition, outImageNames=outImageNames, addName="_D")
        for i in range(len(json_outputs)): appendJson(j_out[i], json_outputs[i]) 
    else:
        json_outputs = solve(args.mzn_file, args.prep_mzn, fileNames, outfileNames, 
            noRed=args.no_reduntants, 
            noSym=args.no_symmetries,
            solvers=[args.solver],
            timeout=args.timeout,
            heuristics=args.heuristic,
            graph=args.graph,
            partitionUB=args.partition,
            outImageNames=outImageNames)

    for i in range(len(outfileNames)):
        with open(outfileNames[i], 'w') as file:
            json.dump(json_outputs[i], file)


def solve(mzn_file, prep_mzn_file, data_files, output_files, solvers=["gecode"], noSym=False, noRed=False, timeout=300, heuristics=False, partitionUB=False, graph=False, outImageNames=None, addName = None):
        prep_model = mzn.Model(prep_mzn_file) 
        model = mzn.Model(mzn_file) 
        model["mzn_ignore_symmetry_breaking_constraints"] = noSym
        model["mzn_ignore_redundant_constraints"] = noRed

        json_outputs = []
        for i in range(len(data_files)):
            json_output = {}
            for solverName in solvers:
                prep_solver = mzn.Solver.lookup(solverName)
                mzn_prep_instance = mzn.Instance(prep_solver, prep_model)
                solver = mzn.Solver.lookup(solverName)
                solverName = solverName + addName if addName is not None else solverName
                solverName = solverName + SYM_STRING if not noSym else solverName
                solverName = solverName + HEU_STRING if heuristics else solverName
                print(f"{data_files[i]} ({solverName})-------------------")

                mzn_instance = mzn.Instance(solver, model)
                                
                m, n, l, s, D = readInst(data_files[i])
                print("Preprocessing...")
                
                
                start_time = tm.time()
                UB = 0
                if partitionUB:
                    partition = preprocess(mzn_prep_instance,m,n,l,s)
                    UB = upperBound3(D,m,partition)
                else :
                    UB = upperBound(D,l)
                l_idx = np.argsort(l)[::-1]
                print(l_idx, np.argsort(l_idx))
                l = l[l_idx]
                LB = lowerBound(D)
                HEU = heuristic1(l, s) if heuristics else 10000
                end_time = tm.time()
                time = end_time-start_time
                print("...Done in {time:1.3f}s".format(time=time))
                print(f"m:{m} n:{n} l:{l} s:{s} D:{D}")
                print(f"UpperBound:{UB} LowerBound:{LB} HEU:{HEU}")

                mzn_instance["m"] = m
                mzn_instance["n"] = n
                mzn_instance["l"] = l
                mzn_instance["s"] = s
                mzn_instance["D"] = D
                mzn_instance["HEU"] = HEU
                mzn_instance["UB"] = UB
                mzn_instance["LB"] = LB
                
                print("Start Solving...")
                output_dict = getOutput(None, None, timeout)
                try:
                    start_time = tm.time()
                    result = mzn_instance.solve(timeout=t.timedelta(seconds=timeout), random_seed=42, processes=1, optimisation_level=1, statistics=True)
                    end_time = tm.time()
                    time = end_time-start_time

                    print(result.solution, result.statistics, result.status)
                    paths = successor2paths(result["successor"], m, n)
                    paths = paths[np.argsort(l_idx)] # restore correct order of solution
                    output_dict = getOutput(result, paths, timeout)
                    print("...End Solving in {time:1.3f}s".format(time=time))
                    print(f"Paths:{paths}")
                    print(f"Obj:{result['max_distance']}")
                    if graph:
                        D_sym = makeSymmetric(D)
                        coords = distance2coords(D_sym)
                        drawSolutions(coords, paths, outImageNames[i])
                except Exception as e:
                    print(f"Exception solving it {e}")  

                json_output[solverName] = output_dict
            json_outputs += [json_output]
        return json_outputs

if __name__ == '__main__':
    main()