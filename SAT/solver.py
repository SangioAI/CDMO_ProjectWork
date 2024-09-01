
from model_std import *
from model_seq import *
import os
import sys
import numpy as np
import time
import argparse
import csv
import json

INSTANCES_PATH = "./Instances/"
OUTPUT_PATH = "./res/SAT/"

models = [("base", std_model),
          ("base_sb", std_model),
          ("base_new_UB", std_model),
          ("base_new_UB_sb", std_model),
          ("base_linear", std_model),
          ("base_linear_sb", std_model),
          ("base_new_UB_linear", std_model),
          ("base_new_UB_linear_sb", std_model),
          ("seq", seq_model),
          ("seq_sb", seq_model),
          ("seq_new_UB", seq_model),
          ("seq_new_UB_sb", seq_model),
        ]
b_models = [
          ("base_sb", std_model),
          ("base_new_UB_sb", std_model),
          ("seq", seq_model),
          ("seq_new_UB", seq_model),
          ("seq_new_UB_sb", seq_model),
        ]

def run_model(file_path, model, search, sym, UB, partitionUB=False):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Data
    m = int(lines[0].strip())  # couriers
    n = int(lines[1].strip())  # obj
    l = list(map(int, lines[2].strip().split()))  # loads
    s = list(map(int, lines[3].strip().split()))  # size
    
    # Distances
    D = []
    for i in range(4, 4 + n + 1):
        D.append(list(map(int, lines[i].strip().split())))
    
    #print(m,n,l,s,D)

    return model(m, n, l, s, D, search=search, symmetry_breaking=sym, UB=UB, partitionUB=partitionUB)

def run_diff_models(instance):
    dic = {}

    for n, m in models:

        sym = True if ('sb' in n) else False
        UB = True if ('UB' in n) else False
        search_strategy = 'Linear' if ('seq' in n  or 'linear' in n) else 'Binary'

        print(f"\nMODEL:{n}, SEARCH:{search_strategy}, SYM: {sym}, UB:{UB}")
        res, time, route = run_model(file_path=instance,model=m, search=search_strategy, sym=sym, UB=UB)

        model_dict = {"time": time, "optimal": (time < 300), "obj": res, "sol": route}

        dic[n] = model_dict
        print(f"Finished running model {n}")


    return dic

def run_best_models(instance):
    dic = {}
    for n, m in b_models:

        sym = True if ('sb' in n) else False
        UB = True if ('UB' in n) else False
        search_strategy = 'Linear' if ('seq' in n  or 'linear' in n) else 'Binary'

        print(f"\nMODEL:{n}, SEARCH:{search_strategy}, SYM: {sym}, UB:{UB}")
        res, time, route = run_model(file_path=instance,model=m, search=search_strategy, sym=sym, UB=UB)

        model_dict = {"time": time, "optimal": (time < 300), "obj": res, "sol": route}

        dic[n] = model_dict
        print(f"Finished running model {n}")


    return dic

def save_dict_to_csv(dic, folder, n, file_type='csv'):
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"res{n}.{file_type}")

    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Time', 'Optimal', 'Objective']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for model_name, model_data in dic.items():
            writer.writerow({
                'Model': model_name,
                'Time': model_data['time'],
                'Optimal': model_data['optimal'],
                'Objective': model_data['obj']
            })
    print(f"File CSV '{file_path}' creato o sovrascritto con successo.")


def main():

    # instances_folder = 'c:/Users/andre/Desktop/AlmaMater/Optimization/Exercise/SAT/SAT/instances'
    # csv_folder = 'c:/Users/andre/Desktop/AlmaMater/Optimization/Exercise/SAT/SAT/output'
    MODEL = seq_model
    UB = True
    SEARCH = 'Linear'
    SYM = True

    parser = argparse.ArgumentParser()

    parser.add_argument("-A", "--run_all", help="Run all solvers and modalities at once", action='store_true')
    parser.add_argument("-nosym", "--no_symmetries", help="Use no symmetries breaking constraints", action='store_true')
    parser.add_argument("-l", "--linear", help="Linear search", action='store_true')
    parser.add_argument("-s", "--std", help="Sequential model", action='store_true')
    parser.add_argument("-b", "--binary", help="Binary search", action='store_true')
    parser.add_argument("-part", "--partition", help="Use solver-based Partitioner to find a better UpperBound in the pre-processing step", action='store_true')
    parser.add_argument("-u", "--upper", help="Use simple and common upper bound", action='store_true')
    parser.add_argument("-o", "--output_dir",help="Directory where the output will be saved", default=OUTPUT_PATH, type= str)
    parser.add_argument("-i", "--input_dir",help="Directory where the instance txt files can be found",default=INSTANCES_PATH, type= str)
    parser.add_argument("-n", "--num_instance",help="Select the instance that you want to solve, default = 0 solve all",default=0, type= int)
    parser.add_argument("-m", "--main_models",help="Run just best models", action='store_true')



    args = parser.parse_args()

    fileNames = sorted([args.input_dir+dat for dat in os.listdir(args.input_dir) if dat.endswith("dat")])
    outfileNames = sorted([args.output_dir+str(i+1).rjust(2,'0')+".json" for i in range(len(fileNames))])

    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)    

    if args.num_instance > 0:
        print("1")
        fileNames = fileNames[args.num_instance-1]
        outfileNames = outfileNames[args.num_instance-1] # TODO        
     
    
        if args.main_models:
            result = run_best_models(fileNames)
            with open(outfileNames, 'w') as file:
                json.dump(result, file)
        elif args.run_all:
            result = run_diff_models(fileNames)
            with open(outfileNames, 'w') as file:
                json.dump(result, file)
        else:
            if args.std:
                MODEL = std_model
                if args.binary:
                    SEARCH = 'Binary'
            if args.upper:
                UB = False
            if args.no_symmetries:
                SYM = False
            
            r_obj, r_time, r_r = run_model(fileNames, model=MODEL, search=SEARCH, sym=SYM, UB=UB, partitionUB=args.partition)
            data = {"time": r_time, "optimal": (r_time < 300), "obj": r_obj, "sol": r_r}

            with open(outfileNames, 'w') as file:
                json.dump(data, file)
            #save_dict_to_csv(results, outfileNames, 1)
    else:
        num = 1
        for fileName, outfileName in zip(fileNames, outfileNames):
            print(f" doing {num}...")
            num += 1
            if args.main_models:
                result = run_best_models(fileName)
                with open(outfileName, 'w') as file:
                    json.dump(result, file)
            elif args.run_all:
                result = run_diff_models(fileName)
                with open(outfileName, 'w') as file:
                    json.dump(result, file)
            else:
                if args.std:
                    MODEL = std_model
                    if args.binary:
                        SEARCH = 'Binary'
                if args.upper:
                    UB = False
                if args.no_symmetries:
                    SYM = False
                
                r_obj, r_time, r_r = run_model(fileName, model=MODEL, search=SEARCH, sym=SYM, UB=UB, partitionUB=args.partition)
                data = {"time": r_time, "optimal": (r_time < 300), "obj": r_obj, "sol": r_r}
                print(data)

                with open(outfileName, 'w') as file:
                    json.dump(data, file)
            
            




    # instance_files = [f for f in os.listdir(args.instances_folder) if f.endswith('.dat')]

    # for instance in instance_files:
    #     instance_path = os.path.join(args.instances_folder, instance)
    #     print(f"Processing {instance_path}...")
    #     results = run_diff_models(instance_path)
    #     save_dict_to_csv(results, args.csv_file)
    #     print(results)

    # instance_files = [f for f in os.listdir(instances_folder) if f.endswith('.dat')]

    # for n,instance in enumerate(instance_files):
    #     instance_path = os.path.join(instances_folder, instance)
    #     print("\n-----o-----o-----o-----o-----o-----o-----o")
    #     print(f"\nProcessing number {n} please wait...")
    #     results = run_diff_models(instance_path)
    #     save_dict_to_csv(results, csv_folder, n+1)
        #print(results)

    #result = run_diff_models(args.json_file)
    #save_dict_to_csv(result, args.csv_file)
    #print(result)


if __name__ == "__main__":
    main()