import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import manifold
import minizinc as mzn
import datetime as t

def readInst(pathToFile):
    with open(pathToFile,"r") as file:
        data_raw = file.read()
        data = data_raw.split("\n")
        n = int(data[1])
        m = int(data[0])
        loads = np.ndarray((int(data[0])), dtype=int)
        for r in range(int(data[0])):
            loads[r] = [int(i) for i in re.split('\s+', data[2].strip())][r]
        sizes = np.ndarray((int(data[1])), dtype=int)
        for r in range(int(data[1])):
            sizes[r] = [int(i) for i in re.split('\s+', data[3].strip())][r]
        
        arr = np.ndarray((int(data[1])+1,int(data[1])+1), dtype=int)
        for r in range(int(data[1])+1):
            arr[r,:] = np.array([int(i) for i in re.split('\s+', data[3+r+1].strip())])
        # print(arr, arr.shape)
        return m,n,loads,sizes,arr
    
def lowerBound(D):
    ''' Lower Bound is just the minumum path traveled by one courier visiting only one item'''
    return np.min([D[-1,i]+D[i,-1] for i in range(D.shape[1]-1)])
    
def upperBound3(D,m,partition):
    ubs = []
    for k in range(m):
        idxs = np.argwhere(partition==k+1)[:,0]
        if len(idxs) == 0:
            ubs += []
            continue
        # print(f"idxs:{list(idxs)}")
        D_max = D[:,idxs][idxs,:].copy()
        D_max[D_max==0] = 1000 # avoid looping on itself
        min_route = [-1] # start from depot
        for i in range(len(idxs)):
            min_route += [np.argmin(D_max[min_route[-1],:])]
            D_max[:,min_route[1:len(min_route)]] = 1000
        # print(f"before {min_route}")
        for i in range(1,len(min_route)):
            min_route[i] = idxs[min_route[i]]
        min_route = min_route[1:] # remove starting depot
        # print(f"after {min_route}")
        _ub = D[-1,min_route[0]] 
        _ubs = [_ub]
        # print(f"[{-1}{min_route[0]}]:{D[-1,min_route[0]] }->{_ub}")
        for i in range(len(min_route)-1):
            _ub += D[min_route[i],min_route[i+1]] 
            _ubs += [_ub]
            # print(f"[{i}{min_route[i+1]}]:{D[-1,min_route[0]] }->{_ub}")
        _ub += D[min_route[-1],-1]
        _ubs += [_ub]
        # print(f"dist:{_ubs}")
        # print(f"[{i}{min_route[i+1]}]:{D[-1,min_route[0]] }->{_ub}")
        ubs += [_ub]
    # print(ubs)
    return np.max(ubs)

# def upperBound(D, loads, sizes):
#     ''' This is an heuristic. Upper Bound is the greedy (so not the real mininum) min distance 
#     needed for one courier to reach all items.'''
#     max_route = [-1] # start from depot
#     D_max = D.copy()
#     D_max[D_max==0] = 1000 # avoid looping on itself
#     l = 0
#     k = 0
#     ub = 0
#     for i in range(D.shape[0]-1): # -1 cause we don't count going back home
#         max_route += [np.argmin(D_max[max_route[-1],:-1])] # :-1 to avoid going back depot
#         D_max[:,max_route[1:len(max_route)]] = 1000
#         while k<len(loads) and l + sizes[max_route[-1]] > loads[k]:
#             # print(f"full courier {k} at {ub} with l= {l}")
#             if l!=0:
#                 ub += D[max_route[-1],-1] # add going back home - do it once
#             l = 0
#             k += 1
#         ub += D[max_route[-2],max_route[-1]]
#         l += sizes[max_route[-1]]
#         # print(max_route)
#         # print(D_max)
#     ub += D[max_route[-1],-1] # add going home
#     return ub

def upperBound(D, loads):
    ''' This is an heuristic. Upper Bound is the greedy (so not the real mininum) min distance 
    needed for one courier to reach all items.'''
    max_route = [-1]
    D_max = D.copy()
    D_max[D_max==0] = 1000
    for i in range(D.shape[0]-1): # -1 cause we don't count going back home
        max_route += [np.argmin(D_max[max_route[-1],:-1])] # :-1 to avoid going back depot
        D_max[:,max_route[1:len(max_route)]] = 1000
        # print(max_route)
        # print(D_max)
    # calculate distance of max_route
    ub = D[max_route[-1],-1] # circuit
    for i in range(len(max_route)-1):
        ub += D[max_route[i],max_route[i+1]]
    return ub

def makeSymmetric(D):
    ltri_mask = ~np.tri(D.shape[0],D.shape[1],-1).astype(bool)
    D2 = D.copy()
    D2[ltri_mask] = D2.T[ltri_mask]
    return D2

def distance2coords(D):
    mds=manifold.MDS(n_components=2, dissimilarity='precomputed')
    return mds.fit_transform(D) 

def drawSolutions(coords, paths, out_file):
    # plt.figure(figsize=(20,20))
    plt.scatter(coords[:,0],coords[:,1])
    plt.title(out_file)
    [plt.text(coords[i,0],coords[i,1],i+1) for i in range(coords.shape[0])]
    colors = cm.rainbow(np.linspace(0, 1, len(paths)))
    for k in range(len(paths)):
        sol_coords = np.array([[coords[v,0],coords[v,1]] for v in paths[k]-1])
        [plt.plot(sol_coords[i:i+2,0],sol_coords[i:i+2,1], color=colors[k]) for i in range(sol_coords.shape[0])]
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)

def successor2paths(successor, m, n):
    end_nodes = successor[-m:]
    start_nodes = successor[-2*m:-m]

    paths = np.ndarray((m,), dtype=list)
    for k in range(m):
        i = start_nodes[k]
        path_k = [n+1]

        while successor[i-1] != end_nodes[k]:
            path_k += [i]
            i = successor[i-1]
        path_k += [n+1]
        paths[k] = np.array(path_k) 
    return paths

def getOutput(result, paths, timeout):
    time = timeout
    optimal= False, 
    obj= "N/A", 
    sol= []
    
    if result is None:
        pass
    elif result.status is mzn.Status.UNSATISFIABLE:
        time = result.statistics['solveTime'].total_seconds(), 
        obj= "UNSAT", 
    elif result.status is mzn.Status.SATISFIED:
        obj = result["max_distance"]
        sol = [[int(i) for i in path[1:-1]] for path in paths]
    elif result.status is mzn.Status.OPTIMAL_SOLUTION:
        time = result.statistics['solveTime'].total_seconds() if result.statistics['solveTime'].total_seconds() < timeout else timeout
        optimal=True
        obj = result["max_distance"]
        sol = [[int(i) for i in path[1:-1]] for path in paths]

    return  {
        'time': int(time),
        'optimal': optimal,
        'obj': obj, 
        'sol': sol
    }

def heuristic2(loads, sizes):
    return loads[np.argmax(np.cumsum(loads)>np.sum(sizes))]


def heuristic1(loads, sizes):
    ll = sorted(loads)
    for l in ll:
        if l*len(loads) > np.sum(sizes):
            break
    return l

def appendJson(jsonFrom, jsonTo):
    for i in jsonFrom: 
        jsonTo[i]=jsonFrom[i]

def preprocess(mzn_instance, m,n,l,s):
    mzn_instance["m"] = m
    mzn_instance["n"] = n
    mzn_instance["l"] = l
    mzn_instance["s"] = s
    result = mzn_instance.solve(timeout=t.timedelta(seconds=5), random_seed=42, processes=1, optimisation_level=1, statistics=True)
    # print(result.solution, result.statistics, result.status)
    return np.array(result["bins"])
