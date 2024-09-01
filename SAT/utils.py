import math
from z3 import *
import numpy as np
import minizinc as mzn
import datetime as t

def millisecs_left(t, timeout):
    """returns the amount of milliseconds left from t to timeout

    Args:
        t (int): timestamp
        timeout (int): timestamp of the timeout

    Returns:
        int: the milliseconds left
    """
    return int((timeout - t) * 1000)

def flatten(matrix):
    """flattens a 2D list into a 1D list

    Args:
        matrix (list[list[Object]]): the matrix to flatten

    Returns:
        list[Object]: the flattened 1D list
    """
    return [e for row in matrix for e in row]


## binary conversions

def num_bits(x):
    """Returns the number of bits necessary to represent the integer x

    Args:
        x (int): the input integer

    Returns:
        int: the number of bits necessary to represent x
    """
    return math.floor(math.log2(x)) + 1

def int_to_bin(x, digits):
    """Converts an integer x into a binary representation of True/False using #bits = digits

    Args:
        x (int): the integer to convert
        digits (int): the number of bits of the output

    Returns:
        list[bool]: the binary representation of x
    """
    x_bin = [(x%(2**(i+1)) // 2**i)==1 for i in range(digits-1,-1,-1)]
    return x_bin

def bin_to_int(x):
    """Converts a binary number of 1s and 0s into its integer form

    Args:
        x (list[int]): the binary number to convert

    Returns:
        int: the converted integer representation of x
    """
    n = len(x)
    x = sum(2**(n-i-1) for i in range(n) if x[i] == 1)
    return x


## evaluate model variables

def evaluate(model, bools):
    """Evaluate every element of bools using model recursively

    Args:
        model (ModelRef): the model to evaluate on
        bools (n-dim list[Bool]): the bools to evaluate, can be of arbitrary dimension

    Returns:
        n-dim list[int]: object of the same dimensions of bools, with a 1 in the corresponding position of 
                         the bools that evaluated to true w.r.t. model
    """
    if not isinstance(bools[0], list):
        return [1 if model.evaluate(b) else 0 for b in bools]
    return [evaluate(model, b) for b in bools]

def retrieve_routes(orders, assignments):
    """Returns for each courier, the list of items that he must deliver, in order of delivery

    Args:
        orders (list[list[bool]]): matrix representing the order of delivery of each object 
                                   in its route, namely orders[j][k] == True iff object j is delivered as k-th by its courier 
        assignments (list[list[bool]]): matrix of assignments, assignments[i][j] = True iff courier i delivers
                                        object j, false otherwise.
    """
    m = len(assignments)
    n = len(assignments[0])
    routes = [[0 for j in range(n)] for i in range(m)]
    for node in range(n):
        for time in range(n):
            if orders[node][time]:
                for courier in range(m):
                    if assignments[courier][node]:
                        routes[courier][time] = node+1
                        break
                break

    routes = [[x for x in row if x != 0] for row in routes] # remove trailing zeros
    return routes

#
# TO CHECK HAM CYCLE
#

def check_all_hamiltonian(tensor):
    """Function to check that all the paths represented in tensor are hamiltonian cycles

    Args:
        tensor (list[list[list[bool]]]): list of adjacency matrices over 2-regular graphs

    Returns:
        bool: true iff all paths in tensor are hamiltonian cycles, false otherwise
    """
    m = len(tensor)
    for i in range(m):
        if not check_hamiltonian(tensor[i]):
            return False
    return True


def check_hamiltonian(matrix):
    """Function to check that the given adjancency matrix over 2-regular graph is a hamiltonian cycle, i.e. the graph is connected

    Args:
        matrix (list[list[bool]]): adjacency matrix over 2-regular graph

    Returns:
        bool: true iff the given adjancency matrix is a hamiltonian cycle, false otherwise
    """
    n = len(matrix)
    visited = [0] * n
    v = n - 1
    while visited[v] == 0:
        visited[v] = 1
        for k in range(n):
            if matrix[v][k] == True:
                v = k
                break
    num_vertices = sum(row.count(True) for row in matrix)
    return (sum(visited) == num_vertices)


#
# FOR VISUALIZATION
#

def displayMCP(orders, distances_bin, obj_value, assignments):
    """Function to display a found solution of the Multiple Couriers Planning problem

    Args:
        orders (list[list[bool]]): matrix representing the order of delivery of each object 
                                   in its route, namely orders[j][k] == True iff object j is delivered as k-th by its courier 
        distances_bin (list[list[bool]]): for each courier, its travelled distance represented in binary
        obj_value (int): the objective value obtained
        assignments (list[list[bool]]): matrix of assignments, assignments[i][j] = True iff courier i delivers
                                        object j, false otherwise.
    """
    distances = [bin_to_int(d) for d in distances_bin]

    print(f"-----------Objective value: {obj_value}-----------")
    print(f"------------------Routes-----------------")
    m = len(assignments)
    n = len(assignments[0])
    routes = [[0 for j in range(n)] for i in range(m)]
    for node in range(n):
        for time in range(n):
            if orders[node][time]:
                for courier in range(m):
                    if assignments[courier][node]:
                        routes[courier][time] = node+1
                        break
                break

    routes = [[x for x in row if x != 0] for row in routes] # remove trailing zeros
    for courier in range(m):
        print("Origin --> " +
              ' --> '.join([str(node) for node in routes[courier]]) +
              f' --> Origin: travelled {distances[courier]}')
        
#
# FIND UPPER BOUND
#

def upperBound(D, loads, sizes):
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

def upperBound3(DD,m,partition):
    ubs = []
    D = np.array(DD)
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
    return int(np.max(ubs))

def getMznInstance(prep_mzn_file):
    prep_model = mzn.Model(prep_mzn_file) 
    prep_solver = mzn.Solver.lookup("gecode")
    return mzn.Instance(prep_solver, prep_model)

def preprocess(mzn_instance, m,n,l,s):
    mzn_instance["m"] = m
    mzn_instance["n"] = n
    mzn_instance["l"] = l
    mzn_instance["s"] = s
    result = mzn_instance.solve(timeout=t.timedelta(seconds=5), random_seed=42, processes=1, optimisation_level=1, statistics=True)
    # print(result.solution, result.statistics, result.status)
    return np.array(result["bins"])
