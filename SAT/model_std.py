import time
import copy

from z3 import *

from utils import *
from encoding import *
import numpy as np


PREP_MZN_FILE = "CP/bin_pack.mzn"

def std_model(m, n, l, s, D, symmetry_breaking=True, search='Binary', display_solution=False, timeout_duration=300, UB=True, partitionUB=False):
   
    start_time = time.time()
    ## VARIABLES

    # a for assignments
    a = [[Bool(f"a_{i}_{j}") for j in range(n)] for i in range(m)]
    # a_ij = 1 indicates that courier i delivers object j

    # r for routes
    r = [[[Bool(f"r_{i}_{j}_{k}") for k in range(n+1)] for j in range(n+1)] for i in range(m)]

    # t for times
    t = [[Bool(f"deliver_{j}_as_{k}-th") for k in range(n)] for j in range(n)]

    courier_loads = [[Bool(f"cl_{i}_{k}") for k in range(num_bits(sum(s)))] for i in range(m)]
    # courier_loads_i = binary representation of actual load carried by each courier

    solver = Solver()


    if symmetry_breaking:
        # sort the list of loads, keeping the permutation used for later
        L = [(l[i], i) for i in range(m)]
        L.sort(reverse=True)
        l, permutation = zip(*L)
        l = list(l)
        permutation = list(permutation)

        ## Symmetry breaking constraint -> after having sorted l above, impose the actually couriers_loads to be sorted decreasingly as well
        solver.add(sort_decreasing(courier_loads))
        # Break symmetry within same load amounts, i.e.:
        # if two couriers carry the same load amount, impose a lexicografic ordering on the respective rows of a,
        # i.e. the first courier will be the one assigned to the route containing the item with higher index j
        for i in range(m - 1):
            solver.add(
                Implies(equal(courier_loads[i], courier_loads[i + 1]),
                        leq(a[i], a[i + 1])))

    ## TODO SB
    
    # Conversions into binary of the loads and sizes
    s_bin = [int_to_bin(s_j, num_bits(s_j)) for s_j in s]
    l_bin = [int_to_bin(l_i, num_bits(l_i)) for l_i in l]


    # each obj one courier
    for j in range(n):
        solver.add(exactly_one_seq([a[i][j] for i in range(m)], f"assignment_{j}"))


    # maximum load
    for i in range(m):
        solver.add(conditional_sum_K_bin(a[i], s_bin, courier_loads[i], f"compute_courier_load_{i}"))
        solver.add(leq(courier_loads[i], l_bin[i]))

    #cevery object is delivered and only once, in the time table
    for i in range(n):
        solver.add(exactly_one_seq(t[i], f"time_of_{i}"))

    # for every courier (3^ dimension of the r matrix). Constraint on r
    for i in range(m):
        # diagonal is full of zeros (imp from j to j)
        solver.add(And([Not(r[i][j][j]) for j in range(n)]))

        # row j has a 1 iff courier i delivers object j
        # obj
        for j in range(n):# If a_ij --> exactly_one(r_ij)
            solver.add(Implies(a[i][j], exactly_one_seq(r[i][j], f"courier_{i}_leaves_{j}"))) 
            solver.add(Implies(Not(a[i][j]), all_false(r[i][j])))   # else all_false(r_ij)
        solver.add(exactly_one_seq(r[i][n], f"courier_{i}_leaves_origin"))    # exactly_one in origin point row === courier i leaves from origin

        # column j has a 1 iff courier i delivers object j
        # obj
        for k in range(n):# If a_ij --> exactly_one(r_i,:,k)
            solver.add(Implies(a[i][k], exactly_one_seq([r[i][j][k] for j in range(n+1)], f"courier_{i}_reaches_{k}")))  
            solver.add(Implies(Not(a[i][k]), all_false([r[i][j][k] for j in range(n+1)])))   # else all_false(r_i,:,k)
        solver.add(exactly_one_seq([r[i][j][n] for j in range(n+1)], f"courier_{i}_returns_to_origin"))         # exactly_one in origin point column === courier i returns to origin

        # TODO
        # use ordering between t_j and t_k in every edge travelled
        # in order to avoid loops not containing the origin
        for j in range(n):
            for k in range(n):
                solver.add(Implies(r[i][j][k], successive(t[j], t[k])))
            solver.add(Implies(r[i][n][j], t[j][0]))



    ## OPTIMIZATION SEARCH

    # flatten r and D
    flat_r = [flatten(r[i]) for i in range(m)]
    flat_D = flatten(D)
    # convert flat_D to binary
    flat_D_bin = [int_to_bin(e, num_bits(e) if e > 0 else 1) for e in flat_D]


    # Constraint 6: distances travelled by each courier
    # distances[i] := binary representation of the distance travelled by courier i
    # Take as upper bound the greater n-(m-1) maximum distances, since that's the maximum items a single courier can be assigned to
    max_distances = [max(D[i]) for i in range(n+1)] # TODO
    max_distances.sort()

    # Decision of upper bound
    if UB:
        if partitionUB:
            mznInstance = getMznInstance(PREP_MZN_FILE)
            partition = preprocess(mznInstance,m,n,l,s)
            upper_bound = upperBound3(D,m,partition)
        else:
            upper_bound = int(upperBound(np.array(D),l,s))
    else:
        upper_bound = sum(max_distances[m-1:])
    lower_bound = max([D[n][j] + D[j][n] for j in range(n)])

    distances = [[Bool(f"dist_bin_{i}_{k}") for k in range(num_bits(upper_bound))] for i in range(m)]

    # definition of distances using constraints TODO
    for i in range(m):
        solver.add(conditional_sum_K_bin(flat_r[i], flat_D_bin, distances[i], f"distances_def_{i}"))

    model = None
    obj_value = None
    encoding_time = time.time()

    timeout = encoding_time + timeout_duration

    solver.push()

    if search == 'Linear':

        solver.set('timeout', millisecs_left(time.time(), timeout))
        while solver.check() == z3.sat:

            model = solver.model()
            obj_value = obj_function(model, distances)

            if obj_value <= lower_bound:
                break

            upper_bound = obj_value - 1
            upper_bound_bin = int_to_bin(upper_bound, num_bits(upper_bound))


            solver.pop()
            solver.push()

            solver.add(AllLessEq_bin(distances, upper_bound_bin))
            now = time.time()
            if now >= timeout:
                break
            solver.set('timeout', millisecs_left(now, timeout))


    elif search == 'Binary':

        upper_bound_bin = int_to_bin(upper_bound, num_bits(upper_bound))
        solver.add(AllLessEq_bin(distances, upper_bound_bin))

        lower_bound_bin = int_to_bin(lower_bound, num_bits(lower_bound))
        solver.add(AtLeastOneGreaterEq_bin(distances, lower_bound_bin))

        while lower_bound <= upper_bound:
            #print(f"upper {upper_bound}, dis {distances}, lower{lower_bound}")

            mid = int((lower_bound + upper_bound)/2)
            mid_bin = int_to_bin(mid, num_bits(mid))
            solver.add(AllLessEq_bin(distances, mid_bin))

            now = time.time()
            if now >= timeout:
                break
            solver.set('timeout', millisecs_left(now, timeout))

            if solver.check() == z3.sat:
                model = solver.model()
                obj_value = obj_function(model, distances)

                if obj_value <= 1:
                    break

                upper_bound = obj_value - 1
                upper_bound_bin = int_to_bin(upper_bound, num_bits(upper_bound))


            else:

                lower_bound = mid + 1
                lower_bound_bin = int_to_bin(lower_bound, num_bits(lower_bound))

            solver.pop()
            solver.push()
            solver.add(AllLessEq_bin(distances, upper_bound_bin))
            solver.add(AtLeastOneGreaterEq_bin(distances, lower_bound_bin))

    else:
        raise ValueError(f"Input parameter [search] mush be either 'Linear' or 'Binary', was given '{search}'")


    # compute time taken
    end_time = time.time()

    print(f"end after {end_time}, res {obj_value}")

    if end_time >= timeout:
        solving_time = timeout_duration    # solving_time has upper bound of timeout_duration if it timeouts
    else:
        solving_time = math.floor(end_time - encoding_time)

    # if no model is found -> UNSAT if solved to optimality else UNKKNOWN
    if model is None:
        print("__________ si unsat")
        ans = "N/A" if solving_time == timeout_duration else "UNSAT"
        return (ans, solving_time, None)
    
    # reorder all variables w.r.t. the original permutation of load capacities, i.e. of couriers
    if symmetry_breaking:
        a_copy = copy.deepcopy(a)
        r_copy = copy.deepcopy(r)
        for i in range(m):
            a[permutation[i]] = a_copy[i]
            r[permutation[i]] = r_copy[i]

    # check that all couriers travel hamiltonian cycles
    R = evaluate(model, r)
    assert(check_all_hamiltonian(R))

    T = evaluate(model, t)
    A = evaluate(model, a)
    
    if display_solution:
        Dists = evaluate(model, distances)
        displayMCP(T, Dists, obj_value, A)

    route = retrieve_routes(T, A)

    #dic = {""}

    return (obj_value, solving_time, route)



    