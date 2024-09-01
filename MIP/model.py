model = r"""
    reset;

    ## VARIABLES
    param m;
    param n;
    set COURIERS := {1..m}; # couriers with load capacities
    set ITEMS := {1..n}; # items with sizes
    set D_SIZE := {1..n+1};

    param capacity {COURIERS} > 0 integer;
    param size {ITEMS} > 0 integer;
    param D {D_SIZE, D_SIZE} >= 0 integer; # matrix of distances
    param dist_upper_bound := sum {i in D_SIZE} max {j in D_SIZE} D[i,j];
    param obj_lower_bound := max {i in ITEMS} (D[n+1,i]+D[i,n+1]);


    var A {COURIERS, D_SIZE, D_SIZE} binary; # tensor defining the route of each courier
    var T {ITEMS} >= 1, <= n integer; # array that encode the visit sequence
    # var items_per_courier {COURIERS} integer;
    # TODO: if implicit constraint, add lower_bound also on tot_dist
    var tot_dist {COURIERS} >= 0, <= dist_upper_bound integer; # distance traveled by each courier
    var Obj >= obj_lower_bound, <= dist_upper_bound integer;

    ## OBJECTIVE FUNCTION
    minimize Obj_function: Obj;

    ## CONSTRAINTS
    ## constraints on Obj
    s.t. def_Obj {i in COURIERS}:
        Obj >= tot_dist[i];
     
    ## constraints to create A 
    s.t. one_arrival_per_node {k in ITEMS}:
        sum {i in COURIERS, j in D_SIZE} A[i,j,k] = 1; # each A[:,:,k] matrix has exaclty 1 item, just one i courier arrive at k-th point
    s.t. one_departure_per_node {j in ITEMS}:
        sum {i in COURIERS, k in D_SIZE} A[i,j,k] = 1; # each A[:,j,:] matrix has exaclty 1 item, just one i courier depart from j-th point
    s.t. origin_arrival {i in COURIERS}:
        sum {j in ITEMS} A[i,j,n+1] = 1; # each A[i,:,n+1] column has exactly 1 item, the courier i return at the origin
    s.t. origin_departure {i in COURIERS}:
        sum {k in ITEMS} A[i,n+1,k] = 1; # each A[i,n+1,:] row has exactly 1 item, the courier i start from the origin
    s.t. no_self_loop {i in COURIERS, j in ITEMS}:
        A[i,j,j] = 0; # the diagonal of each A[i,:,:] is zero, the i courier must move from a point to another
    s.t. implied_constraint {i in COURIERS}: # TODO
        A[i,n+1,n+1] = 0; # each courier transoprts at least one item
    s.t. balanced_flow {i in COURIERS, j in ITEMS}:
        sum {k in D_SIZE} A[i,k,j] = sum {k in D_SIZE} A[i,j,k]; # for each i courier the sum of each column A[i,:,j] is equal to the sum of each row A[i,j,:]
                                                                 # if the i courier enter arrive at the j-th point it has to depart from it
    s.t. load_capacity {i in COURIERS}:
        sum {j in D_SIZE, k in ITEMS} A[i,j,k]*size[k] <= capacity[i]; # each courier respects its own load capacity 

    ## constraints to create T
    s.t. first_visit {i in COURIERS, k in ITEMS}:
        T[k] <= 1 + 2*n * (1-A[i,n+1,k]); # for every courier the first element delivered, call it k, gets T[k]=1
    s.t. successive_visit_1 {i in COURIERS, j in ITEMS, k in ITEMS}:
        T[j]-T[k] >= 1 - 2*n * (1-A[i,k,j]); # if the A[i,j,k] is 1 (vehicle i leaves node k and enter the node j) then T[j]-T[i]=1, the point j-th is visited exactly after the k-th point
                                             # value of big-M = 2*n
    s.t. successive_visit_2 {i in COURIERS, j in ITEMS, k in ITEMS}:
        T[j]-T[k] <= 1 + 2*n * (1-A[i,k,j]);
    
          
    ## constraint to create tot_dist[i]
    s.t. def_tot_dist {i in COURIERS}:
        sum {j in D_SIZE, k in D_SIZE} A[i,j,k] * D[j,k] = tot_dist[i]; # calculate distance traveled by each courier
          
    ## symmetry breaking with ordered capacity 
    s.t. symmetry_breaking {i in {1..m-1}}:
        sum {j in ITEMS, k in ITEMS} A[i,j,k]*size[k] >= sum {j in ITEMS, k in ITEMS} A[i+1,j,k]*size[k]; # the load of each courier is ordered as the capacity
          
"""