include "globals.mzn";
% Parameters
int: m;  % N. of couriers
int: n;  % N. of items
set of int: COURIERS = 1..m; % Range of couriers from 1 to m
int: ORIGIN = n+1;
set of int: ITEMS = 1..n+1; % Range of items from 1 to n+1 (origin counted)
array[COURIERS] of int: l;  % Max load size for each courier
array[1..n] of int: s;  % Size of each item (origin not counted)
array[ITEMS, ITEMS] of 0..300: D; % matrix of distances
% array[1..(n+1)*(n+1)] of int: D; % matrix of distances
% Decision variables
array[COURIERS, 1..n+2] of var 0..n+1: x;  % Array of item choices, for each courier
array[COURIERS, ITEMS] of var 0..1: y;  % Array of binaries assignment between gotten variables and their couriers
array[1..n] of var int: bins;  % Load carried by each courier
array[COURIERS] of var int: loads;  % Load carried by each courier
% array[COURIERS] of var int: K;  % weight of each path of each courier == distances of each courier
% array[COURIERS, 1..(n+1)*(n+1)] of var bool: e_path;
% Output variables
array[COURIERS] of var int: d;  % Distance travelled by each courier
var int: max_distance;


%--------------- ASSIGNEMENTS -----------------------
% Assignment: last y of each courier is 1 -> origin is always present
constraint 
  forall(k in COURIERS) (y[k, ORIGIN] = 1);
% Assignment: first and last x are set to origin -> loop guranteed
constraint 
  forall(k in COURIERS) (x[k, 1] = ORIGIN /\ x[k, n+2] = ORIGIN);
% Assignment: populate distances with sum of selected items
loads = [sum(j in 1..n) (s[j] * y[k, j]) | k in COURIERS];

%--------------- SCORE FUNCTION -----------------------
constraint 
  forall(k in COURIERS) (
    d[k] = sum(j in 1..n+1) (D[x[k, j], x[k, j+1]]) % NOTE: loop from 1..n+1 to avoid out-of-array indexing by j+1
); % distances of each courier
constraint 
  max_distance = max(k in COURIERS)(d[k]); % max distance to minimize


%--------------- CONSTRAINTS -----------------------
constraint 
  bin_packing_capa(l, bins, s); % pack items in bins that don't exceed capacities of couriers


%--------------- CHANNELING CONSTRAINTS -----------------------
constraint 
  forall(k in COURIERS, j in 1..n) (
   (y[k,j]=1) <-> (k=bins[j])
); % link y values to bins values

constraint 
  int_set_channel(bins, [{x[k,j] | j in 2..(count(bins,k)+1)} | k in COURIERS])::domain; % link x values and k values

%--------------- REDUNDANT CONSTRAINTS -----------------------
constraint 
  implied_constraint(forall (k in COURIERS) (fzn_nvalue(sum(j in ITEMS)(y[k, j]), row(x,k))))::domain; % number of different values in x must be uqual to number of 1s in y --> implied by channeling constraints

constraint 
  implied_constraint(forall(j in 1..n) (sum(k in 1..m) (y[k, j]) = 1)); % sum of values in y columns must be 1, except for last containing all 1s (origin is always visited) --> implied by channaling constraints



%--------------- SYMMETRY-BREAKING CONSTRAINTS -----------------------
constraint 
  symmetry_breaking_constraint(
    forall(k1,k2 in COURIERS where l[k1] == l[k2] /\ k1 < k2) (
      lex_lesseq(array1d([y[k1,j] | j in ITEMS ]), array1d([y[k2,j] | j in ITEMS]))
    )
); % symmetry-breaking for couriers: couriers with same load can be interchangeable 

% constraint 
%   symmetry_breaking_constraint(
%     forall(k1,k2 in COURIERS where k1 < k2 /\ loads[k1] <= l[k2] /\ loads[k2] <= l[k1]) (
%       lex_lesseq([y[k1,j] | j in ITEMS], [y[k2,j] | j in ITEMS])
%     )
% ); % symmetry-breaking for bin packing solution: pairs of couriers with enough capacity for each others' loads are interchangable -> this impose increasing ordering constraint w.r.t. courier indexing

constraint 
  let {
    % Alternative Boolean model:
    array[COURIERS] of var ITEMS: reps;
    int: rep_idx = n+2;
  } in
    forall (k in COURIERS) (
          reps[k] = (n+2) - sum(j in ITEMS)(y[k, j]) % calculate number of repetition
           /\ count_eq(row(x,k), x[k,rep_idx], reps[k]+1) % last item visited must be repeted reps[k] times
          /\ forall (j in rep_idx-reps[k]+1..rep_idx) (x[k,rep_idx] = x[k,j]) % force repetition on last part of x[k,:]
); % constraint: only the last item can be repeated


% include "path.mzn";

% predicate my_bounded_dpath(int: N, int: E, array[int] of int: from, array[int] of int: to, array[int] of int: w,
%                     var int: s, var int: t, array[int] of var bool: ns, array[int] of var bool: es, var int: K) =
%    dpath(N,E,from,to,s,t,ns,es) /\
%    K = sum(e in 1..E)(es[e]*w[e]) + w[(n+1-1)*(n+1)+(s)];

% constraint
%   let {
%     array[COURIERS] of var int: first;
%   } in
%   forall (k in COURIERS) (
%     first[k] = arg_max(row(y,k)) 
%     /\
%     my_bounded_dpath(n+1,
%                   (n+1)*(n+1),
%                   [i | i,j in ITEMS],
%                   [j | i,j in ITEMS],
%                   array1d(D),
%                   first[k],
%                   ORIGIN,
%                   [y[k,j] == 1 | j in ITEMS],
%                   row(e_path,k),
%                   d[k])
% %       bounded_dpath([i | i,j in ITEMS],
% %                   [j | i,j in ITEMS],
% %                   array1d([1 | i,j in ITEMS]),
% %                   1,
% %                   n+1,
% %                   [true | j in ITEMS],
% %                   e_path,
% %                   K[k])              
% );

%--------------- OBJECTIVE -----------------------
% search: exploration + exploitation
solve 
      :: seq_search([
%               int_search(x, first_fail, indomain_min), % not so useful
              int_search(bins, first_fail, indomain_random), % bins-search to explore the space randomly..in conjuction with y-search is very effective for some instances
              int_search(y, first_fail, indomain_random), % y-search to be done after bins..works better
             ])
      :: restart_luby(200) % use Luby restart to avoid getting stuck in deep part of search trees
%       :: relax_and_reconstruct(array1d(x),60) % Nearest Neighborhood search using 60% of x solution...may not be the best solution since x has mostly repeated items
      minimize max_distance;
%     satisfy;
    

%--------------- OUTPUT -----------------------
output [
   "max_distance: " ++ show(max_distance) ++ "\n" ++
   "distances: " ++ show(d) ++ "\n" ++
%    "K: " ++ show(K) ++ "\n" ++
%    "e_path: " ++ show([i | k in COURIERS, i,j in ITEMS where e_path[k,(i-1)*(n+1)+j]]) ++ "\n" ++
   "x: " ++ show2d(x) ++ "\n" ++
   "y: " ++ show2d(y) ++ "\n" ++
   "bins: " ++ show(bins) ++ "\n" ++
   "loads: " ++ show(loads) ++ "\n" ++
   "(allowed loads): " ++ show(l) ++ "\n"
];

