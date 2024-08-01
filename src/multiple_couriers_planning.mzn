include "fzn_nvalue.mzn";
include "bin_packing_capa.mzn";
% Parameters
int: m;  % N. of couriers
int: n;  % N. of items
set of int: COURIERS = 1..m; % Range of couriers from 1 to m
set of int: ITEMS = 1..n+1; % Range of items from 1 to n+1 (origin counted)
array[COURIERS] of int: l;  % Max load size for each courier
array[1..n] of int: s;  % Size of each item (origin not counted)
array[ITEMS, ITEMS] of int: D; % matrix of distances
% Decision variables
array[COURIERS, 1..n+2] of var ITEMS: x;  % Array of item choices, for each courier
array[COURIERS, ITEMS] of var 0..1: y;  % Array of binaries assignment between gotten variables and their couriers
% Output variables
array[COURIERS] of var int: d;  % Distance travelled by each courier
array[COURIERS] of var int: loads;  % Load carried by each courier
array[1..n] of var int: bins;  % Load carried by each courier

%%% Assignment: last y of each courier is 1 -> origin is always present
constraint forall(k in COURIERS) (y[k,n+1] = 1);
%%% Assignment: first and last x are set to origin -> loop guranteed
constraint forall(k in COURIERS) (x[k, 1] = n+1 /\ x[k, n+2] = n+1);
%%% Assignment: populate distances with sum of selected items
%%% Note: this will forse x values to be at least good indexes!! (>=1)
constraint forall(k in COURIERS) (
  d[k] = sum(j in 1..n+1) (D[x[k, j], x[k, j+1]]) % NOTE: loop from 1..n+1 to avoid out-of-array indexing by j+1
);
constraint forall(k in COURIERS) (loads[k] = sum(j in 1..n) (s[j] * y[k, j]));
%%% Objective
var int: max_distance = max(k in COURIERS)(d[k]);

%%% constraint: only one item per y column -> only one courier per item
constraint implied_constraint(forall(j in 1..n) (sum(k in 1..m) (y[k, j]) = 1));
% constraint forall(j in 1..n) (at_most(1, col(y,j), 1));
%%% constraint: max load must be respected for each courier
% constraint forall(k in COURIERS) (loads[k] = sum(j in 1..n) (s[j] * y[k, j]) /\ loads[k] <= l[k]);
constraint bin_packing_capa(l, bins, s) /\ forall(k in COURIERS, j in 1..n) ((y[k,j]=1) <-> (k=bins[j]));
%%% constraint: values in x must be mapped in y as 1s, e.g. x[1,3]=5 <-> y[1,5]=1
constraint forall (k in COURIERS, j in ITEMS) (x[k,j]!=0 <-> y[k,x[k,j]]=1 
     /\ y[k,j]=1 <-> j in row(x,k)
    );
%%% constraint: numb. of distinct values in x must be equal to the number of 1s in y
constraint forall (k in COURIERS) (fzn_nvalue(sum(j in ITEMS)(y[k, j]), row(x,k)));
%%% constraint: only the first item can be repeated
constraint symmetry_breaking_constraint(
let {
  % Alternative Boolean model:
  array[COURIERS] of var ITEMS: reps;
} in
  forall (k in COURIERS) (
%         forall (j in 2..1+((n+1) - sum(j in 1..n+1)(y[k, j]))) (
%           x[k,j] = x[k,2]
%           )
        reps[k] = (n+2) - sum(j in ITEMS)(y[k, j])
         /\ count_eq(row(x,k), x[k,2], reps[k]) 
        /\ forall (j in 2..2+reps[k]-1) (x[k,2] = x[k,j]) % enforce reps on first item
        )
);
%%% constraint: symmetry-breaking for couriers -> couriers with same load may be interchangeable
constraint symmetry_breaking_constraint(
  lex_lesseq(array1d([y[k1,j] | k1,k2 in COURIERS,j in ITEMS where l[k1] = l[k2]]), array1d([y[k2,j] | k1,k2 in COURIERS,j in ITEMS where l[k1] = l[k2]]))
);

%%% search: exploration + exploitation + restart + reuse_past_solution
solve 
      :: seq_search([
              int_search(y, first_fail, indomain_random),
              int_search(x, first_fail, indomain_min), % using relax and restart these two are not always needed
%               int_search(x, first_fail, indomain_median), % only this works with 9th instance
             ])
      :: restart_luby(100)
      :: relax_and_reconstruct(array1d(x),60) % questa riga da uno speedup CLAMOROSO!! (10-100x)
      minimize max_distance;
%     satisfy;

output [
   "max_distance: " ++ show(max_distance) ++ "\n" ++
   "distances: " ++ show(d) ++ "\n" ++
   "x: " ++ show2d(x) ++ "\n" ++
   "y: " ++ show2d(y) ++ "\n" ++
   "bins: " ++ show(bins) ++ "\n" ++
   "loads: " ++ show(loads) ++ "\n" ++
   "(allowed loads): " ++ show(l) ++ "\n"
];

