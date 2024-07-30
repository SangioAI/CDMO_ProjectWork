include "fzn_nvalue.mzn";
% Parameters
int: m;  % N. of couriers
int: n;  % N. of items
set of int: COURIERS = 1..m; % Range of couriers from 1 to m
set of int: ITEMS = 1..n+1; % Range of items from 1 to n+1 (origin counted)
array[COURIERS] of int: l;  % Max load size for each courier
array[1..n] of int: s;  % Size of each item (origin not counted)
array[ITEMS, ITEMS] of int: D; % matrix of distances
% Decision variables
array[COURIERS, ITEMS] of var 0..n+1: x;  % Array of item choices, for each courier
array[COURIERS, ITEMS] of var 0..1: y;  % Array of binaries assignment between gotten variables and their couriers
% Output variables
array[COURIERS] of var int: d;  % Distance travelled by each courier
array[COURIERS] of var int: loads;  % Load carried by each courier

% Assignment: last y of each courier is 1 -> origin is always present
constraint forall(k in 1..m) (y[k,n+1] = 1);
% Assignment: first and last x are set to origin -> loop guranteed
constraint forall(k in 1..m) (x[k, 1] = n+1 /\ x[k, n+1] = n+1);
% Assignment: populate distances with sum of selected items
constraint forall(k in 1..m) (
  d[k] = sum(j in 1..n) (D[x[k, j], x[k, j+1]]) % NOTE: loop from 1..n to avoid out-of-array indexing by j+1
);

% constraint: only one item per y column -> only one courier per item
constraint forall(j in 1..n) (sum(k in 1..m) (y[k, j]) = 1);
% constraint: max load must be respected for each courier
constraint forall(k in 1..m) (loads[k] = sum(j in 1..n) (s[j] * y[k, j]) /\ loads[k] <= l[k]);
% constraint: values in x must be mapped in y as 1s, e.g. x[1,3]=5 <-> y[1,5]=1
constraint forall (k in 1..m, j in ITEMS) (x[k,j]!=0 <-> y[k,x[k,j]]=1);
% constraint: numb. of distinct values in x must be equal to the number of 1s in y
constraint forall (k in 1..m) (fzn_nvalue(sum(j in 1..n+1)(y[k, j]), row(x,k)));

% Objective
var int: max_distance = max(k in 1..m)(d[k]);
% search: exploration + exploitation
% Exploration is needed(int_search): without it the model get initial wrong guesses
% Exploitation is needed(int_search): without it the model get initial wrong guesses
solve 
      :: seq_search([
             int_search(y, first_fail, indomain_random),
             int_search(x, first_fail, indomain_min),
%              int_search(loads, smallest, indomain_min),
%              restart_luby(10)% 
             ])
%       :: int_search(x, first_fail, indomain_min)
      :: restart_luby(10)% 
      minimize max_distance;

output [
   "max_distance: " ++ show(max_distance) ++ "\n" ++
   "distances: " ++ show(d) ++ "\n" ++
   "x: " ++ show2d(x) ++ "\n" ++
   "y: " ++ show2d(y) ++ "\n" ++
   "loads: " ++ show(loads) ++ "\n" ++
   "(allowed loads): " ++ show(l) ++ "\n"
];

