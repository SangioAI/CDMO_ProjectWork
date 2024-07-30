include "globals.mzn";
% Parameters
int: m;  % N. of couriers
int: n;  % N. of items
set of int: COURIERS = 1..m; % Range of couriers from 1 to m
set of int: ITEMS = 1..n+1; % Range of items from 1 to n+1 (origin counted)
array[COURIERS] of int: l;  % Max load size for each courier
array[1..n] of int: s;  % Size of each item (origin not counted)
array[ITEMS, ITEMS] of int: D; % matrix of distances
% Decision variables
array[COURIERS, ITEMS] of var 0..n+1: y;  % Array of binaries assignment between gotten variables and their
array[COURIERS, ITEMS] of var 0..n+1: y_sup;  % Array of binaries assignment between gotten variables and their couriers
% Output variables
array[COURIERS] of var int: d;  % Distance travelled by each courier
array[COURIERS] of var int: loads;  % Load carried by each courier
array[COURIERS, ITEMS] of var ITEMS: x;  % Array of binaries assignment between gotten variables


% Assignment: last y of each courier is 1 -> origin is always present
constraint forall(k in 1..m) (y[k,n+1] = 1);
% constraint forall (k in 1..m, j in ITEMS) (x[k,y[k,j]]!=j <- y[k,j]!=0);

% constraint: only one item per y column, all other to 0s -> only one courier per item
constraint forall(j in 1..n) (
  count_eq(col(y,j), 0, m-1)
 );
% constraint: max load must be respected for each courier
constraint forall(k in 1..m) (loads[k] = sum(j in 1..n) (s[j] * min(1, y[k, j])) /\ loads[k] <= l[k]);
% constraint: numb. of distinct values in x must be equal to the number of 1s in y
constraint forall (k in 1..m) (alldifferent_except_0(row(y,k)));
% constraint: populate distances with sum of selected items

% constraint forall(k in 1..m) (
% % y: 3, 4, 0, 2, 0, 0, 0, 0, 0, 1
%    d[k] = sum(i in ITEMS) (
%    forall(j in ITEMS) (
%       y[k,j]
%    )
% %  d[k] = sum(j in 2..(arg_min(row(x,k))-2)) (D[y[k, j], j]) + D[x[k, arg_min(row(x,k))-1],n+1]% NOTE: loop from 1..n to avoid out-of-array indexing by j+1
% );


% Objective
var int: max_distance = max(k in 1..m)(d[k]);
% search: exploration + exploitation
% Exploration is needed(int_search): without it the model get initial wrong guesses
% Exploitation is needed(int_search): without it the model get initial wrong guesses
solve satisfy;
%       :: seq_search([
%              int_search(y, first_fail, indomain_random),
%              int_search(x, first_fail, indomain_min),
% %              int_search(loads, smallest, indomain_min),
% %              restart_luby(10)% 
%              ])
%       :: int_search(x, first_fail, indomain_min)
%       :: restart_luby(10)% 
%       minimize max_distance;

output [
   "max_distance: " ++ show(max_distance) ++ "\n" ++
   "distances: " ++ show(d) ++ "\n" ++
%    "x: " ++ show2d(x) ++ "\n" ++
   "y: " ++ show2d(y) ++ "\n" ++
   "loads: " ++ show(loads) ++ "\n" ++
   "(allowed loads): " ++ show(l) ++ "\n"
];

