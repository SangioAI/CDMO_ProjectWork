include "globals.mzn";

% =================================================%
% Parameters
% =================================================%
int: m;                                    % N. of couriers
int: n;                                    % N. of items
set of int: ITEMS = 1..n;                  % Range of items from 1 to n+1 (origin not counted)
set of int: COURIERS = 1..m;               % Range of couriers from 1 to m
array[COURIERS] of int: l;                 % Max load size for each courier
array[ITEMS] of int: s;                    % Size of each item (origin not counted)

% =================================================%
% Decision Variables
% =================================================%
array[ITEMS] of var COURIERS: bins;         % item assigment to each courier


% ----- load constraints ------------ %
constraint  bin_packing_capa(l, bins, s); % pack items in bins that don't exceed capacities of couriers


% =================================================%
% Objective
% =================================================%
% search: exploration + exploitation
solve 
      :: seq_search([
              int_search(bins, first_fail, indomain_split, complete),
             ])
      satisfy;

%--------------- OUTPUT -----------------------
% output [
%    "max_distance: " ++ show(max_distance) ++ "\n" ++
%    "successor: " ++ show(successor) ++ "\n" ++
%    "START_NODES: " ++ show(START_NODES) ++ "\n" ++
%    "END_NODES: " ++ show(END_NODES) ++ "\n" ++
%    "bins: " ++ show(bins) ++ "\n" ++
%    "loads: " ++ show(loads) ++ "\n" ++
%    "loads[END_NODES]: " ++ show([loads[i] | i in END_NODES]) ++ "\n" ++
%    "(allowed loads): " ++ show(l) ++ "\n" ++
%    "distances: " ++ show(d_cumulative) ++ "\n"
% ];

