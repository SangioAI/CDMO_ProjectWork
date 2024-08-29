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
int: UB;                           % Upper Bound
int: LB;                               % Lower Bound
array[1..n+1, 1..n+1] of int: D;  % distances
int: HEU;

% =================================================%
% Sets
% =================================================%
int: DEPOT = n+1;
set of int: NODES = 1..(n + 2*m); 
set of int: DEPOT_NODES = (n + 1)..(n + 2*m);
set of int: START_NODES = (n + 1)..(n + m);
set of int: END_NODES = (n + m + 1)..(n + 2*m);
set of int: END_NODES_NOLAST = (n + m + 1)..(n + 2*m)-1;

% =================================================%
% Decision Variables
% =================================================%
array[NODES] of var NODES: successor;       % Successor list for each node visited
array[NODES] of var NODES: predecessor;       % Predecessor list for each node visited - redundant but adding it we have speedup
array[NODES] of var COURIERS: bins;         % item assigment to each courier
array[NODES] of var 0..max(l): loads;          % Load carried by each courier - redundant
% Output variables
array[NODES] of var 0..UB: d_cumulative;           % Distance travelled by each courier
var LB..UB: max_distance;


% =================================================%
% Constraints
% =================================================%

%--------------- ASSIGNEMENTS CONSTRAINTS -----------------------

constraint forall(i in END_NODES_NOLAST) (successor[i] = i-m+1); % each END_NODES must preceed next courier START_NODES
constraint successor[END_NODES[m]] == START_NODES[1];            % last END_NODES must preceed first START_NODES
   
%--------------- MAIN CONSTRAINTS -----------------------

% ----- circuit constraints ------------ %
constraint circuit(successor)::domain;
constraint redundant_constraint(circuit(predecessor))::domain;
constraint redundant_constraint(forall(i in NODES) ((successor[predecessor[i]] == i) /\ (predecessor[successor[i]] == i)));

% ----- channaling constraints ------------ %
constraint redundant_constraint(forall(i in START_NODES)  (bins[i] = i - n));          % retrieve couriers number from START_NODES
constraint forall(i in END_NODES)  (bins[i] = i - n - m);          % retrieve couriers number from END_NODES
constraint forall(i in ITEMS)      (bins[successor[i]] = bins[i]); % recursive assignement of couriers to nodes
constraint redundant_constraint(forall(i in ITEMS)      (bins[predecessor[i]] = bins[i])); % recursive assignement of couriers to nodes

% ----- load constraints ------------ %
constraint  bin_packing_capa(l, [bins[i] | i in  ITEMS], s); % pack items in bins that don't exceed capacities of couriers

constraint 
  symmetry_breaking_constraint(
    forall(k1,k2 in COURIERS where ((l[k1]>=HEU /\ l[k2]>=HEU) \/ (l[k1] == l[k2])) /\ k1 < k2) (
      lex_lesseq(array1d([bool2int(bins[j]==k1) | j in NODES ]), array1d([bool2int(bins[j]==k2) | j in NODES ]))
    )
);

% ----- distance constraints ------------ %
constraint forall(i in START_NODES)(d_cumulative[i] = 0); % set distance in START_NODES to 0
constraint forall(i in START_NODES)(d_cumulative[successor[i]] = D[DEPOT, successor[i]]);   % set distance from START_NODES as distance from DEPOT 
constraint forall(i in ITEMS)(d_cumulative[successor[i]] = d_cumulative[i] + D[i, min([successor[i], DEPOT])]); % set distance from ITEMS as cumulative distance chosen paying attention to not exceed real nodes number (DEPOT is max)

% vehicle load when starting at the depot
constraint forall(i in START_NODES) (loads[i] = 0 /\ 0 = loads[successor[i]]); % set loads of START_NODES to 0
constraint forall(i in ITEMS) (loads[i] + s[i] = loads[successor[i]]); % set loads of ITEMS[i] to loads[successor[i]]-s[i]


%--------------- SCORE FUNCTION -----------------------
constraint max_distance = max(d_cumulative);

% =================================================%
% Objective
% =================================================%
% search: exploration + exploitation
solve 
      :: seq_search([
              int_search(successor, first_fail, indomain_split, complete),
              int_search(predecessor, first_fail, indomain_split, complete),
              int_search(bins, first_fail, indomain_split, complete),
%               int_search([d_cumulative[j] | j in NODES],first_fail, indomain_min, complete),
             ])
      minimize max_distance::domain;

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

