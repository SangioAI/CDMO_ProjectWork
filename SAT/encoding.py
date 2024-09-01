# 
# LOGICS
#
from z3 import *
from utils import *
from utils import bin_to_int


def at_least_one(bool_vars):
    """Z3 encoding of "At least one" over bool_vars

    Args:
        bool_vars (list[Bool]): the list of input Z3 Bool objects 

    Returns:
        Z3-expression: the Z3 encoding of "At least one" over bool_vars
    """
    return Or(bool_vars)

def at_most_one_seq(x, name):
    """Z3 encoding of "At most one" using sequential encoding

    Args:
        x (list[Bool]): the list of input Z3 Bool objects
        name (str): the name to append to the auxiliary variables

    Returns:
        Z3-expression: the Z3 encoding of "At most one" over x
    """
    n = len(x)
    if n == 1:
        return True
    s = [Bool(f"s_{i}_{name}") for i in range(n-1)]     # s[i] modeled as: s[i] is true iff the sum up to index i is 1

    clauses = []
    clauses.append(Or(Not(x[0]), s[0]))                 # x[0] -> s[0]
    for i in range(1, n-1):
        clauses.append(Or(Not(x[i]), s[i]))             # these two clauses model (x[i] v s[i-1]) -> s[i]
        clauses.append(Or(Not(s[i-1]), s[i]))
        clauses.append(Or(Not(s[i-1]), Not(x[i])))      # this one models s[i-1] -> not x[i]
    clauses.append(Or(Not(s[-1]), Not(x[-1])))          # s[n-2] -> not x[n-1]
    return And(clauses)

def exactly_one_seq(bool_vars, name):
    """Z3 encoding of "Exactly one" using sequential encoding

    Args:
        bool_vars (list[Bool]): the list of input Z3 Bool objects
        name (str): the name to append to the auxiliary variables

    Returns:
        Z3-expression: the Z3 encoding of "Exactly one" over x
    """
    return And(at_least_one(bool_vars), at_most_one_seq(bool_vars, name))

def equal(v, u):
    """Z3 encoding of "Equal" position-wise

    Args:
        v (list[Bool]): the first term
        u (list[Bool]): the second term

    Returns:
        Z3-expression: the Z3 encoding of "Equal"
    """
    assert(len(v) == len(u))
    return And([v[k]==u[k] for k in range(len(v))])

def all_false(v):
    """Z3 encoding of "All false"

    Args:
        v (list[Bool]): the input list of Bools

    Returns:
        Z3-expression: the Z3 encoding of "All false"
    """
    return And([Not(v[k]) for k in range(len(v))])

## Orderings encoding

def successive(v, u):
    """Encoding of the fact that the ONLY True value present in v is followed 
    by the ONLY True value present in u, in its successive position
    e.g. v = 00010000
         u = 00001000

    Args:
        v (list[Bool]): input list of Z3 Bool variables, already constrained to have exactly one True value
        u (list[Bool]): input list of Z3 Bool variables, already constrained to have exactly one True value

    Returns:
        Z3-Expression: Encoding of the "successive" constraint described
    """
    n = len(v)
    clauses = []

    clauses.append(Not(u[0]))
    for i in range(n-1):
        clauses.append(v[i] == u[i+1])
    clauses.append(Not(v[n-1]))

    return And(clauses)

#
# NUMBERS
#

def leq_same_digits(v, u, digits):
    """Encoding of v <= u, implementation with digits fixed and equal between v and u

    Args:
        v (list[Bool]): binary representation of v with Z3 Bool variables
        u (list[Bool]): binary representation of u with Z3 Bool variables
        digits (int): number of digits of v and u

    Returns:
        (Z3-expression): encoding of v <= u in binary considering their {digits} most significant bits
    """
    assert(len(v) == len(u) and len(u) == digits)
    if digits == 1:
        return Or(v[0]==u[0], And(Not(v[0]), u[0]))
    else:
        return Or(And(Not(v[0]), u[0]),
                  And(v[0]==u[0], leq_same_digits(v[1:], u[1:], digits-1)))


def leq(v, u):
    """Encoding of v <= u, implementation with different digits btw v and u

    Args:
        v (list[Bool]): binary representation of v with Z3 Bool variables
        u (list[Bool]): binary representation of u with Z3 Bool variables

    Returns:
        (Z3-expression): encoding of v <= u in binary
    """
    digits_v = len(v)
    digits_u = len(u)

    if digits_v == digits_u:
        return leq_same_digits(v, u, digits_v)
    elif digits_v < digits_u:
        delta_digits = digits_u - digits_v
        return Or(Or(u[:delta_digits]),
                  leq_same_digits(v, u[delta_digits:], digits_v))
    else:
        delta_digits = digits_v - digits_u
        return And(all_false(v[:delta_digits]), leq_same_digits(v[delta_digits:], u, digits_u))


def sum_bin_same_digits(a_bin, b_bin, d_bin, digits, name):
    """Encodes into a SAT formula the binary sum {a_bin + b_bin = d_bin}, each number having {digits} num of bits

    Args:
        a_bin (list[Bool]): binary representation of a with Z3 Bool variables
        b_bin (list[Bool]): binary representation of b with Z3 Bool variables
        d_bin (list[Bool]): binary representation of d with Z3 Bool variables
        digits (int): number of bits of each number
        name (str): string to identify carry boolean variables

    Returns:
        formula (Z3-expression): formula representing SAT encoding of binary sum
        c[0] (Bool): last carry of binary encoding
    """
    # c_k represents carry at bit position k
    c = [Bool(f"c_{k}_{name}") for k in range(digits + 1)]
    c[-1] = False

    clauses = []
    for k in range(digits - 1, -1, -1):
        clauses.append((a_bin[k] == b_bin[k]) == (c[k + 1] == d_bin[k]))
        clauses.append(c[k] == Or(And(
            a_bin[k], b_bin[k]), And(a_bin[k], c[k +
                                                 1]), And(b_bin[k], c[k + 1])))

    formula = And(clauses)
    return (formula, c[0])


def sum_bin(a_bin, b_bin, d_bin, name):
    """Encodes into a SAT formula the binary sum {a_bin + b_bin = d_bin}, with digits(a_bin) <= digits(b_bin) == digits(d_bin)

    Args:
        a_bin (list[Bool]): binary representation of a with Z3 Bool variables
        b_bin (list[Bool]): binary representation of b with Z3 Bool variables
        d_bin (list[Bool]): binary representation of d with Z3 Bool variables
        name (str): string to identify carry boolean variables

    Returns:
        (Z3-expression): formula representing SAT encoding of binary sum
    """
    digits_a = len(a_bin)
    digits_b = len(b_bin)
    digits_d = len(d_bin)
    assert (digits_a <= digits_b and digits_b == digits_d)

    delta_digits = digits_b - digits_a

    if delta_digits == 0:
        formula, last_carry = sum_bin_same_digits(a_bin, b_bin, d_bin,
                                                  digits_a, name)
        return And(formula, Not(last_carry))  # imposing no overflow

    sub_sum_formula, last_carry = sum_bin_same_digits(a_bin,
                                                      b_bin[delta_digits:],
                                                      d_bin[delta_digits:],
                                                      digits_a, name)
    c = [Bool(f"c_propagated_{k}_{name}")
         for k in range(delta_digits)] + [last_carry]
    c[0] = False  # imposing no further overflow

    clauses = []
    for k in range(delta_digits - 1, -1, -1):
        clauses.append(d_bin[k] == Xor(b_bin[k], c[k + 1]))
        clauses.append(c[k] == And(b_bin[k], c[k + 1]))

    return And(And(clauses), sub_sum_formula)


def conditional_sum_K_bin(x, alpha, delta, name):
    """Encodes into a SAT formula the constraint {delta = sum_over_j(alpha[j] | x[j] == True)}

    Args:
        x (list[Bool]): list of Z3 Variables, i.e. x_j tells wether or not to add alpha_j to the sum
        alpha (list[list[bool]]): list of known coefficients, each one represented as list[bool] i.e. binary number, whose subset will be summed in the constraint
        delta (list[Bool]): list of Z3 Variables, which will be constrained to represent the sum
        name (string): to uniquely identify the created variables
    Returns:
        formula (Z3-expression): And of clauses representing SAT encoding of Linear Integer constraint

    """
    n = len(x)
    digits = len(delta)

    # matrix containing temporary results of sum_bin
    d = [[Bool(f"d_{j}_{k}_{name}") for k in range(digits)]
         for j in range(n - 1)]  # j = 1..n-1 because last row will be delta
    d.append(delta)

    clauses = []

    # row 0
    diff_digits = digits - len(alpha[0])
    assert (diff_digits >= 0)
    clauses.append(
        And(
            Implies(
                x[0],
                And(all_false(d[0][:diff_digits]),
                    equal(d[0][diff_digits:], alpha[0]))
            ),  # If x[0] == 1 then d_0 == alpha_0 (with eventual padding of zeros)
            Implies(Not(x[0]),
                    all_false(d[0]))))  # elif x[0] == 0 then d_0 == [0..0]

    # row j>1
    for j in range(1, n):
        clauses.append(
            And(
                Implies(x[j],
                        sum_bin(alpha[j], d[j - 1], d[j], f"{name}_{j-1}_{j}")
                        ),  # If c_j == 1 then d_j == d_j-1 + alpha_j
                Implies(Not(x[j]), equal(d[j], d[j - 1]))))

    return And(clauses)

## Symmetry breaking constraint

def sort_decreasing(matrix):
    """Encoding of the constraint that the binary numbers represented by the rows of {matrix} are sorted in decreasing order

    Args:
        matrix (list[list[Bool]]): matrix[i] represents an integer in binary using Z3 Bool variables

    Returns:
        (Z3-expression): constraint representing the decreasing ordering of binary numbers in the rows of matrix
    """
    m = len(matrix)
    clauses = []
    for i in range(m-1):
        clauses.append(leq(matrix[i+1], matrix[i]))
    return And(clauses)

#
# OBJECTS
#


def obj_function(model, distances):
    """Given a model, returns the objective function value that we are interested in (i.e. max of distances) as an integer

    Args:
        model (ModelRef): model of which to compute the objective function
        distances (list[list[Bool]]): list containing the binary representation (using Z3 Bool variables) of each distance

    Returns:
        int: the maximum distance travelled
    """
    m = len(distances)
    maxdist = -1
    for i in range(m):
        dist = bin_to_int([
            1 if model.evaluate(distances[i][j]) else 0
            for j in range(len(distances[i]))
        ])
        maxdist = max(maxdist, dist)
    return maxdist


def AllLessEq_bin(distances, upper_bound_bin):
    """Encodes the constraint {Forall i. distances[i] <= upper_bound_bin}

    Args:
        distances (list[list[Bool]]): list containing the binary representation (using Z3 Bool variables) of each distance
        upper_bound_bin (list[bool]): binary representation of the upper bound

    Returns:
        Z3-Expression: the constraint encoding
    """
    m = len(distances)

    clauses = []

    for i in range(m):
        clauses.append(leq(distances[i], upper_bound_bin))

    return And(clauses)


def AtLeastOneGreaterEq_bin(distances, lower_bound_bin):
    """Encodes the constraint {Exists i. distances[i] >= lower_bound_bin}

    Args:
        distances (list[list[Bool]]): list containing the binary representation (using Z3 Bool variables) of each distance
        lower_bound_bin (list[bool]): binary representation of the lower bound

    Returns:
        Z3-Expression: the constraint encoding
    """
    m = len(distances)

    clauses = []

    for i in range(m):
        clauses.append(leq(lower_bound_bin, distances[i]))

    return Or(clauses)