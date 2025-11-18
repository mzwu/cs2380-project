# Max-Group-PB: (g+2)-approx via LP relaxation + rounding

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.optimize import linprog

def max_group_pb_approx(
    projects: List[Dict[str, Any]],
    group_budgets: Dict[Any, float],
    global_budget: float,
    tol: float = 1e-9,
) -> Tuple[List[Any], float, float]:
    """
    Inputs:
    - projects: list of dicts, each with keys 
        "id" (str), 
        "cost" (float), 
        "utility" (float), 
        optional "groups" (list of group ID strings)
    - group_budgets: dict mapping group ID (str) to budget (float)
    - global_budget: overall budget (float)
    - tol: tolerance for going over budgets

    Returns: list of
        chosen_project_ids (list of str), 
        total_utility (float), 
        total_cost (float)

    Approximation factor: (g+2) where g = number of groups.
    Based on d-dimensional knapsack LP rounding with d = g+1 constraints
    (all group budgets + global budget). We select the better of:
      - all items with LP value x_p = 1 (rounded-down set),
      - the single best fractional item alone.
    Corollary 35: d-DK can be approximated in polynomial time up to d+1 factor
    so we have a polytime (g+2)-approximation for Max-Group-PB
    """
    # Build indexing
    pid = [p["id"] for p in projects]
    cost = [float(p["cost"]) for p in projects]
    util = [float(p["utility"]) for p in projects]
    groups_per_item = [set(p.get("groups", [])) for p in projects]

    group_ids = list(group_budgets.keys())
    g = len(group_ids)  # number of group constraints
    n = len(projects)

    # Check feasibility for staying in budget with projects in S
    def feasible(S):
        total_cost = sum(cost[i] for i in S)
        if total_cost - global_budget > tol:
            return False
        for F in group_ids:
            F_cost = sum(cost[i] for i in S if F in groups_per_item[i])
            if F_cost - group_budgets[F] > tol:
                return False
        return True

    # Try to solve LP relaxation: maximize sum util_i * x_i
    # subject to:
    #    sum cost_i * x_i <= global_budget
    #    for each group F: sum cost_i * x_i (i s.t. F in item i) <= group_budget(F)
    #    0 <= x_i <= 1
    # Variables:
    #    util_i: utility of project i
    #    cost_i: cost of project i
    #    x_i: decision variable, 1 if project i is funded, else 0
    x = [0.0] * n

    c = -np.array(util, dtype=float)  # linprog minimizes
    # Ax <= b
    A = []
    b = []

    # Global budget
    A.append(np.array(cost, dtype=float))
    b.append(global_budget)

    # Group budgets
    for F in group_ids:
        row = [cost[i] if F in groups_per_item[i] else 0.0 for i in range(n)]
        A.append(np.array(row, dtype=float))
        b.append(group_budgets[F])

    A = np.vstack(A)
    b = np.array(b, dtype=float)

    # Bounds 0 <= x_i <= 1
    bounds = [(0.0, 1.0)] * n

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if res.success and res.x is not None:
        # convert solution to floats
        x = list(map(float, res.x))

    # Identify integral and fractional variables in the basic solution
    ones = [i for i, xi in enumerate(x) if xi >= 1 - tol]
    zeros = [i for i, xi in enumerate(x) if xi <= tol]
    fracs = [i for i, xi in enumerate(x) if tol < xi < 1 - tol]

    # Candidate A: rounded-down set (all x_i == 1)
    candA = set(ones)
    if not feasible(candA):
        # greedily add while feasible
        A_list = sorted(list(candA), key=lambda i: util[i] / (cost[i] + 1e-12), reverse=True)
        fix = set()
        for i in A_list:
            fix.add(i)
            if feasible(fix):
                continue
        # drop worst density items until feasible
        cur = set(A_list)
        while not feasible(cur) and cur:
            worst = min(cur, key=lambda i: util[i] / (cost[i] + 1e-12))
            cur.remove(worst)
        candA = cur

    utilA = sum(util[i] for i in candA)

    # Candidate B: best single fractional item alone (must be feasible by itself)
    best_single = None
    utilB = -1.0
    for i in fracs:
        if feasible({i}) and util[i] > utilB:
            utilB = util[i]
            best_single = i

    if best_single is not None and utilB > utilA + 1e-12:
        chosen = {best_single}
    else:
        chosen = set(candA)

    return [pid[i] for i in chosen], sum(util[i] for i in chosen), sum(cost[i] for i in chosen)


if __name__ == "__main__":
    projects = [
        {"id": "p1", "cost": 4, "utility": 15, "groups": ["G1"]},
        {"id": "p2", "cost": 3, "utility": 10, "groups": ["G1","G2"]},
        {"id": "p3", "cost": 5, "utility": 12, "groups": ["G2"]},
        # ...
    ]
    group_budgets = {"G1": 5, "G2": 6}
    B = 9

    chosen_ids, total_utility, total_cost = max_group_pb_approx(projects, group_budgets, B)
    print(chosen_ids, total_utility, total_cost)
