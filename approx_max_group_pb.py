from collections import defaultdict
import math
import json

def approx_max_group_pb(project_costs,
                 approvals,
                 groups,
                 global_budget,
                 group_budgets,
                 epsilon=0.10):
    """
    (1+epsilon)-approximation for Max-Group-PB (Theorem 26)
    Inputs
    ------
    project_costs : dict[str, int]
        project -> nonnegative integer cost
    approvals : list[set[str]]
        approvals[v] is the set of projects approved by voter v
    groups : dict[str, set[str]]
        group_name -> set of projects in that group (groups may overlap)
    global_budget : int
        Global budget limit B
    group_budgets : dict[str, int]
        group_name -> budget limit b(F) for that group
    epsilon : float
        Accuracy parameter for the (1+epsilon)-approximation

    Returns
    -------
    chosen_projects : set[str]
        The set of funded projects
    utility : int
        Sum of approvals (i.e., number of approved projects funded, summed over voters)
    meta : dict
        Extra info (e.g. total cost used)
    """
    print("----- STARTING approx_max_group_pb -----")

    # Setup and sanity checks
    P = set(project_costs.keys())
    G = list(groups.keys())
    gset = set(G)

    for F in G:
        if F not in group_budgets:
            raise ValueError(f"Missing budget for group {F}")
    if not isinstance(global_budget, int) or global_budget < 0:
        raise ValueError("Global budget must be a nonnegative integer")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    # Calculate approval scores a(p) for each project p
    approval_score = defaultdict(int)
    for ballot in approvals:
        for p in ballot:
            if p in P:
                approval_score[p] += 1

    # Drop projects with zero utility
    P = {p for p in P if approval_score[p] > 0}
    if not P:
        return set(), 0, {"note": "No project has positive utility."}

    # Type R of each project (subset of groups it belongs to)
    print("----- COMPUTING PROJECT TYPES -----")
    proj_groups = {p: frozenset({F for F in G if p in groups[F]}) for p in P}
    type_to_projects = defaultdict(list)
    for p in P:
        type_to_projects[proj_groups[p]].append(p)
    types = list(type_to_projects.keys())  # at most 2^g types

    # Total approval mass A
    A_total = sum(approval_score[p] for p in P)

    # For each type R, compute min-cost for exact utility v and store bundle (DP)
    def min_cost_bundles_for_type(R):
        items = [(p, project_costs[p], approval_score[p]) for p in type_to_projects[R]]
        INF = 10**18
        dp_cost = [INF] * (A_total + 1)
        dp_prev = [None] * (A_total + 1)
        dp_cost[0] = 0  # cost 0 for utility 0

        for p, c, a in items:
            for v in range(A_total - a, -1, -1):
                if dp_cost[v] == INF:
                    continue
                nv = v + a
                nc = dp_cost[v] + c
                if nc < dp_cost[nv]:
                    dp_cost[nv] = nc  # minimum cost to reach total utility v
                    dp_prev[nv] = (v, p)  # to get utility nv, we came from utility v by adding project p

        bundles = {0: (0, set())}  # utility v -> (min_cost, bundle_set)
        for v in range(1, A_total + 1):
            if dp_cost[v] < INF:
                chosen = set()
                cur = v
                while cur != 0:
                    prev_v, p = dp_prev[cur]
                    chosen.add(p)
                    cur = prev_v
                bundles[v] = (dp_cost[v], chosen)
        return bundles

    print("----- MIN COST BUNDLES -----")
    type_options_exact = {R: min_cost_bundles_for_type(R) for R in types}
    # Make sure each type has 0 cost 0 utility option
    for R in types:
        if 0 not in type_options_exact[R]:
            type_options_exact[R][0] = (0, set())

    # Geometric bucketing (logarithmic in u)
    # Bucket index k = floor(log_{1+epsilon}(u)), with u=0 mapped to k=0
    # Keep the cheapest option per bucket index (tie-break by larger original u)
    def bucket_index(u):
        if u <= 0:
            return 0
        return int(math.floor(math.log(u) / math.log(1.0 + epsilon)))

    print("----- BUCKETING -----")
    type_options_bucketed = {}
    for R in types:
        best_for_bucket = {}  # k -> (cost, bundle, original_u)
        for u, (cost, bundle) in type_options_exact[R].items():
            k = bucket_index(u)
            rec = best_for_bucket.get(k)
            # Choose if cheaper cost, or same cost but larger original utility
            if (rec is None or cost < rec[0] or (cost == rec[0] and u > rec[2])):
                best_for_bucket[k] = (cost, bundle, u)

        # Build option list for this type: (bucket_k, cost, bundle, original_u)
        opts = [(k, cost, bundle, u) for k, (cost, bundle, u) in best_for_bucket.items()]
        # Ensure a zero option remains
        if 0 not in best_for_bucket:
            opts.append((0, 0, set(), 0))
        # Sort by (cost asc, original_u desc) to encourage early feasible picks
        opts.sort(key=lambda x: (x[1], -x[3]))
        type_options_bucketed[R] = opts

    # Group type -> which groups belong to it
    group_set_per_type = {R: set(R) for R in types}

    # Enumerate one option per type (branch & bound DFS)
    best_solution = {
        "utility": -1,  # actual (unbucketed) utility
        "bucket_utility": -1,
        "cost": None,
        "projects": set(),
        "choices": {}
    }

    types_sorted = sorted(types, key=lambda R: len(type_options_bucketed[R]))

    # Search over types: pick one compressed option per type, obey budgets, and keep the best utility
    def dfs(i, used_cost, used_group_costs, bucket_util, actual_util, chosen_projects, choices):
        # i = which type we're assigning now
        nonlocal best_solution

        # Cut branches that exceed group or global budgets
        if used_cost > global_budget:
            return
        for F, used in used_group_costs.items():
            if used > group_budgets[F]:
                return

        # Leaf: all types assigned
        # Compare to best_solution, choose higher utility (tie-break by lower cost)
        if i == len(types_sorted):
            if actual_util > best_solution["utility"] or (
                actual_util == best_solution["utility"] and (best_solution["cost"] is None or used_cost < best_solution["cost"])
            ):
                best_solution = {
                    "utility": actual_util,
                    "bucket_utility": bucket_util,
                    "cost": used_cost,
                    "projects": set(chosen_projects),
                    "choices": dict(choices),
                }
            return

        # Iterate through the current type's options
        R = types_sorted[i]
        for b_k, cost, bundle, orig_u in type_options_bucketed[R]:
            n_used_cost = used_cost + cost

            # Copy-on-write for group costs only when needed
            if group_set_per_type[R]:
                # Add cost for every group this type belongs to
                n_used_group_costs = dict(used_group_costs)
                for F in group_set_per_type[R]:
                    n_used_group_costs[F] = n_used_group_costs.get(F, 0) + cost
            else:
                n_used_group_costs = used_group_costs

            n_bucket_util = bucket_util + b_k
            n_actual_util = actual_util + orig_u
            n_chosen_projects = chosen_projects | bundle
            choices[R] = (b_k, cost, set(bundle), orig_u)

            dfs(i + 1, n_used_costs := n_used_cost, n_used_group_costs, n_bucket_util, n_actual_util, n_chosen_projects, choices)

            del choices[R]

    print("----- DFS TO GET BEST BUNDLE -----")
    start_group_costs = {F: 0 for F in gset}
    dfs(0, 0, start_group_costs, 0, 0, set(), {})

    return best_solution["projects"], best_solution["utility"], {
        "total_cost": best_solution["cost"],
        "bucket_utility": best_solution["bucket_utility"],
        "epsilon": epsilon,
        "notes": "Utility is exact (sum of a(p) on returned projects). Budget constraints are strictly enforced. Options are geometrically bucketed."
    }



if __name__ == "__main__":
    with open("data/poland_warszawa_2026_marysin-wawerski-anin.json", "r") as f:
        data = json.load(f)

        # Access data
        project_costs = data["project_costs"]
        approvals = [set(a) for a in data["approvals"]]
        groups = {k: set(v) for k, v in data["groups"].items()}

    group_budgets = {group: 725604 // len(groups) for group in groups}
    B = sum([group_budgets[F] for F in groups])

    chosen, util, meta = approx_max_group_pb(project_costs, approvals, groups, B, group_budgets, epsilon=0.10)
    print("Chosen projects:", chosen)
    print("Utility:", util)
    print("Total cost used:", meta["total_cost"])

    data = {
        "chosen_projects": list(chosen),
        "utility": util,
        "total_cost": meta["total_cost"]
    }

    # Save to a JSON file
    with open("data/max_group_pb_result.json", "w") as file:
        json.dump(data, file, indent=4)
