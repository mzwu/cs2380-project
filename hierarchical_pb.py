from itertools import chain, combinations


def solve_hierarchical_pb(layers,
                          hierarchy,
                          project_costs,
                          approvals,
                          groups,
                          global_budget,
                          group_budgets):
    """
    Hierarchical (Lemma 13)
    Inputs
    ------
    layers: list[list[str]]
        layers[i] is the list of projects in layer i (0-indexed from top to bottom)
    hierarchy : dict[str, str]
        project group -> subgroups in the next layer (None if no subgroups)
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
    # TODO: paper has a bug, add in a new base case so that on the lowermost layer the costs having utility of one group is correct
    # Enhance with a list of the projects used for each particular utility level
    # One of the sizes of the dp is sum of all the project utilities

    # Get approvals per project and maximum possible utility
    print(f"  Initializing hierarchical PB solver...")
    print(
        f"    Projects: {len(project_costs)}, Voters: {len(approvals)}, Groups: {len(groups)}")
    project_approvals = {
        p: sum(p in approval for approval in approvals) for p in project_costs}
    group_approvals = {group: sum(
        project_approvals[p] for p in groups[group]) for group in groups}
    total_utility = sum(project_approvals.values())
    print(f"    Total utility across all projects: {total_utility:,}")
    max_utility = 0
    max_utility_projects = set()

    # Initialize DP table
    # Format: group_name -> matrix(list of lists) with indices [# parts used, utility] storing min cost
    dp = {}
    projects_used = {}

    # Base cases
    print(f"  Initializing DP tables for {len(groups)} groups...")
    leaf_groups = [g for g in groups if len(hierarchy.get(g, [])) == 0]
    print(f"    Leaf groups to process: {len(leaf_groups)}")

    for group in groups:
        dp[group] = [[float('inf')] * (total_utility + 1)
                     for _ in range(len(hierarchy[group]) + 1)]
        dp[group][0][0] = 0  # Zero cost for zero parts and zero utility
        # Store projects used for each utility level
        projects_used[group] = [[set() for _ in range(total_utility + 1)]
                                for _ in range(len(hierarchy[group]) + 1)]

        if len(hierarchy[group]) == 0:
            # Leaf group: no more subgroups
            # Min cost of using this group to achieve its utility is the cost of the group itself
            dp[group].append([float('inf')] * (total_utility + 1))
            projects_used[group].append(
                [set() for _ in range(total_utility + 1)])
            # Compute utility of all subsets of projects in this leaf group

            def all_subsets(ss):
                return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

            num_projects_in_group = len(groups[group])
            num_subsets = 2 ** num_projects_in_group
            print(
                f"    Processing leaf group '{group}': {num_projects_in_group} projects, {num_subsets:,} subsets...")

            subset_count = 0
            for subset in all_subsets(groups[group]):
                subset_count += 1
                if subset_count % 100000 == 0:
                    print(
                        f"      Progress: {subset_count:,}/{num_subsets:,} subsets ({100*subset_count/num_subsets:.1f}%)")
                utility = sum(project_approvals[p] for p in subset)
                cost = sum(project_costs[p] for p in subset)
                if cost < dp[group][1][utility] and cost <= group_budgets[group]:
                    dp[group][1][utility] = cost
                    projects_used[group][1][utility] = set(subset)

    # Fill DP table layer by layer from bottom to top
    print(f"  Filling DP tables layer by layer (bottom to top)...")
    for layer_idx, layer in enumerate(reversed(layers)):
        print(f"    Layer {layer_idx + 1}/{len(layers)}: {layer}")
        for group in layer:
            subgroups = hierarchy[group]
            if len(subgroups) == 0:
                continue  # Leaf groups already initialized
            print(
                f"      Processing group '{group}' with {len(subgroups)} subgroups...")
            # Fill DP table for this group based on its subgroups
            for parts_used in range(1, len(subgroups) + 1):
                for utility in range(total_utility + 1):
                    min_cost = float('inf')
                    projects_used_temp = set()
                    # Consider all ways to split utility among parts before parts_used and parts_used
                    # Equations from recursive step
                    for u in range(utility + 1):
                        # print(f"  Trying utility split: {utility - u} (before part {parts_used}) + {u} (current part group {subgroups[parts_used - 1]})")
                        cost_before = dp[group][parts_used - 1][utility - u]
                        current_part = subgroups[parts_used - 1]
                        cost_current = dp[current_part][len(
                            hierarchy[current_part]) + 1][u]
                        total_cost = cost_before + cost_current

                        # Update minimum cost and projects used if this split is better
                        if total_cost < min_cost:
                            min_cost = total_cost
                            projects_used_temp = projects_used[group][parts_used - 1][utility - u].union(
                                projects_used[current_part][len(
                                    hierarchy[current_part]) + 1][u]
                            )
                    # Check if there is some subset of projects that meet the requirements and budget constraint
                    if min_cost > group_budgets[group]:
                        min_cost = float('inf')  # Exceeds group budget

                    # Update DP table and projects used
                    dp[group][parts_used][utility] = min_cost
                    projects_used[group][parts_used][utility] = projects_used_temp

                    # Update maximum utility and the associated projects found so far
                    if min_cost <= global_budget and utility > max_utility:
                        max_utility = utility
                        max_utility_projects = projects_used_temp
    # print(dp)
    # print(projects_used)
    total_cost = sum(project_costs[p] for p in max_utility_projects)
    print(
        f"  Done! Max utility: {max_utility:,}, Projects selected: {len(max_utility_projects)}, Total cost: {total_cost:,}")
    # meta can be filled with additional info if needed
    return max_utility_projects, max_utility, {"total cost": total_cost}


# Test case
if __name__ == "__main__":
    import json

    # Load region assignments (layers, hierarchy, groups, group_budgets)
    with open("data/region_assignments_wesola.json", "r") as f:
        region_data = json.load(f)

    # Load Warsaw PB data (project_costs, approvals)
    with open("data/poland_warszawa_2023_wesola.json", "r") as f:
        pb_data = json.load(f)

    # Extract and convert data to expected formats
    layers = region_data["layers"]
    hierarchy = region_data["hierarchy"]
    project_costs = {k: int(v) for k, v in pb_data["project_costs"].items()}
    approvals = [set(a) for a in pb_data["approvals"]]
    groups = {k: set(v) for k, v in region_data["groups"].items()}
    global_budget = region_data["group_budgets"]["All"]
    group_budgets = region_data["group_budgets"]
    chosen_projects, utility, meta = solve_hierarchical_pb(
        layers,
        hierarchy,
        project_costs,
        approvals,
        groups,
        global_budget,
        group_budgets
    )
    print(f"Chosen Projects: {chosen_projects}")
    print(f"Total Utility: {utility}")
    print(f"Meta Info: {meta}")

    # Save results to JSON
    result = {
        "chosen_projects": list(chosen_projects),
        "utility": utility,
        "total_cost": meta["total cost"]
    }

    with open("data/hierarchical_pb_result_wesola.json", "w") as f:
        json.dump(result, f, indent=4)
