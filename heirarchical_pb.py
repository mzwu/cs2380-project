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
        project group -> subgroups in next layer (None if no subgroups)
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
    # TODO: paper has a bug, add in a new base case
    # Enhance with a list of the projects used for each particular utility level
    # One of the sizes of the dp is sum of all the project utilities
    pass