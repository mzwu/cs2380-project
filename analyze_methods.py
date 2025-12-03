"""
Analysis script for Participatory Budgeting methods.
Analyzes:
1. Total utility (each approved project per person = 1 point)
2. How much budget is not used
3. EJR (Extended Justified Representation) violations via brute force
"""

import json
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    """Load PB data and results from all three methods."""
    # Load input data
    with open("results/poland_warszawa_2023_wesola.json", "r") as f:
        pb_data = json.load(f)

    with open("results/region_assignments_wesola.json", "r") as f:
        region_data = json.load(f)

    # Load results
    with open("results/hierarchical_pb_result_wesola.json", "r") as f:
        hierarchical_result = json.load(f)

    with open("results/max_group_pb_result_wesola.json", "r") as f:
        max_group_result = json.load(f)

    with open("results/equal_shares_result_wesola.json", "r") as f:
        equal_shares_result = json.load(f)

    return pb_data, region_data, hierarchical_result, max_group_result, equal_shares_result


def compute_utility(chosen_projects, approvals):
    """
    Compute total utility: each approved project per person = 1 point.
    Total utility = sum over all voters of (# of their approved projects that are funded).
    """
    chosen_set = set(chosen_projects)
    total_utility = 0
    for voter_approvals in approvals:
        # Count how many of this voter's approved projects are in the chosen set
        voter_utility = len(set(voter_approvals) & chosen_set)
        total_utility += voter_utility
    return total_utility


def compute_unused_budget(chosen_projects, project_costs, total_budget):
    """Compute how much budget is not used."""
    used_budget = sum(int(project_costs[p]) for p in chosen_projects)
    return total_budget - used_budget


def check_ejr_violations(chosen_projects, approvals, project_costs, total_budget, max_ell=5, verbose=False):
    """
    Check for EJR (Extended Justified Representation) violations using brute force.

    EJR Definition:
    If there is S ⊆ N such that |S| ≥ (n/k) · ℓ and |∩_{i∈S} α_i| ≥ ℓ
    then ∃i ∈ S such that u_i(W) ≥ ℓ

    Where:
    - N = set of all voters, n = |N|
    - S = a cohesive group of voters
    - α_i = approval ballot of voter i
    - ∩_{i∈S} α_i = projects approved by ALL voters in S
    - k = total budget B (treating each unit of budget as a "seat")
    - ℓ = entitlement level (positive integer)
    - W = winning set of projects
    - u_i(W) = |W ∩ α_i| = number of funded projects voter i approves

    A VIOLATION occurs when:
    - A group S has |S| ≥ (n/B) · ℓ · avg_cost (adjusted for PB)
    - AND |∩_{i∈S} α_i| ≥ ℓ (they unanimously approve ≥ ℓ projects)
    - BUT no voter in S has u_i(W) ≥ ℓ

    Parameters:
    - max_ell: Maximum value of ℓ to check
    - verbose: Print progress

    Returns:
    - List of violations found
    """
    n = len(approvals)
    B = total_budget
    chosen_set = set(chosen_projects)

    # Precompute voter data
    voter_approved = [set(a) for a in approvals]
    voter_utility = [len(voter_approved[i] & chosen_set) for i in range(n)]

    # For each project, which voters approve it
    project_approvers = defaultdict(set)
    for i, approvals_set in enumerate(voter_approved):
        for p in approvals_set:
            project_approvers[p].add(i)

    all_projects = list(project_costs.keys())
    violations = []
    checked_groups = 0

    # For PB, we adapt: a group S "deserves" ℓ projects if they can afford ℓ projects
    # with their proportional budget share: |S|/n * B ≥ cost of ℓ cheapest projects they all approve

    # For each ℓ from 1 to max_ell
    for ell in range(1, max_ell + 1):
        if verbose:
            print(f"      Checking ℓ = {ell}...")

        # Enumerate all subsets of ℓ projects
        for T in combinations(all_projects, ell):
            T_set = set(T)
            T_cost = sum(int(project_costs[p]) for p in T)

            # Find S = voters who approve ALL projects in T (i.e., ∩_{i∈S} α_i ⊇ T)
            S = None
            for p in T:
                if S is None:
                    S = project_approvers[p].copy()
                else:
                    S &= project_approvers[p]

            if not S:
                continue

            checked_groups += 1

            # Check the EJR condition:
            # |S| ≥ (n/B) · ℓ means the group deserves ℓ "units"
            # In PB: |S| · B / n ≥ T_cost (their budget share covers T)
            group_budget_share = len(S) * B / n

            # Also check: |∩_{i∈S} α_i| ≥ ℓ
            # The intersection contains at least T, so it has at least ℓ projects
            # But we should check actual intersection size
            common_approvals = voter_approved[list(S)[0]].copy()
            for voter_idx in S:
                common_approvals &= voter_approved[voter_idx]

            # EJR condition: |S| ≥ n·ℓ/k where k relates to budget
            # Adapted: group deserves ℓ if their share covers cost of ℓ projects
            if T_cost <= group_budget_share and len(common_approvals) >= ell:
                # Group S deserves ℓ projects
                # Check if ∃i ∈ S such that u_i(W) ≥ ℓ
                max_utility_in_S = max(voter_utility[v] for v in S)

                if max_utility_in_S < ell:
                    # VIOLATION!
                    violations.append({
                        "ell": ell,
                        "projects_T": list(T),
                        "T_cost": T_cost,
                        "group_size": len(S),
                        "group_budget_share": group_budget_share,
                        "common_approvals_count": len(common_approvals),
                        "required_utility": ell,
                        "max_utility_in_group": max_utility_in_S,
                        "voter_indices": list(S)[:10]
                    })

    if verbose:
        print(f"      Total cohesive groups checked: {checked_groups}")
        print(f"      Violations found: {len(violations)}")

    return violations


def check_ejr_simple(chosen_projects, approvals, project_costs, total_budget, verbose=False):
    """
    Simplified EJR check based on the formal definition:

    EJR: If |S| ≥ (n/k)·ℓ and |∩_{i∈S} α_i| ≥ ℓ, then ∃i ∈ S: u_i(W) ≥ ℓ

    This version iterates over possible group sizes and checks if any
    cohesive group is underrepresented.
    """
    n = len(approvals)
    B = total_budget
    chosen_set = set(chosen_projects)

    # Precompute
    voter_approved = [set(a) for a in approvals]
    voter_utility = [len(voter_approved[i] & chosen_set) for i in range(n)]

    violations = []

    # For ℓ = 1, 2, 3, ... check if any group is underrepresented
    # The threshold is: |S| ≥ (n/B) · ℓ · (some cost factor)

    # Simpler approach: for each ℓ, minimum group size = ℓ * n / num_affordable_projects
    # Or just use: if |S| voters all approve the same ℓ projects, and |S| is "large enough"

    # Get average project cost for threshold calculation
    avg_cost = sum(int(c) for c in project_costs.values()) / len(project_costs)

    for ell in range(1, 6):
        # Threshold: group needs |S| ≥ ℓ · n · avg_cost / B to "deserve" ℓ projects
        min_group_size = ell * n * avg_cost / B

        if verbose:
            print(f"      ℓ={ell}: min group size = {min_group_size:.1f}")

        # Find all maximal cohesive groups of size ≥ min_group_size
        # that unanimously approve ≥ ℓ projects
        # This is expensive, so we sample by looking at project combinations

        all_projects = list(project_costs.keys())
        for T in combinations(all_projects, ell):
            # Find voters who approve all projects in T
            S = set(range(n))
            for p in T:
                S &= set(i for i in range(n) if p in voter_approved[i])

            if len(S) >= min_group_size:
                # Check if any voter in S has utility ≥ ℓ
                max_util = max(voter_utility[v] for v in S)
                if max_util < ell:
                    violations.append({
                        "ell": ell,
                        "projects": list(T),
                        "group_size": len(S),
                        "min_required": min_group_size,
                        "max_utility": max_util
                    })

    return violations


def analyze_method(name, chosen_projects, pb_data, total_budget):
    """Analyze a single method and return results."""
    approvals = [set(a) for a in pb_data["approvals"]]
    project_costs = pb_data["project_costs"]

    print(f"\n{'='*60}")
    print(f"Analysis for: {name}")
    print(f"{'='*60}")

    # 1. Total Utility
    utility = compute_utility(chosen_projects, approvals)
    print(f"\n1. TOTAL UTILITY")
    print(f"   Chosen projects: {len(chosen_projects)}")
    print(f"   Total utility (sum of approved projects per voter): {utility}")

    # 2. Unused Budget
    unused = compute_unused_budget(
        chosen_projects, project_costs, total_budget)
    used = total_budget - unused
    print(f"\n2. BUDGET USAGE")
    print(f"   Total budget: {total_budget:,}")
    print(f"   Used budget: {used:,}")
    print(f"   Unused budget: {unused:,} ({100*unused/total_budget:.2f}%)")

    # 3. EJR Violations
    print(f"\n3. EJR VIOLATION CHECK")
    print(f"   Definition: If |S| ≥ (n/B)·ℓ·cost and |∩_{{i∈S}} α_i| ≥ ℓ,")
    print(f"               then ∃i ∈ S such that u_i(W) ≥ ℓ")

    # Check EJR with the formal definition
    print(f"\n   Checking EJR for ℓ = 1 to 5...")
    violations = check_ejr_violations(
        chosen_projects,
        pb_data["approvals"],
        project_costs,
        total_budget,
        max_ell=5,
        verbose=True
    )

    if violations:
        print(f"\n   FOUND {len(violations)} EJR VIOLATIONS!")
        print(f"\n   Sample violations (first 5):")
        for i, v in enumerate(violations[:5]):
            print(f"\n   Violation {i+1} (ℓ={v['ell']}):")
            print(f"     Projects T: {v['projects_T']}")
            print(f"     Cost of T: {v['T_cost']:,}")
            print(f"     Group size |S|: {v['group_size']}")
            print(f"     Group budget share: {v['group_budget_share']:,.2f}")
            print(f"     Common approvals: {v['common_approvals_count']}")
            print(f"     Required utility ≥: {v['required_utility']}")
            print(f"     Max utility in group: {v['max_utility_in_group']}")
    else:
        print(f"\n   NO EJR violations found!")
        print(f"   All cohesive groups are adequately represented.")

    # Also run simplified check
    print(f"\n   Running simplified EJR check...")
    simple_violations = check_ejr_simple(
        chosen_projects,
        pb_data["approvals"],
        project_costs,
        total_budget,
        verbose=True
    )

    if simple_violations:
        print(
            f"   Found {len(simple_violations)} violations in simplified check.")
    else:
        print(f"   Simplified check also confirms no violations.")

    total_violations = len(violations) + len(simple_violations)

    return {
        "method": name,
        "num_projects": len(chosen_projects),
        "utility": utility,
        "used_budget": used,
        "unused_budget": unused,
        "unused_percentage": 100*unused/total_budget,
        "ejr_violations": total_violations,
        "ejr_formal_violations": len(violations),
        "ejr_simple_violations": len(simple_violations),
        "violation_details": violations[:10],
        "simple_violation_details": simple_violations[:10]
    }


def main():
    # Load all data
    pb_data, region_data, hierarchical_result, max_group_result, equal_shares_result = load_data()

    total_budget = region_data["group_budgets"]["All"]

    print("\n" + "="*60)
    print("PARTICIPATORY BUDGETING METHODS ANALYSIS")
    print("="*60)
    print(f"\nTotal Budget: {total_budget:,}")
    print(f"Number of Voters: {len(pb_data['approvals'])}")
    print(f"Number of Projects: {len(pb_data['project_costs'])}")

    results = []

    # Analyze each method
    results.append(analyze_method(
        "Hierarchical PB",
        hierarchical_result["chosen_projects"],
        pb_data,
        total_budget
    ))

    results.append(analyze_method(
        "Approx Max Group PB",
        max_group_result["chosen_projects"],
        pb_data,
        total_budget
    ))

    results.append(analyze_method(
        "Method of Equal Shares",
        equal_shares_result["chosen_projects"],
        pb_data,
        total_budget
    ))

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"\n{'Method':<25} {'Projects':<10} {'Utility':<10} {'Unused Budget':<15} {'EJR Violations':<15}")
    print("-"*75)
    for r in results:
        print(f"{r['method']:<25} {r['num_projects']:<10} {r['utility']:<10} {r['unused_budget']:<15,} {r['ejr_violations']:<15}")

    # Save results to JSON
    output = {
        "total_budget": total_budget,
        "num_voters": len(pb_data["approvals"]),
        "num_projects": len(pb_data["project_costs"]),
        "results": results
    }

    with open("results/analysis_results_wesola.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nDetailed results saved to results/analysis_results_wesola.json")

    # Generate graphs
    generate_graphs(results, total_budget)


def generate_graphs(results, total_budget):
    """Generate individual visualization graphs for the analysis results."""

    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#2ecc71', '#3498db', '#9b59b6']  # Green, Blue, Purple

    methods = [r['method'] for r in results]
    short_names = ['Hierarchical', 'Max Group', 'Equal Shares']

    # 1. Total Utility Bar Chart
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    utilities = [r['utility'] for r in results]
    bars1 = ax1.bar(short_names, utilities, color=colors,
                    edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Total Utility', fontsize=14)
    ax1.set_title('Total Utility\n(Sum of Approved Projects per Voter)',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylim(0, max(utilities) * 1.15)

    # Add value labels on bars
    for bar, val in zip(bars1, utilities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Highlight the best
    best_utility_idx = utilities.index(max(utilities))
    bars1[best_utility_idx].set_edgecolor('#e74c3c')
    bars1[best_utility_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('data/graph_01_total_utility_wesola.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("📊 Saved: graph_01_total_utility_wesola.png")

    # 2. Budget Usage Stacked Bar Chart
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    used = [r['used_budget'] for r in results]
    unused = [r['unused_budget'] for r in results]

    x = np.arange(len(short_names))
    width = 0.6

    bars_used = ax2.bar(x, used, width, label='Used Budget',
                        color='#27ae60', edgecolor='black')
    bars_unused = ax2.bar(x, unused, width, bottom=used, label='Unused Budget',
                          color='#e74c3c', edgecolor='black', alpha=0.7)

    ax2.set_ylabel('Budget (PLN)', fontsize=14)
    ax2.set_title('Budget Usage', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=12)
    # Legend removed per user request
    ax2.axhline(y=total_budget, color='black', linestyle='--',
                linewidth=1.5)

    # Add percentage labels
    for i, (u, un) in enumerate(zip(used, unused)):
        pct = un / total_budget * 100
        ax2.text(i, total_budget + 10000, f'{pct:.1f}% unused',
                 ha='center', va='bottom', fontsize=11, color='#c0392b', fontweight='bold')

    plt.tight_layout()
    plt.savefig('data/graph_02_budget_usage_wesola.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("📊 Saved: graph_02_budget_usage_wesola.png")

    # 3. Unused Budget Comparison
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    unused_pct = [r['unused_percentage'] for r in results]
    bars3 = ax3.bar(short_names, unused_pct, color=colors,
                    edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Unused Budget (%)', fontsize=14)
    ax3.set_title('Budget Efficiency\n(Lower is Better)',
                  fontsize=16, fontweight='bold', pad=20)
    ax3.set_ylim(0, max(unused_pct) * 1.3)

    # Add value labels
    for bar, val in zip(bars3, unused_pct):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Highlight the best (lowest)
    best_efficiency_idx = unused_pct.index(min(unused_pct))
    bars3[best_efficiency_idx].set_edgecolor('#e74c3c')
    bars3[best_efficiency_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('data/graph_03_budget_efficiency_wesola.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("📊 Saved: graph_03_budget_efficiency_wesola.png")

    # 4. Projects Funded & EJR Status
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    num_projects = [r['num_projects'] for r in results]
    ejr_violations = [r['ejr_violations'] for r in results]

    x = np.arange(len(short_names))
    width = 0.35

    bars4a = ax4.bar(x - width/2, num_projects, width, label='Projects Funded',
                     color='#3498db', edgecolor='black')
    bars4b = ax4.bar(x + width/2, ejr_violations, width, label='EJR Violations',
                     color='#e74c3c', edgecolor='black')

    ax4.set_ylabel('Count', fontsize=14)
    ax4.set_title('Projects Funded & EJR Violations',
                  fontsize=16, fontweight='bold', pad=20)
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_names, fontsize=12)
    # Legend removed per user request

    # Add value labels
    for bar, val in zip(bars4a, num_projects):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')

    for bar, val in zip(bars4b, ejr_violations):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add checkmarks for EJR compliance
    for i, violations in enumerate(ejr_violations):
        if violations == 0:
            ax4.text(x[i] + width/2, 0.5, 'OK', ha='center', va='bottom',
                     fontsize=12, color='#27ae60', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/graph_04_projects_ejr_wesola.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("📊 Saved: graph_04_projects_ejr_wesola.png")

    # Create a summary radar chart
    create_radar_chart(results)


def create_radar_chart(results):
    """Create a radar chart comparing methods across multiple metrics."""

    # Normalize metrics to 0-100 scale
    methods = ['Hierarchical', 'Max Group', 'Equal Shares']
    metrics = ['Utility', 'Budget\nEfficiency',
               'Projects\nFunded', 'EJR\nCompliance']

    # Get values
    utilities = [r['utility'] for r in results]
    efficiency = [100 - r['unused_percentage']
                  for r in results]  # Higher is better
    projects = [r['num_projects'] for r in results]
    ejr = [100 if r['ejr_violations'] == 0 else max(
        0, 100 - r['ejr_violations'] * 10) for r in results]

    # Normalize to 0-100
    def normalize(vals):
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return [100] * len(vals)
        return [(v - min_v) / (max_v - min_v) * 100 for v in vals]

    utilities_norm = normalize(utilities)
    efficiency_norm = normalize(efficiency)
    projects_norm = normalize(projects)
    ejr_norm = ejr  # Already 0-100

    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    colors = ['#2ecc71', '#3498db', '#9b59b6']

    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [utilities_norm[i], efficiency_norm[i],
                  projects_norm[i], ejr_norm[i]]
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title('Multi-Metric Comparison\n(Normalized Scores, Higher is Better)',
                 fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(
        1.25, 1.15), fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('results/graph_05_radar_chart_wesola.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("📊 Saved: graph_05_radar_chart_wesola.png")


if __name__ == "__main__":
    main()
