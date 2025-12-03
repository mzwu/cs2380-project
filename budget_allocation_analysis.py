"""
Script to analyze the best and worst budget allocations for hierarchical PB 
and approx_max_group_pb methods by enumerating all possible allocations
in 20% increments. Measures performance by budget usage (higher = better).

Usage:
    python budget_allocation_analysis.py          # Run full analysis
    python budget_allocation_analysis.py --plots  # Only regenerate plots from saved data
"""

import json
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from hierarchical_pb import solve_hierarchical_pb
from approx_max_group_pb import approx_max_group_pb

RESULTS_FILE = "results/budget_allocation_results.json"


def generate_percentage_allocations(n_groups, increment=20):
    """
    Generate all possible percentage allocations for n_groups that sum to 100%.
    Each allocation is in increments of `increment`%.

    Returns a list of tuples, each tuple containing n_groups percentages.
    """
    percentages = list(range(0, 101, increment))
    allocations = []

    # Generate all combinations that sum to 100
    for combo in itertools.product(percentages, repeat=n_groups):
        if sum(combo) == 100:
            allocations.append(combo)

    return allocations


def run_hierarchical_analysis(region_data, pb_data, total_budget, allocations, region_names):
    """
    Run hierarchical PB for all budget allocations and return results.
    """
    results = []
    total = len(allocations)

    layers = region_data["layers"]
    hierarchy = region_data["hierarchy"]
    project_costs = {k: int(v) for k, v in pb_data["project_costs"].items()}
    approvals = [set(a) for a in pb_data["approvals"]]
    groups = {k: set(v) for k, v in region_data["groups"].items()}

    for idx, alloc in enumerate(allocations):
        if (idx + 1) % 5 == 0 or idx == 0:
            print(
                f"  Hierarchical: Processing allocation {idx + 1}/{total}...")

        # Create group budgets based on allocation percentages
        group_budgets = {"All": total_budget}
        for i, region in enumerate(region_names):
            group_budgets[region] = int(total_budget * alloc[i] / 100)

        try:
            chosen_projects, utility, meta = solve_hierarchical_pb(
                layers, hierarchy, project_costs, approvals, groups,
                total_budget, group_budgets
            )
            cost = meta["total cost"]
            budget_pct = (cost / total_budget) * 100 if total_budget > 0 else 0
            results.append({
                "allocation": alloc,
                "utility": utility,
                "cost": cost,
                "budget_pct": budget_pct,
                "projects": chosen_projects
            })
        except Exception as e:
            results.append({
                "allocation": alloc,
                "utility": 0,
                "cost": 0,
                "budget_pct": 0,
                "projects": set(),
                "error": str(e)
            })

    return results


def run_approx_max_group_analysis(pb_data, total_budget, allocations, category_names):
    """
    Run approx_max_group_pb for all budget allocations and return results.
    """
    results = []
    total = len(allocations)

    project_costs = {k: int(v) for k, v in pb_data["project_costs"].items()}
    approvals = [set(a) for a in pb_data["approvals"]]
    groups = {k: set(v) for k, v in pb_data["groups"].items()}

    for idx, alloc in enumerate(allocations):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(
                f"  Approx Max Group: Processing allocation {idx + 1}/{total}...")

        # Create group budgets based on allocation percentages
        group_budgets = {}
        for i, category in enumerate(category_names):
            group_budgets[category] = int(total_budget * alloc[i] / 100)

        # Global budget is sum of group budgets
        global_budget = sum(group_budgets.values())

        try:
            chosen_projects, utility, meta = approx_max_group_pb(
                project_costs, approvals, groups, global_budget, group_budgets, epsilon=0.10
            )
            cost = meta["total_cost"] if meta["total_cost"] else 0
            budget_pct = (cost / total_budget) * 100 if total_budget > 0 else 0
            results.append({
                "allocation": alloc,
                "utility": utility,
                "cost": cost,
                "budget_pct": budget_pct,
                "projects": chosen_projects
            })
        except Exception as e:
            results.append({
                "allocation": alloc,
                "utility": 0,
                "cost": 0,
                "budget_pct": 0,
                "projects": set(),
                "error": str(e)
            })

    return results


def analyze_results(results, group_names):
    """
    Analyze results to find best and worst allocations based on budget usage.
    """
    if not results:
        return None, None, []

    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        return None, None, []

    # Best = highest budget usage, Worst = lowest budget usage
    best = max(valid_results, key=lambda x: x["budget_pct"])
    worst = min(valid_results, key=lambda x: x["budget_pct"])

    return best, worst, valid_results


def format_allocation(alloc, group_names):
    """Format allocation as a readable string."""
    return ", ".join([f"{name}: {pct}%" for name, pct in zip(group_names, alloc)])


def regenerate_plots_from_saved():
    """Load saved results and regenerate plots without recomputing."""
    print("=" * 60)
    print("REGENERATING PLOTS FROM SAVED DATA")
    print("=" * 60)

    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Run full analysis first.")
        return

    with open(RESULTS_FILE, "r") as f:
        saved = json.load(f)

    total_budget = saved["total_budget"]
    h_data = saved["hierarchical"]
    a_data = saved["approx_max_group"]

    # Reconstruct the data structures needed for plotting
    h_all = [{"budget_pct": pct} for pct in h_data["all_budget_pcts"]]
    a_all = [{"budget_pct": pct} for pct in a_data["all_budget_pcts"]]

    h_best = {
        "budget_pct": h_data["best_budget_pct"],
        "cost": h_data["best_cost"],
        "allocation": tuple(h_data["best_allocation"]),
        "utility": h_data["best_utility"]
    } if h_data["best_budget_pct"] is not None else None

    h_worst = {
        "budget_pct": h_data["worst_budget_pct"],
        "cost": h_data["worst_cost"],
        "allocation": tuple(h_data["worst_allocation"]),
        "utility": h_data["worst_utility"]
    } if h_data["worst_budget_pct"] is not None else None

    a_best = {
        "budget_pct": a_data["best_budget_pct"],
        "cost": a_data["best_cost"],
        "allocation": tuple(a_data["best_allocation"]),
        "utility": a_data["best_utility"]
    } if a_data["best_budget_pct"] is not None else None

    a_worst = {
        "budget_pct": a_data["worst_budget_pct"],
        "cost": a_data["worst_cost"],
        "allocation": tuple(a_data["worst_allocation"]),
        "utility": a_data["worst_utility"]
    } if a_data["worst_budget_pct"] is not None else None

    region_names = h_data["group_names"]
    category_names = a_data["group_names"]

    print(f"\nTotal budget: {total_budget:,}")
    print(
        f"Hierarchical best: {h_best['budget_pct']:.1f}%" if h_best else "No data")
    print(
        f"Approx Max Group best: {a_best['budget_pct']:.1f}%" if a_best else "No data")

    print("\nCreating plots...")
    create_comparison_plots(h_all, a_all, h_best, h_worst, a_best, a_worst,
                            region_names, category_names, total_budget)

    print("\nPlots regenerated successfully!")


def main():
    print("=" * 60)
    print("BUDGET ALLOCATION ANALYSIS")
    print("(Measuring by Budget Usage - Higher is Better)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    with open("data/region_assignments.json", "r") as f:
        region_data = json.load(f)

    with open("data/poland_warszawa_2026_marysin-wawerski-anin.json", "r") as f:
        pb_data = json.load(f)

    # Define groups
    region_names = ["Syrenka", "Marysin", "Anin", "SP218"]
    category_names = list(pb_data["groups"].keys())

    print(f"\nGeographic regions ({len(region_names)}): {region_names}")
    print(f"Project categories ({len(category_names)}): {category_names}")

    # Generate allocations with 20% increments
    region_allocations = generate_percentage_allocations(
        len(region_names), increment=20)
    category_allocations = generate_percentage_allocations(
        len(category_names), increment=20)

    print(f"\nNumber of region allocations to test: {len(region_allocations)}")
    print(
        f"Number of category allocations to test: {len(category_allocations)}")

    # Use the original total budget
    total_budget = 725604
    print(f"\nTotal budget: {total_budget:,}")

    # Run hierarchical analysis
    print("\n" + "=" * 60)
    print("RUNNING HIERARCHICAL PB (Geographic Regions)")
    print("=" * 60)
    h_results = run_hierarchical_analysis(region_data, pb_data, total_budget,
                                          region_allocations, region_names)
    h_best, h_worst, h_all = analyze_results(h_results, region_names)

    print("\n--- Hierarchical PB Results ---")
    if h_best:
        print(
            f"BEST budget usage: {h_best['budget_pct']:.1f}% ({h_best['cost']:,} of {total_budget:,})")
        print(
            f"  Allocation: {format_allocation(h_best['allocation'], region_names)}")
        print(f"  Utility: {h_best['utility']}")
        print(
            f"\nWORST budget usage: {h_worst['budget_pct']:.1f}% ({h_worst['cost']:,} of {total_budget:,})")
        print(
            f"  Allocation: {format_allocation(h_worst['allocation'], region_names)}")
        print(f"  Utility: {h_worst['utility']}")

    # Run approx max group analysis
    print("\n" + "=" * 60)
    print("RUNNING APPROX MAX GROUP PB (Project Categories)")
    print("=" * 60)
    a_results = run_approx_max_group_analysis(pb_data, total_budget,
                                              category_allocations, category_names)
    a_best, a_worst, a_all = analyze_results(a_results, category_names)

    print("\n--- Approx Max Group PB Results ---")
    if a_best:
        print(
            f"BEST budget usage: {a_best['budget_pct']:.1f}% ({a_best['cost']:,} of {total_budget:,})")
        print(
            f"  Allocation: {format_allocation(a_best['allocation'], category_names)}")
        print(f"  Utility: {a_best['utility']}")
        print(
            f"\nWORST budget usage: {a_worst['budget_pct']:.1f}% ({a_worst['cost']:,} of {total_budget:,})")
        print(
            f"  Allocation: {format_allocation(a_worst['allocation'], category_names)}")
        print(f"  Utility: {a_worst['utility']}")

    # Create plots
    print("\n" + "=" * 60)
    print("CREATING PLOTS")
    print("=" * 60)
    create_comparison_plots(h_all, a_all, h_best, h_worst, a_best, a_worst,
                            region_names, category_names, total_budget)

    # Save results to JSON
    save_results = {
        "total_budget": total_budget,
        "hierarchical": {
            "best_budget_pct": h_best["budget_pct"] if h_best else None,
            "best_cost": h_best["cost"] if h_best else None,
            "best_allocation": list(h_best["allocation"]) if h_best else None,
            "best_utility": h_best["utility"] if h_best else None,
            "worst_budget_pct": h_worst["budget_pct"] if h_worst else None,
            "worst_cost": h_worst["cost"] if h_worst else None,
            "worst_allocation": list(h_worst["allocation"]) if h_worst else None,
            "worst_utility": h_worst["utility"] if h_worst else None,
            "group_names": region_names,
            "all_budget_pcts": [r["budget_pct"] for r in h_all]
        },
        "approx_max_group": {
            "best_budget_pct": a_best["budget_pct"] if a_best else None,
            "best_cost": a_best["cost"] if a_best else None,
            "best_allocation": list(a_best["allocation"]) if a_best else None,
            "best_utility": a_best["utility"] if a_best else None,
            "worst_budget_pct": a_worst["budget_pct"] if a_worst else None,
            "worst_cost": a_worst["cost"] if a_worst else None,
            "worst_allocation": list(a_worst["allocation"]) if a_worst else None,
            "worst_utility": a_worst["utility"] if a_worst else None,
            "group_names": category_names,
            "all_budget_pcts": [r["budget_pct"] for r in a_all]
        }
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    print("Plots saved to data/graph_budget_allocation_*.png")


def create_comparison_plots(h_all, a_all, h_best, h_worst, a_best, a_worst,
                            region_names, category_names, total_budget):
    """Create visualization plots comparing best and worst allocations by budget usage."""

    # Set up the style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11

    # Color palette - white background
    colors = {
        'hierarchical': '#E65100',  # Deep orange
        'approx': '#1565C0',  # Deep blue
        'best': '#2E7D32',  # Dark green
        'worst': '#C62828',  # Dark red
        'bg': 'white',
        'text': '#212121',  # Dark text
        'grid': '#BDBDBD'  # Light gray grid
    }

    # Plot 1: Distribution of budget usage for both methods (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), facecolor=colors['bg'])

    # Hierarchical budget usage distribution
    ax1 = axes[0]
    ax1.set_facecolor(colors['bg'])
    h_budget_pcts = [r["budget_pct"] for r in h_all]
    ax1.hist(h_budget_pcts, bins=20, color=colors['hierarchical'], alpha=0.8,
             edgecolor='#333333', linewidth=1)
    if h_best:
        ax1.axvline(h_best["budget_pct"], color=colors['best'], linestyle='--',
                    linewidth=2.5, label=f'Best: {h_best["budget_pct"]:.1f}%')
        ax1.axvline(h_worst["budget_pct"], color=colors['worst'], linestyle='--',
                    linewidth=2.5, label=f'Worst: {h_worst["budget_pct"]:.1f}%')
    ax1.set_xlabel('Budget Usage (%)', fontsize=14,
                   color=colors['text'], fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=14,
                   color=colors['text'], fontweight='bold')
    ax1.set_title('Hierarchical PB (Geographic Regions)',
                  fontsize=16, color=colors['text'], fontweight='bold')
    ax1.legend(fontsize=12, framealpha=0.9)
    ax1.tick_params(colors=colors['text'], labelsize=12)
    for spine in ax1.spines.values():
        spine.set_color(colors['text'])
    ax1.grid(True, alpha=0.4, color=colors['grid'])

    # Approx Max Group budget usage distribution
    ax2 = axes[1]
    ax2.set_facecolor(colors['bg'])
    a_budget_pcts = [r["budget_pct"] for r in a_all]
    ax2.hist(a_budget_pcts, bins=20, color=colors['approx'], alpha=0.8,
             edgecolor='#333333', linewidth=1)
    if a_best:
        ax2.axvline(a_best["budget_pct"], color=colors['best'], linestyle='--',
                    linewidth=2.5, label=f'Best: {a_best["budget_pct"]:.1f}%')
        ax2.axvline(a_worst["budget_pct"], color=colors['worst'], linestyle='--',
                    linewidth=2.5, label=f'Worst: {a_worst["budget_pct"]:.1f}%')
    ax2.set_xlabel('Budget Usage (%)', fontsize=14,
                   color=colors['text'], fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=14,
                   color=colors['text'], fontweight='bold')
    ax2.set_title('Approx Max Group PB (Project Categories)',
                  fontsize=16, color=colors['text'], fontweight='bold')
    ax2.legend(fontsize=12, framealpha=0.9)
    ax2.tick_params(colors=colors['text'], labelsize=12)
    for spine in ax2.spines.values():
        spine.set_color(colors['text'])
    ax2.grid(True, alpha=0.4, color=colors['grid'])

    plt.suptitle(f'Budget Usage Distribution Across All Allocations\nTotal Budget: {total_budget:,}',
                 fontsize=18, color=colors['text'], fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/graph_budget_allocation_distribution.png',
                dpi=150, facecolor=colors['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print("  Saved: graph_budget_allocation_distribution.png")

    # Plot 2: Best and Worst Allocation Breakdown (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor=colors['bg'])

    region_colors = ['#E53935', '#43A047', '#1E88E5', '#FB8C00']
    cat_colors = plt.cm.tab10(np.linspace(0, 1, len(category_names)))

    # Hierarchical Best
    ax = axes[0]
    ax.set_facecolor(colors['bg'])
    if h_best:
        labels = [n for n, v in zip(
            region_names, h_best["allocation"]) if v > 0]
        values = [v for v in h_best["allocation"] if v > 0]
        pie_colors = [region_colors[i]
                      for i, v in enumerate(h_best["allocation"]) if v > 0]
        if values:
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct='%1.0f%%',
                colors=pie_colors,
                textprops={'color': colors['text'],
                           'fontsize': 16, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)
        ax.set_title(f'Hierarchical PB - BEST\nBudget Used: {h_best["budget_pct"]:.1f}%',
                     fontsize=18, color=colors['best'], fontweight='bold')

    # Hierarchical Worst
    ax = axes[1]
    ax.set_facecolor(colors['bg'])
    if h_worst:
        labels = [n for n, v in zip(
            region_names, h_worst["allocation"]) if v > 0]
        values = [v for v in h_worst["allocation"] if v > 0]
        pie_colors = [region_colors[i]
                      for i, v in enumerate(h_worst["allocation"]) if v > 0]
        if values:
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct='%1.0f%%',
                colors=pie_colors,
                textprops={'color': colors['text'],
                           'fontsize': 16, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)
        ax.set_title(f'Hierarchical PB - WORST\nBudget Used: {h_worst["budget_pct"]:.1f}%',
                     fontsize=18, color=colors['worst'], fontweight='bold')

    # Approx Best
    ax = axes[2]
    ax.set_facecolor(colors['bg'])
    if a_best:
        labels = [n[:10] + '..' if len(n) > 10 else n
                  for n, v in zip(category_names, a_best["allocation"]) if v > 0]
        values = [v for v in a_best["allocation"] if v > 0]
        pie_colors = [cat_colors[i]
                      for i, v in enumerate(a_best["allocation"]) if v > 0]
        if values:
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct='%1.0f%%',
                colors=pie_colors,
                textprops={'color': colors['text'],
                           'fontsize': 16, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)
        ax.set_title(f'Approx Max Group PB - BEST\nBudget Used: {a_best["budget_pct"]:.1f}%',
                     fontsize=18, color=colors['best'], fontweight='bold')

    # Approx Worst
    ax = axes[3]
    ax.set_facecolor(colors['bg'])
    if a_worst:
        labels = [n[:10] + '..' if len(n) > 10 else n
                  for n, v in zip(category_names, a_worst["allocation"]) if v > 0]
        values = [v for v in a_worst["allocation"] if v > 0]
        pie_colors = [cat_colors[i]
                      for i, v in enumerate(a_worst["allocation"]) if v > 0]
        if values:
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct='%1.0f%%',
                colors=pie_colors,
                textprops={'color': colors['text'],
                           'fontsize': 16, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)
        ax.set_title(f'Approx Max Group PB - WORST\nBudget Used: {a_worst["budget_pct"]:.1f}%',
                     fontsize=18, color=colors['worst'], fontweight='bold')

    plt.suptitle(f'Best vs Worst Budget Allocations (by % Used)',
                 fontsize=22, color=colors['text'], fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/graph_budget_allocation_pies.png',
                dpi=150, facecolor=colors['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print("  Saved: graph_budget_allocation_pies.png")

    # Plot 3: Comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=colors['bg'])
    ax.set_facecolor(colors['bg'])

    methods = ['Hierarchical PB\n(Geographic)',
               'Approx Max Group PB\n(Categories)']
    best_pcts = [h_best["budget_pct"] if h_best else 0,
                 a_best["budget_pct"] if a_best else 0]
    worst_pcts = [h_worst["budget_pct"] if h_worst else 0,
                  a_worst["budget_pct"] if a_worst else 0]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, best_pcts, width, label='Best Allocation',
                   color=colors['best'], edgecolor='#333333', linewidth=1)
    bars2 = ax.bar(x + width/2, worst_pcts, width, label='Worst Allocation',
                   color=colors['worst'], edgecolor='#333333', linewidth=1)

    # Add horizontal line for Method of Equal Shares
    # Equal shares: total_cost=662750, total_budget=725604 -> 91.34%
    equal_shares_cost = 662750
    equal_shares_pct = (equal_shares_cost / total_budget) * 100
    ax.axhline(y=equal_shares_pct, color='#7B1FA2', linestyle=':', linewidth=2.5)
    
    # Annotate the line directly (inside the graph, right-aligned)
    ax.annotate(f'Method of Equal Shares: {equal_shares_pct:.1f}%',
                xy=(0.98, equal_shares_pct), xycoords=('axes fraction', 'data'),
                xytext=(0, 5), textcoords='offset points',
                va='bottom', ha='right', fontsize=10, fontweight='bold',
                color='#7B1FA2',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#7B1FA2'))

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', color=colors['text'], fontweight='bold', fontsize=11)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', color=colors['text'], fontweight='bold', fontsize=11)

    ax.set_ylabel('Budget Usage (%)', fontsize=12,
                  color=colors['text'], fontweight='bold')
    ax.set_title(f'Best vs Worst Budget Usage by Method\nTotal Budget: {total_budget:,}',
                 fontsize=14, color=colors['text'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, color=colors['text'])
    ax.tick_params(colors=colors['text'])
    for spine in ax.spines.values():
        spine.set_color(colors['text'])
    ax.grid(True, alpha=0.4, color=colors['grid'], axis='y')
    ax.set_ylim(0, 110)  # Up to 110% to leave room for labels

    plt.tight_layout()
    plt.savefig('results/graph_budget_allocation_comparison.png',
                dpi=150, facecolor=colors['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print("  Saved: graph_budget_allocation_comparison.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plots":
        regenerate_plots_from_saved()
    else:
        main()
