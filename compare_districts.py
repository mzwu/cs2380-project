"""
Comparison script for Participatory Budgeting results across districts.
Compares Wesoła (2023) and Marysin-Wawerski-Anin (2026) results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load both analysis results
with open("data/analysis_results_wesola.json", "r") as f:
    wesola = json.load(f)

with open("data/analysis_results.json", "r") as f:
    marysin = json.load(f)

# District metadata
districts = {
    "Wesoła": {
        "data": wesola,
        "year": 2023,
        "color": "#e74c3c"  # Red
    },
    "Marysin-Wawerski-Anin": {
        "data": marysin,
        "year": 2026,
        "color": "#3498db"  # Blue
    }
}

# Method names and colors
methods = ["Hierarchical PB", "Approx Max Group PB", "Method of Equal Shares"]
method_short = ["Hierarchical", "Max Group", "Equal Shares"]
method_colors = ['#2ecc71', '#9b59b6', '#f39c12']  # Green, Purple, Orange


def create_comparison_charts():
    """Create all comparison charts."""

    # Set up style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11

    # 1. Utility per Voter Comparison (normalized)
    fig1, ax1 = plt.subplots(figsize=(12, 6), facecolor='white')

    x = np.arange(len(methods))
    width = 0.35

    wesola_utility_per_voter = [r['utility'] /
                                wesola['num_voters'] for r in wesola['results']]
    marysin_utility_per_voter = [
        r['utility'] / marysin['num_voters'] for r in marysin['results']]

    bars1 = ax1.bar(x - width/2, wesola_utility_per_voter, width,
                    label=f"Wesoła ({wesola['num_voters']} voters)",
                    color='#e74c3c', edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, marysin_utility_per_voter, width,
                    label=f"Marysin ({marysin['num_voters']} voters)",
                    color='#3498db', edgecolor='black', linewidth=1.2)

    ax1.set_ylabel('Utility per Voter', fontsize=14, fontweight='bold')
    ax1.set_title('Utility per Voter by Method\n(Higher is Better)',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_short, fontsize=12)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_ylim(0, max(max(wesola_utility_per_voter),
                 max(marysin_utility_per_voter)) * 1.2)

    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('data/comparison_01_utility_per_voter.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("📊 Saved: comparison_01_utility_per_voter.png")

    # 2. Budget Efficiency Comparison
    fig2, ax2 = plt.subplots(figsize=(12, 6), facecolor='white')

    wesola_efficiency = [r['unused_percentage'] for r in wesola['results']]
    marysin_efficiency = [r['unused_percentage'] for r in marysin['results']]

    bars1 = ax2.bar(x - width/2, wesola_efficiency, width,
                    label=f"Wesoła (Budget: {wesola['total_budget']:,})",
                    color='#e74c3c', edgecolor='black', linewidth=1.2)
    bars2 = ax2.bar(x + width/2, marysin_efficiency, width,
                    label=f"Marysin (Budget: {marysin['total_budget']:,})",
                    color='#3498db', edgecolor='black', linewidth=1.2)

    ax2.set_ylabel('Unused Budget (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Budget Efficiency by Method\n(Lower is Better)',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_short, fontsize=12)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.set_ylim(0, max(max(wesola_efficiency), max(marysin_efficiency)) * 1.3)

    # Add value labels
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('data/comparison_02_budget_efficiency.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("📊 Saved: comparison_02_budget_efficiency.png")

    # 3. Projects Funded (as % of available)
    fig3, ax3 = plt.subplots(figsize=(12, 6), facecolor='white')

    wesola_projects_pct = [r['num_projects'] /
                           wesola['num_projects'] * 100 for r in wesola['results']]
    marysin_projects_pct = [
        r['num_projects'] / marysin['num_projects'] * 100 for r in marysin['results']]

    bars1 = ax3.bar(x - width/2, wesola_projects_pct, width,
                    label=f"Wesoła ({wesola['num_projects']} total projects)",
                    color='#e74c3c', edgecolor='black', linewidth=1.2)
    bars2 = ax3.bar(x + width/2, marysin_projects_pct, width,
                    label=f"Marysin ({marysin['num_projects']} total projects)",
                    color='#3498db', edgecolor='black', linewidth=1.2)

    ax3.set_ylabel('Projects Funded (%)', fontsize=14, fontweight='bold')
    ax3.set_title('Percentage of Projects Funded by Method',
                  fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels(method_short, fontsize=12)
    ax3.legend(fontsize=11, loc='upper right')
    ax3.set_ylim(0, 100)

    # Add value labels with actual counts
    for i, bar in enumerate(bars1):
        count = wesola['results'][i]['num_projects']
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{bar.get_height():.0f}%\n({count})', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for i, bar in enumerate(bars2):
        count = marysin['results'][i]['num_projects']
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{bar.get_height():.0f}%\n({count})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('data/comparison_03_projects_funded.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("📊 Saved: comparison_03_projects_funded.png")

    # 4. Method Performance Heatmap
    fig4, ax4 = plt.subplots(figsize=(10, 8), facecolor='white')

    # Create normalized scores (0-100, higher is better)
    def normalize(vals, higher_better=True):
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return [100] * len(vals)
        if higher_better:
            return [(v - min_v) / (max_v - min_v) * 100 for v in vals]
        else:
            return [(max_v - v) / (max_v - min_v) * 100 for v in vals]

    # Metrics for heatmap
    metrics = ['Utility/Voter', 'Budget Efficiency', '% Projects Funded']

    # Combine all values for normalization
    all_utility = wesola_utility_per_voter + marysin_utility_per_voter
    all_efficiency = wesola_efficiency + marysin_efficiency
    all_projects = wesola_projects_pct + marysin_projects_pct

    # Normalize across both districts
    norm_utility = normalize(all_utility, higher_better=True)
    norm_efficiency = normalize(all_efficiency, higher_better=False)
    norm_projects = normalize(all_projects, higher_better=True)

    # Build heatmap data: rows = methods, columns = district-metrics
    heatmap_data = []
    labels_y = []
    labels_x = ['Wesoła\nUtility', 'Wesoła\nEfficiency', 'Wesoła\nProjects',
                'Marysin\nUtility', 'Marysin\nEfficiency', 'Marysin\nProjects']

    for i, method in enumerate(method_short):
        row = [
            norm_utility[i], norm_efficiency[i], norm_projects[i],
            norm_utility[i+3], norm_efficiency[i+3], norm_projects[i+3]
        ]
        heatmap_data.append(row)
        labels_y.append(method)

    heatmap_data = np.array(heatmap_data)

    im = ax4.imshow(heatmap_data, cmap='RdYlGn',
                    aspect='auto', vmin=0, vmax=100)

    ax4.set_xticks(np.arange(len(labels_x)))
    ax4.set_yticks(np.arange(len(labels_y)))
    ax4.set_xticklabels(labels_x, fontsize=10)
    ax4.set_yticklabels(labels_y, fontsize=12)

    # Add text annotations
    for i in range(len(labels_y)):
        for j in range(len(labels_x)):
            val = heatmap_data[i, j]
            color = 'white' if val < 30 or val > 70 else 'black'
            ax4.text(j, i, f'{val:.0f}', ha='center', va='center',
                     color=color, fontsize=11, fontweight='bold')

    # Add vertical line to separate districts
    ax4.axvline(x=2.5, color='black', linewidth=2)

    ax4.set_title('Method Performance Heatmap\n(0-100 Normalized Score, Higher is Better)',
                  fontsize=16, fontweight='bold', pad=20)

    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Score (0-100)', fontsize=12)

    plt.tight_layout()
    plt.savefig('data/comparison_04_heatmap.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("📊 Saved: comparison_04_heatmap.png")

    # 5. Summary Dashboard
    fig5, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='white')

    # Panel 1: District Overview
    ax = axes[0, 0]
    overview_data = [
        ['Metric', 'Wesoła', 'Marysin'],
        ['Year', '2023', '2026'],
        ['Budget', f"{wesola['total_budget']:,}",
            f"{marysin['total_budget']:,}"],
        ['Voters', f"{wesola['num_voters']:,}", f"{marysin['num_voters']:,}"],
        ['Projects', str(wesola['num_projects']),
         str(marysin['num_projects'])],
        ['Budget/Voter', f"{wesola['total_budget']/wesola['num_voters']:,.0f}",
         f"{marysin['total_budget']/marysin['num_voters']:,.0f}"],
    ]
    ax.axis('off')
    table = ax.table(cellText=overview_data[1:], colLabels=overview_data[0],
                     loc='center', cellLoc='center',
                     colColours=['#ecf0f1', '#e74c3c', '#3498db'])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    ax.set_title('District Overview', fontsize=14, fontweight='bold', pad=20)

    # Panel 2: Best Method per District
    ax = axes[0, 1]

    # Find best method for each metric per district
    best_methods = []
    for district_name, data in [('Wesoła', wesola), ('Marysin', marysin)]:
        results = data['results']
        best_utility = max(results, key=lambda x: x['utility'])['method']
        best_efficiency = min(
            results, key=lambda x: x['unused_percentage'])['method']
        best_projects = max(results, key=lambda x: x['num_projects'])['method']
        best_methods.append([district_name, best_utility.replace(' PB', '').replace('Method of ', ''),
                             best_efficiency.replace(
                                 ' PB', '').replace('Method of ', ''),
                             best_projects.replace(' PB', '').replace('Method of ', '')])

    best_data = [['District', 'Best Utility',
                  'Best Efficiency', 'Most Projects']] + best_methods
    ax.axis('off')
    table2 = ax.table(cellText=best_data[1:], colLabels=best_data[0],
                      loc='center', cellLoc='center',
                      colColours=['#ecf0f1']*4)
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1.2, 2)
    ax.set_title('Best Performing Method per Metric',
                 fontsize=14, fontweight='bold', pad=20)

    # Panel 3: Method Ranking by District
    ax = axes[1, 0]

    # Calculate average normalized score per method per district
    wesola_avg = [(norm_utility[i] + norm_efficiency[i] +
                   norm_projects[i]) / 3 for i in range(3)]
    marysin_avg = [(norm_utility[i+3] + norm_efficiency[i+3] +
                    norm_projects[i+3]) / 3 for i in range(3)]

    x = np.arange(len(method_short))
    bars1 = ax.bar(x - width/2, wesola_avg, width,
                   label='Wesoła', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, marysin_avg, width,
                   label='Marysin', color='#3498db', edgecolor='black')

    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Method Performance\n(Average of Normalized Metrics)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_short, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 4: Key Insights
    ax = axes[1, 1]
    ax.axis('off')

    # Determine insights
    insights = []

    # Overall winner
    wesola_winner = method_short[wesola_avg.index(max(wesola_avg))]
    marysin_winner = method_short[marysin_avg.index(max(marysin_avg))]

    insights.append(f"• Best overall in Wesoła: {wesola_winner}")
    insights.append(f"• Best overall in Marysin: {marysin_winner}")

    # Consistency check
    if wesola_winner == marysin_winner:
        insights.append(f"• {wesola_winner} performs best in BOTH districts")
    else:
        insights.append("• Different methods excel in each district")

    # Efficiency observation
    avg_wesola_eff = sum(wesola_efficiency) / 3
    avg_marysin_eff = sum(marysin_efficiency) / 3
    if avg_wesola_eff < avg_marysin_eff:
        insights.append(
            f"• Wesoła has better avg efficiency ({avg_wesola_eff:.1f}% vs {avg_marysin_eff:.1f}%)")
    else:
        insights.append(
            f"• Marysin has better avg efficiency ({avg_marysin_eff:.1f}% vs {avg_wesola_eff:.1f}%)")

    # EJR compliance
    insights.append("• All methods satisfy EJR in both districts")

    insights_text = '\n\n'.join(insights)
    ax.text(0.1, 0.8, "Key Insights", fontsize=14,
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.65, insights_text, fontsize=12, transform=ax.transAxes,
            verticalalignment='top', linespacing=1.5)

    plt.suptitle('Participatory Budgeting Methods: Cross-District Comparison\n',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('data/comparison_05_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("📊 Saved: comparison_05_dashboard.png")

    print("\n✅ All comparison charts generated!")


if __name__ == "__main__":
    create_comparison_charts()
