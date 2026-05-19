#!/usr/bin/env python3
"""
Comprehensive Analysis of 2025_07_28_01_28_case_* Negotiation Logs

This script analyzes the specific negotiation logs from the 2025_07_28_01_28_case_* 
directories and provides detailed insights and statistics for academic research.
"""

import os
import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import math

# Set up plotting style for academic publication
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


def _scenario_name_from_config(base_dir: str):
    config_path = os.path.join(base_dir, "darp_nego", "runners", "config.py")
    if not os.path.exists(config_path):
        return None

    import ast
    with open(config_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                continue

    if values.get("USE_LATEST_SCENARIO", False):
        return None

    scenario_path = values.get("SCENARIO_JSON_PATH")
    if not scenario_path:
        return None
    return os.path.splitext(os.path.basename(scenario_path))[0]


def _valid_session_folders(log_root: str):
    folders = []
    for folder in glob.glob(os.path.join(log_root, "*")):
        if not os.path.isdir(folder):
            continue
        session = os.path.basename(folder)
        negotiation_json = os.path.join(folder, "negotiation", f"negotiation_{session}.json")
        if os.path.exists(negotiation_json):
            folders.append(folder)
    return folders


def _mean_ci95(series):
    values = pd.Series(series).dropna()
    if values.empty:
        return 0.0, 0.0, 0.0
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half_width = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def find_latest_dataset_logs(log_root_name: str = "logs"):
    """Find all log directories for the latest dataset based on data folder name.

    The latest dataset name is taken from the newest timestamp folder under
    scenario_generation/data (e.g., 2025_11_03_18_13), and then
    matching logs '<dataset>_case_*' are collected from logs/.

    Returns:
        tuple[str, list[str]]: (dataset_name, list of session folder paths)
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    log_root = os.path.abspath(os.path.join(base_dir, log_root_name))

    configured_dataset = _scenario_name_from_config(base_dir)
    if configured_dataset:
        session_folders = sorted(glob.glob(os.path.join(log_root, f"{configured_dataset}_case_*")))
        session_folders = [folder for folder in session_folders if os.path.isdir(folder)]
        if session_folders:
            print(f"Selected dataset from config: {configured_dataset}")
            print(f"Found {len(session_folders)} log directories for {configured_dataset}_case_*")
            return configured_dataset, session_folders

    all_sessions = _valid_session_folders(log_root)
    if not all_sessions:
        print(f"❌ No negotiation logs found under {log_root}/.")
        return "", []

    latest_session = max(all_sessions, key=os.path.getmtime)
    latest_name = os.path.basename(latest_session)
    dataset_name = latest_name.split("_case_")[0] if "_case_" in latest_name else latest_name
    session_folders = sorted(glob.glob(os.path.join(log_root, f"{dataset_name}_case_*")))
    print(f"Selected latest logged dataset: {dataset_name}")
    print(f"Found {len(session_folders)} log directories for {dataset_name}_case_*")
    return dataset_name, session_folders


def extract_negotiation_data(session_folders):
    """Extract comprehensive negotiation data from all log files."""
    stats_data = []
    detailed_rounds_data = []

    for session_folder in session_folders:
        session_name = os.path.basename(session_folder)
        json_file = os.path.join(session_folder, "negotiation", f"negotiation_{session_name}.json")

        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found for {session_name}")
            continue

        try:
            with open(json_file, 'r') as f:
                log_data = json.load(f)

            # Extract case number from session name
            case_match = re.search(r'case_(\d+)', session_name)
            case_number = int(case_match.group(1)) if case_match else 0

            # Extract prenegotiation data
            prenegotiation = log_data.get("prenegotiation", {})
            final_state = log_data.get("final_state", {})
            rounds = log_data.get("rounds", [])

            # Basic statistics
            participants = prenegotiation.get("participants", [])
            num_agents = len(participants)

            # Cost analysis
            initial_utilities = prenegotiation.get("initial_utilities", {})
            final_utilities = final_state.get("final_utilities", {})
            initial_total_cost = sum(initial_utilities.values())
            final_total_cost = sum(final_utilities.values())

            # NEW LOGIC: Calculate cost change and determine agreement based on cost change
            cost_change = final_total_cost - initial_total_cost
            cost_change_percentage = ((
                                                  final_total_cost - initial_total_cost) / initial_total_cost * 100) if initial_total_cost > 0 else 0

            # NEW LOGIC: Agreement is reached if there's any cost change (positive or negative)
            agreement_reached = abs(cost_change) > 0.01  # Small threshold to avoid floating point issues

            # Negotiation results
            total_rounds = final_state.get("total_rounds", 0)
            execution_time = final_state.get("execution_time", 0)

            # Utility changes
            utility_changes = final_state.get("utility_changes", {})
            avg_utility_change = final_state.get("avg_utility_change", 0)

            # Client information
            revealed_clients = prenegotiation.get("revealed_clients", {})
            total_revealed_clients = sum(len(clients) for clients in revealed_clients.values())

            # Check last round for full acceptance
            last_round_accepted = False
            if rounds:
                last_round = rounds[-1]  # Get the last round
                agent_responses = last_round.get("agent_responses", {})
                last_round_accepted = bool(
                    last_round.get("full_acceptance", all(agent_responses.values()) if agent_responses else False)
                )

            # Detailed round analysis
            accepted_rounds = 0
            partial_rounds = 0
            total_proposed_transfers = 0
            total_accepted_transfers = 0
            total_partial_swaps = 0

            for round_data in rounds:
                round_number = round_data.get("round_number", 0)
                proposed_transfers = round_data.get("proposed_transfers", [])
                agent_responses = round_data.get("agent_responses", {})

                total_proposed_transfers += len(proposed_transfers)

                round_accepted = bool(
                    round_data.get("full_acceptance", all(agent_responses.values()) if agent_responses else False)
                )
                partial_accepted = bool(round_data.get("partial_acceptance", False))
                applied_swaps = round_data.get("applied_swaps", [])
                swaps_applied = len(applied_swaps) if applied_swaps else int(round_data.get("num_swaps", 0) or 0)
                if round_accepted:
                    accepted_rounds += 1
                if partial_accepted:
                    partial_rounds += 1
                    total_partial_swaps += swaps_applied
                if round_accepted or partial_accepted:
                    total_accepted_transfers += swaps_applied if swaps_applied else len(proposed_transfers)

                # Store detailed round data
                detailed_rounds_data.append({
                    'case_number': case_number,
                    'session_name': session_name,
                    'round_number': round_number,
                    'num_agents': num_agents,
                    'proposed_transfers': len(proposed_transfers),
                    'is_accepted': round_accepted,  # Based on all agents responding true
                    'partial_acceptance': partial_accepted,
                    'swaps_applied': swaps_applied,
                    'positive_responses': sum(1 for response in agent_responses.values() if response),
                    'negative_responses': sum(1 for response in agent_responses.values() if not response),
                    'all_agents_accepted': round_accepted
                })

            # Calculate approval rates
            full_approval_rate = accepted_rounds / total_rounds if total_rounds > 0 else 0
            partial_approval_rate = partial_rounds / total_rounds if total_rounds > 0 else 0
            transfer_acceptance_rate = total_accepted_transfers / total_proposed_transfers if total_proposed_transfers > 0 else 0

            # Store comprehensive data
            stats_data.append({
                'case_number': case_number,
                'session_name': session_name,
                'num_agents': num_agents,
                'initial_total_cost': initial_total_cost,
                'final_total_cost': final_total_cost,
                'cost_change': cost_change,
                'cost_change_percentage': cost_change_percentage,
                'agreement_reached': agreement_reached,  # Based on cost change
                'last_round_accepted': last_round_accepted,  # Based on all agents responding true in last round
                'total_rounds': total_rounds,
                'execution_time': execution_time,
                'avg_utility_change': avg_utility_change,
                'total_revealed_clients': total_revealed_clients,
                'accepted_rounds': accepted_rounds,
                'partial_rounds': partial_rounds,
                'full_approval_rate': full_approval_rate,
                'partial_approval_rate': partial_approval_rate,
                'total_proposed_transfers': total_proposed_transfers,
                'total_accepted_transfers': total_accepted_transfers,
                'total_partial_swaps': total_partial_swaps,
                'transfer_acceptance_rate': transfer_acceptance_rate,
                'initial_utilities': initial_utilities,
                'final_utilities': final_utilities,
                'utility_changes': utility_changes,
                'revealed_clients': revealed_clients
            })

            print(f"Processed {session_name}: {num_agents} agents, "
                  f"Initial cost: {initial_total_cost:.1f}, Final cost: {final_total_cost:.1f}, "
                  f"Cost change: {cost_change:.1f} ({cost_change_percentage:+.1f}%), "
                  f"Agreement: {agreement_reached}, Last round accepted: {last_round_accepted}, Rounds: {total_rounds}")

        except Exception as e:
            print(f"Error processing {session_name}: {e}")
            continue

    return pd.DataFrame(stats_data), pd.DataFrame(detailed_rounds_data)


def calculate_summary_statistics(df):
    """Calculate comprehensive summary statistics."""
    if df.empty:
        return {}

    ci_metrics = {
        "initial_cost": _mean_ci95(df["initial_total_cost"]),
        "final_cost": _mean_ci95(df["final_total_cost"]),
        "cost_change": _mean_ci95(df["cost_change"]),
        "cost_change_percentage": _mean_ci95(df["cost_change_percentage"]),
        "rounds": _mean_ci95(df["total_rounds"]),
        "execution_time": _mean_ci95(df["execution_time"]),
        "utility_change": _mean_ci95(df["avg_utility_change"]),
        "full_approval_rate": _mean_ci95(df["full_approval_rate"] * 100),
        "partial_approval_rate": _mean_ci95(df["partial_approval_rate"] * 100),
        "transfer_acceptance_rate": _mean_ci95(df["transfer_acceptance_rate"] * 100),
    }

    summary_stats = {
        'total_cases': len(df),
        'cases_with_agreement': len(df[df['agreement_reached'] == True]),
        'agreement_rate': len(df[df['agreement_reached'] == True]) / len(df) * 100,
        'cases_with_last_round_accepted': len(df[df['last_round_accepted'] == True]),
        'last_round_acceptance_rate': len(df[df['last_round_accepted'] == True]) / len(df) * 100,
        'avg_agents_per_case': df['num_agents'].mean(),
        'avg_initial_cost': df['initial_total_cost'].mean(),
        'avg_final_cost': df['final_total_cost'].mean(),
        'avg_cost_change': df['cost_change'].mean(),
        'avg_cost_change_percentage': df['cost_change_percentage'].mean(),
        'avg_rounds': df['total_rounds'].mean(),
        'avg_execution_time': df['execution_time'].mean(),
        'avg_utility_change': df['avg_utility_change'].mean(),
        'avg_revealed_clients': df['total_revealed_clients'].mean(),
        'avg_full_approval_rate': df['full_approval_rate'].mean() * 100,
        'avg_partial_approval_rate': df['partial_approval_rate'].mean() * 100,
        'total_proposed_transfers': df['total_proposed_transfers'].sum(),
        'total_accepted_transfers': df['total_accepted_transfers'].sum(),
        'total_partial_swaps': df['total_partial_swaps'].sum(),
        'overall_transfer_acceptance_rate': (
                    df['total_accepted_transfers'].sum() / df['total_proposed_transfers'].sum() * 100) if df[
                                                                                                              'total_proposed_transfers'].sum() > 0 else 0
    }

    for metric_name, (mean, low, high) in ci_metrics.items():
        summary_stats[f"{metric_name}_mean"] = mean
        summary_stats[f"{metric_name}_ci95_low"] = low
        summary_stats[f"{metric_name}_ci95_high"] = high

    return summary_stats


def create_visualizations(df, rounds_df, output_dir="analysis", dataset_name=None):
    """Create comprehensive visualizations for academic publication."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. AGREEMENT SUCCESS RATE PIE CHART
    plt.figure(figsize=(10, 6))
    agreement_counts = df['agreement_reached'].value_counts()
    colors = ['#ff6b6b', '#51cf66'] if len(agreement_counts) == 2 else ['#ff6b6b', '#51cf66', '#868e96']
    plt.pie(agreement_counts.values,
            labels=['No Cost Change' if not x else 'Cost Change Detected' for x in agreement_counts.index],
            autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Negotiation Success Rate (Cost Change Detection)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agreement_success_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. LAST ROUND ACCEPTANCE RATE PIE CHART
    plt.figure(figsize=(10, 6))
    last_round_counts = df['last_round_accepted'].value_counts()
    colors = ['#ff6b6b', '#51cf66'] if len(last_round_counts) == 2 else ['#ff6b6b', '#51cf66', '#868e96']
    plt.pie(last_round_counts.values,
            labels=['Last Round Rejected' if not x else 'Last Round Accepted' for x in last_round_counts.index],
            autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Last Round Acceptance Rate (All Agents Responded True)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'last_round_acceptance_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. COST COMPARISON BAR CHART
    plt.figure(figsize=(15, 8))
    x = range(len(df))
    width = 0.35

    plt.bar([i - width / 2 for i in x], df['initial_total_cost'], width, label='Initial Cost',
            color='#339af0', alpha=0.8)
    plt.bar([i + width / 2 for i in x], df['final_total_cost'], width, label='Final Cost',
            color='#51cf66', alpha=0.8)

    plt.xlabel('Case Number')
    plt.ylabel('Total Cost')
    title_ds = f" ({dataset_name} Dataset)" if dataset_name else ""
    plt.title(f'Initial vs Final Total Costs by Case{title_ds}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xticks(x, df['case_number'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. COST CHANGE PERCENTAGE SCATTER PLOT
    plt.figure(figsize=(12, 8))
    colors = ['#51cf66' if x else '#ff6b6b' for x in df['agreement_reached']]
    plt.scatter(df['num_agents'], df['cost_change_percentage'], c=colors, alpha=0.7, s=100)
    plt.xlabel('Number of Agents')
    plt.ylabel('Cost Change Percentage (%)')
    plt.title(f'Cost Change vs Number of Agents{title_ds}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#51cf66', label='Cost Change Detected'),
                       Patch(facecolor='#ff6b6b', label='No Cost Change')]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_change_vs_agents.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. NEGOTIATION ROUNDS ANALYSIS
    plt.figure(figsize=(12, 8))
    plt.hist(df['total_rounds'], bins=range(0, max(df['total_rounds']) + 2, 1),
             alpha=0.7, color='#339af0', edgecolor='black')
    plt.xlabel('Number of Rounds')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Negotiation Rounds{title_ds}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'negotiation_rounds_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. EXECUTION TIME ANALYSIS
    plt.figure(figsize=(12, 8))
    plt.scatter(df['num_agents'], df['execution_time'],
                c=df['agreement_reached'].map({True: '#51cf66', False: '#ff6b6b'}),
                alpha=0.7, s=100)
    plt.xlabel('Number of Agents')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Execution Time vs Number of Agents{title_ds}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [Patch(facecolor='#51cf66', label='Cost Change Detected'),
                       Patch(facecolor='#ff6b6b', label='No Cost Change')]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_vs_agents.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. FULL APPROVAL RATES ANALYSIS
    plt.figure(figsize=(12, 8))
    x = range(len(df))
    width = 0.6

    plt.bar(x, df['full_approval_rate'] * 100, width, label='Full Approval Rate (All Agents True)',
            color='#51cf66', alpha=0.8)

    plt.xlabel('Case Number')
    plt.ylabel('Rate (%)')
    plt.title('Full Approval Rates by Case (All Agents Responded True)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xticks(x, df['case_number'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'full_approval_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7b. PARTIAL APPROVAL RATES ANALYSIS
    plt.figure(figsize=(12, 8))
    plt.bar(x, df['partial_approval_rate'] * 100, width, label='Partial Approval Rate',
            color='#ffd166', alpha=0.8)
    plt.xlabel('Case Number')
    plt.ylabel('Rate (%)')
    plt.title('Partial Approval Rates by Case', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xticks(x, df['case_number'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'partial_approval_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8. ROUND-BY-ROUND ANALYSIS
    if not rounds_df.empty:
        plt.figure(figsize=(15, 8))

        # Group by round number and calculate averages
        round_stats = rounds_df.groupby('round_number').agg({
            'proposed_transfers': 'mean',
            'is_accepted': 'mean',
            'partial_acceptance': 'mean',
            'swaps_applied': 'mean',
            'positive_responses': 'mean',
            'negative_responses': 'mean',
            'all_agents_accepted': 'mean'
        }).reset_index()

        x = round_stats['round_number']
        plt.plot(x, round_stats['proposed_transfers'], 'o-', label='Proposed Transfers', linewidth=2, markersize=6)
        plt.plot(x, round_stats['positive_responses'], 's-', label='Positive Responses', linewidth=2, markersize=6)
        plt.plot(x, round_stats['negative_responses'], '^-', label='Negative Responses', linewidth=2, markersize=6)
        plt.plot(x, round_stats['swaps_applied'], 'x-', label='Applied Swaps', linewidth=2, markersize=6)
        plt.plot(x, round_stats['all_agents_accepted'] * max(round_stats['proposed_transfers']), 'd-',
                 label='All Agents Accepted', linewidth=2, markersize=6)

        plt.xlabel('Round Number')
        plt.ylabel('Average Count')
        plt.title(f'Round-by-Round Negotiation Dynamics{title_ds}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'round_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 9. SUMMARY STATISTICS TABLE
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    summary_stats = calculate_summary_statistics(df)

    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Total Cases', summary_stats['total_cases']],
        ['Cases with Cost Change', f"{summary_stats['cases_with_agreement']} ({summary_stats['agreement_rate']:.1f}%)"],
        ['Cases with Last Round Accepted',
         f"{summary_stats['cases_with_last_round_accepted']} ({summary_stats['last_round_acceptance_rate']:.1f}%)"],
        ['Average Agents per Case', f"{summary_stats['avg_agents_per_case']:.1f}"],
        ['Average Initial Cost', f"{summary_stats['avg_initial_cost']:.1f} [{summary_stats['initial_cost_ci95_low']:.1f}, {summary_stats['initial_cost_ci95_high']:.1f}]"],
        ['Average Final Cost', f"{summary_stats['avg_final_cost']:.1f} [{summary_stats['final_cost_ci95_low']:.1f}, {summary_stats['final_cost_ci95_high']:.1f}]"],
        ['Average Cost Change',
         f"{summary_stats['avg_cost_change']:.1f} [{summary_stats['cost_change_ci95_low']:.1f}, {summary_stats['cost_change_ci95_high']:.1f}] ({summary_stats['avg_cost_change_percentage']:.1f}%)"],
        ['Average Rounds', f"{summary_stats['avg_rounds']:.1f} [{summary_stats['rounds_ci95_low']:.1f}, {summary_stats['rounds_ci95_high']:.1f}]"],
        ['Average Execution Time', f"{summary_stats['avg_execution_time']:.1f} [{summary_stats['execution_time_ci95_low']:.1f}, {summary_stats['execution_time_ci95_high']:.1f}] seconds"],
        ['Average Utility Change', f"{summary_stats['avg_utility_change']:.1f} [{summary_stats['utility_change_ci95_low']:.1f}, {summary_stats['utility_change_ci95_high']:.1f}]"],
        ['Average Revealed Clients', f"{summary_stats['avg_revealed_clients']:.1f}"],
        ['Average Full Approval Rate', f"{summary_stats['avg_full_approval_rate']:.1f}% [{summary_stats['full_approval_rate_ci95_low']:.1f}, {summary_stats['full_approval_rate_ci95_high']:.1f}]"],
        ['Average Partial Approval Rate', f"{summary_stats['avg_partial_approval_rate']:.1f}% [{summary_stats['partial_approval_rate_ci95_low']:.1f}, {summary_stats['partial_approval_rate_ci95_high']:.1f}]"],
        ['Total Proposed Transfers', summary_stats['total_proposed_transfers']],
        ['Total Accepted Transfers', summary_stats['total_accepted_transfers']],
        ['Total Partial Swaps', summary_stats['total_partial_swaps']],
        ['Overall Transfer Acceptance Rate', f"{summary_stats['overall_transfer_acceptance_rate']:.1f}%"]
    ]

    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center', colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(summary_data)):
        for j in range(2):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#2E86AB')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                table[(i, j)].set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')

    plt.title(f'{dataset_name} Dataset Analysis Summary' if dataset_name else 'Dataset Analysis Summary', fontsize=16,
              fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_detailed_analysis(df, rounds_df, dataset_name=None):
    """Print detailed analysis results."""
    print("\n" + "=" * 80)
    header_ds = f" - {dataset_name} DATASET" if dataset_name else ""
    print(f"DETAILED ANALYSIS RESULTS{header_ds}")
    print("=" * 80)

    summary_stats = calculate_summary_statistics(df)

    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total cases analyzed: {summary_stats['total_cases']}")
    print(
        f"   Cases with cost change: {summary_stats['cases_with_agreement']} ({summary_stats['agreement_rate']:.1f}%)")
    print(
        f"   Cases with last round accepted: {summary_stats['cases_with_last_round_accepted']} ({summary_stats['last_round_acceptance_rate']:.1f}%)")
    print(f"   Average agents per case: {summary_stats['avg_agents_per_case']:.1f}")

    print(f"\n💰 COST ANALYSIS:")
    print(f"   Average initial cost: {summary_stats['avg_initial_cost']:.1f} "
          f"[95% CI: {summary_stats['initial_cost_ci95_low']:.1f}, {summary_stats['initial_cost_ci95_high']:.1f}]")
    print(f"   Average final cost: {summary_stats['avg_final_cost']:.1f} "
          f"[95% CI: {summary_stats['final_cost_ci95_low']:.1f}, {summary_stats['final_cost_ci95_high']:.1f}]")
    print(
        f"   Average cost change: {summary_stats['avg_cost_change']:.1f} "
        f"[95% CI: {summary_stats['cost_change_ci95_low']:.1f}, {summary_stats['cost_change_ci95_high']:.1f}] "
        f"({summary_stats['avg_cost_change_percentage']:.1f}%)")

    print(f"\n⏱️  PERFORMANCE METRICS:")
    print(f"   Average rounds: {summary_stats['avg_rounds']:.1f} "
          f"[95% CI: {summary_stats['rounds_ci95_low']:.1f}, {summary_stats['rounds_ci95_high']:.1f}]")
    print(f"   Average execution time: {summary_stats['avg_execution_time']:.1f} "
          f"[95% CI: {summary_stats['execution_time_ci95_low']:.1f}, {summary_stats['execution_time_ci95_high']:.1f}] seconds")
    print(f"   Average utility change: {summary_stats['avg_utility_change']:.1f} "
          f"[95% CI: {summary_stats['utility_change_ci95_low']:.1f}, {summary_stats['utility_change_ci95_high']:.1f}]")

    print(f"\n🤝 NEGOTIATION DYNAMICS:")
    print(f"   Average full approval rate: {summary_stats['avg_full_approval_rate']:.1f}% "
          f"[95% CI: {summary_stats['full_approval_rate_ci95_low']:.1f}, {summary_stats['full_approval_rate_ci95_high']:.1f}]")
    print(f"   Average partial approval rate: {summary_stats['avg_partial_approval_rate']:.1f}% "
          f"[95% CI: {summary_stats['partial_approval_rate_ci95_low']:.1f}, {summary_stats['partial_approval_rate_ci95_high']:.1f}]")
    print(f"   Total proposed transfers: {summary_stats['total_proposed_transfers']}")
    print(f"   Total accepted transfers: {summary_stats['total_accepted_transfers']}")
    print(f"   Total partial swaps: {summary_stats['total_partial_swaps']}")
    print(f"   Overall transfer acceptance rate: {summary_stats['overall_transfer_acceptance_rate']:.1f}%")

    print(f"\n CASE-BY-CASE BREAKDOWN:")
    for _, row in df.iterrows():
        cost_status = "✅ COST CHANGE" if row['agreement_reached'] else "❌ NO COST CHANGE"
        last_round_status = "✅ LAST ROUND ACCEPTED" if row['last_round_accepted'] else "❌ LAST ROUND REJECTED"
        print(f"   Case {row['case_number']:2d}: {row['num_agents']} agents, "
              f"Cost: {row['initial_total_cost']:6.1f} → {row['final_total_cost']:6.1f} "
              f"({row['cost_change_percentage']:+6.1f}%), "
              f"Rounds: {row['total_rounds']:2d}, {cost_status}, {last_round_status}")

    if not rounds_df.empty:
        print(f"\n ROUND ANALYSIS:")
        round_stats = rounds_df.groupby('round_number').agg({
            'proposed_transfers': 'mean',
            'is_accepted': 'mean',
            'partial_acceptance': 'mean',
            'swaps_applied': 'mean',
            'positive_responses': 'mean',
            'negative_responses': 'mean',
            'all_agents_accepted': 'mean'
        }).reset_index()

        for _, row in round_stats.iterrows():
            print(f"   Round {int(row['round_number']):2d}: "
                  f"{row['proposed_transfers']:.1f} transfers, "
                  f"{row['is_accepted'] * 100:.1f}% full acceptance (all agents true), "
                  f"{row['partial_acceptance'] * 100:.1f}% partial acceptance, "
                  f"{row['swaps_applied']:.1f} applied swaps, "
                  f"{row['positive_responses']:.1f} positive responses, "
                  f"{row['negative_responses']:.1f} negative responses")


def main():
    """Main analysis function."""
    print(" Analyzing latest *_case_* Negotiation Logs")
    print("=" * 80)

    try:
        log_roots = ["logs"]
        base_dir = os.path.abspath(os.path.dirname(__file__))

        for log_root_name in log_roots:
            print(f"\n=== Log root: {log_root_name} ===")
            # Find log directories for the latest dataset
            dataset_name, session_folders = find_latest_dataset_logs(log_root_name=log_root_name)

            if not session_folders:
                print("❌ No log directories found matching the pattern!")
                continue

            # Output directory under stats/<dataset_name>_<log_root>
            output_dir = os.path.join(base_dir, "stats", f"{(dataset_name or 'analysis')}_{log_root_name}")

            # Extract data
            print("\n📊 Extracting negotiation data...")
            df, rounds_df = extract_negotiation_data(session_folders)

            if df.empty:
                print("❌ No valid log data found!")
                continue

            # Create visualizations
            print("\n🎨 Creating visualizations...")
            create_visualizations(df, rounds_df, output_dir, dataset_name=dataset_name)

            # Print detailed analysis
            print_detailed_analysis(df, rounds_df, dataset_name=dataset_name)

            # Save data to CSV
            print("\n💾 Saving analysis data...")
            df.to_csv(f"{output_dir}/negotiation_analysis.csv", index=False)
            rounds_df.to_csv(f"{output_dir}/rounds_analysis.csv", index=False)

            # Save summary statistics
            summary_stats = calculate_summary_statistics(df)
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)

            print(f"\n✅ Analysis complete! Results saved to: {output_dir}/")
            print("\n📁 Generated files:")
            print("   - negotiation_analysis.csv (detailed case data)")
            print("   - rounds_analysis.csv (round-by-round data)")
            print("   - summary_statistics.csv (summary metrics)")
            print("   - agreement_success_rate.png")
            print("   - last_round_acceptance_rate.png")
            print("   - cost_comparison.png")
            print("   - cost_change_vs_agents.png")
            print("   - negotiation_rounds_distribution.png")
            print("   - execution_time_vs_agents.png")
            print("   - full_approval_rates.png")
            print("   - round_dynamics.png")
            print("   - summary_statistics.png")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
