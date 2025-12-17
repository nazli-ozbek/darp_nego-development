import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

def analyze_agent_costs(log_dirs):
    """
    Analyze agent costs from routing logs.
    
    Args:
        log_dirs (list): List of log directory paths
    
    Returns:
        pd.DataFrame: DataFrame with cost statistics for each agent count
    """
    # Data structure to store results
    results = []
    
    for log_dir in log_dirs:
        # Extract directory name and case number
        dir_name = os.path.basename(log_dir)
        case_match = re.search(r'case_(\d+)', dir_name)
        case_number = int(case_match.group(1)) if case_match else 0
        
        # Path to routing JSON file
        routing_file = os.path.join(log_dir, "routing", f"routing_{dir_name}.json")
        
        if not os.path.exists(routing_file):
            print(f"Warning: No routing file found for {dir_name}")
            continue
        
        try:
            with open(routing_file, 'r') as f:
                routing_data = json.load(f)
                
            # Derive agent_count from routing data
            agent_count = len(routing_data.get("initial_routes", {}))
            if agent_count == 0:
                print(f"Warning: initial_routes empty for {dir_name}")
                continue
                
            # Extract initial and final costs
            initial_costs = {}
            final_costs = {}
            
            # Get initial costs for each agent
            for agent_id, agent_data in routing_data.get("initial_routes", {}).items():
                initial_costs[agent_id] = agent_data.get("cost", 0)
            
            # Get final costs for each agent
            for agent_id, agent_data in routing_data.get("final_routes", {}).items():
                final_costs[agent_id] = agent_data.get("cost", 0)
            
            # Calculate total costs
            total_initial_cost = sum(initial_costs.values())
            total_final_cost = sum(final_costs.values())
            
            # Cost reduction percentage
            cost_reduction = 100 * (total_initial_cost - total_final_cost) / total_initial_cost if total_initial_cost > 0 else 0
            
            # Store results
            results.append({
                "agent_count": agent_count,
                "case_number": case_number,
                "directory": dir_name,
                "initial_total_cost": total_initial_cost,
                "final_total_cost": total_final_cost,
                "cost_reduction": cost_reduction,
                "cost_reduction_absolute": total_initial_cost - total_final_cost
            })
            
        except Exception as e:
            print(f"Error processing {routing_file}: {str(e)}")
    
    # Convert to DataFrame
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    
    return df

def generate_statistics(df):
    """
    Generate statistics grouped by agent count
    
    Args:
        df (pd.DataFrame): DataFrame with cost analysis results
    
    Returns:
        pd.DataFrame: DataFrame with statistics for each agent count
    """
    if df.empty:
        return pd.DataFrame()
    
    # Group by agent_count and calculate statistics
    stats = df.groupby('agent_count').agg({
        'initial_total_cost': ['mean', 'std', 'min', 'max', 'count'],
        'final_total_cost': ['mean', 'std', 'min', 'max'],
        'cost_reduction': ['mean', 'std', 'min', 'max'],
        'cost_reduction_absolute': ['mean', 'std', 'min', 'max']
    })
    
    # Flatten the column hierarchy
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    
    return stats

def generate_plots(df, stats, output_dir="stats"):
    """
    Generate plots for the cost analysis
    
    Args:
        df (pd.DataFrame): DataFrame with cost analysis results
        stats (pd.DataFrame): DataFrame with statistics
        output_dir (str): Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        print("No data to plot")
        return
    
    # Plot 1: Bar chart of average costs for each agent count
    plt.figure(figsize=(10, 6))
    
    # Get agent counts and sort them
    agent_counts = sorted(stats.index.tolist())
    
    # Bar positions
    x = np.arange(len(agent_counts))
    width = 0.35
    
    # Plot bars
    initial_bars = plt.bar(x - width/2, 
                          stats['initial_total_cost_mean'], 
                          width, 
                          yerr=stats['initial_total_cost_std'],
                          label='No Negotiation (Initial)')
    
    final_bars = plt.bar(x + width/2, 
                        stats['final_total_cost_mean'], 
                        width, 
                        yerr=stats['final_total_cost_std'],
                        label='With Negotiation (Final)')
    
    # Add labels and title
    plt.xlabel('Number of Agents')
    plt.ylabel('Total System Cost')
    plt.title('Average Total System Cost by Number of Agents')
    plt.xticks(x, agent_counts)
    plt.legend()
    
    # Add value labels on bars
    for bar in initial_bars + final_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_costs_by_agent_count.png'))
    
    # Plot 2: Cost reduction percentage
    plt.figure(figsize=(10, 6))
    
    # Plot bars for cost reduction percentage
    bars = plt.bar(agent_counts, 
                  stats['cost_reduction_mean'], 
                  yerr=stats['cost_reduction_std'])
    
    # Add labels and title
    plt.xlabel('Number of Agents')
    plt.ylabel('Cost Reduction (%)')
    plt.title('Average Cost Reduction through Negotiation by Number of Agents')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_reduction_percentage.png'))
    
    # Plot 3: Box plot of final costs by agent count
    plt.figure(figsize=(10, 6))
    
    # Create box plot
    df.boxplot(column='final_total_cost', by='agent_count', figsize=(10, 6))
    
    # Add labels and title
    plt.xlabel('Number of Agents')
    plt.ylabel('Total System Cost')
    plt.title('Distribution of Total System Costs with Negotiation')
    plt.suptitle('')  # Remove the default pandas boxplot title
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_distribution_boxplot.png'))
    
    # Plot 4: Scatter plot showing relationship between agent count and cost reduction
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with jittering to see overlapping points
    for count in agent_counts:
        subset = df[df['agent_count'] == count]
        # Add small random noise for better visualization
        x_jittered = np.random.normal(count, 0.05, size=len(subset))
        plt.scatter(x_jittered, subset['cost_reduction'], alpha=0.6, label=f'{count} agents')
    
    # Add labels and title
    plt.xlabel('Number of Agents')
    plt.ylabel('Cost Reduction (%)')
    plt.title('Cost Reduction through Negotiation for Each Scenario')
    
    # Add a line showing the mean for each agent count
    for count in agent_counts:
        mean_reduction = df[df['agent_count'] == count]['cost_reduction'].mean()
        plt.hlines(mean_reduction, count-0.2, count+0.2, colors='red', linestyles='dashed')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_reduction_scatter.png'))
    
    print(f"Plots saved to {output_dir} directory")

def save_results(df, stats, output_dir="stats"):
    """
    Save analysis results to CSV files with improved column names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a copy of dataframes to rename columns
    df_formatted = df.copy()
    stats_formatted = stats.copy()
    
    # Rename columns with proper capitalization for full results
    column_mapping = {
        'agent_count': 'Agent Count',
        'case_number': 'Case Number',
        'directory': 'Directory',
        'initial_total_cost': 'Initial Total Cost',
        'final_total_cost': 'Final Total Cost',
        'cost_reduction': 'Cost Reduction (%)',
        'cost_reduction_absolute': 'Absolute Cost Reduction'
    }
    df_formatted.rename(columns=column_mapping, inplace=True)
    
    # Rename stats columns - need to handle multi-level columns
    new_stats_columns = []
    for col in stats.columns:
        parts = col.split('_')
        if len(parts) > 1:
            metric = '_'.join(parts[:-1])
            stat = parts[-1]
            
            # Convert to title case and handle special cases
            metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
            stat_name = stat.capitalize()
            
            new_stats_columns.append(f"{metric_name} ({stat_name})")
        else:
            new_stats_columns.append(' '.join(word.capitalize() for word in col.split('_')))
    
    stats_formatted.columns = new_stats_columns
    
    # Save to CSV with 2 decimal places
    df_formatted.to_csv(os.path.join(output_dir, "agent_costs_full.csv"), 
                        float_format="%.2f", index=False)
    stats_formatted.to_csv(os.path.join(output_dir, "agent_costs_stats.csv"), 
                          float_format="%.2f")
    
    print(f"Results saved to {output_dir} directory")

def analyze_negotiation_results(log_dirs):
    """
    Analyze both routing and negotiation results with additional metrics
    
    Args:
        log_dirs (list): List of log directory paths
    
    Returns:
        pd.DataFrame: DataFrame with negotiation results
    """
    results = []
    
    for log_dir in log_dirs:
        dir_name = os.path.basename(log_dir)
        case_match = re.search(r'case_(\d+)', dir_name)
        case_number = int(case_match.group(1)) if case_match else 0
        
        routing_file = os.path.join(log_dir, "routing", f"routing_{dir_name}.json")
        negotiation_file = os.path.join(log_dir, "negotiation", f"negotiation_{dir_name}.json")
        
        if not os.path.exists(routing_file) or not os.path.exists(negotiation_file):
            continue
        
        try:
            # Load routing data
            with open(routing_file, 'r') as f:
                routing_data = json.load(f)
                
            # Load negotiation data
            with open(negotiation_file, 'r') as f:
                negotiation_data = json.load(f)
            
            # Derive agent_count from routing data
            agent_count = len(routing_data.get("initial_routes", {}))
            if agent_count == 0:
                continue
            
            # Extract routing costs
            initial_costs = {}
            final_costs = {}
            
            for agent_id, agent_data in routing_data.get("initial_routes", {}).items():
                initial_costs[agent_id] = agent_data.get("cost", 0)
            
            for agent_id, agent_data in routing_data.get("final_routes", {}).items():
                final_costs[agent_id] = agent_data.get("cost", 0)
            
            total_initial_cost = sum(initial_costs.values())
            total_final_cost = sum(final_costs.values())
            
            # Extract negotiation data
            rounds = negotiation_data.get("rounds", [])
            total_rounds = len(rounds)
            
            # Calculate exchange approval rate
            approved_rounds = sum(1 for round_data in rounds if round_data.get("is_accepted", False))
            exchange_approval_rate = approved_rounds / total_rounds if total_rounds > 0 else 0
            
            # No partial acceptances in this protocol
            partial_approval_rate = 0.0
            
            # Count all proposed transfers in accepted rounds (only full acceptances)
            total_proposed_swaps = sum(len(round_data.get("proposed_transfers", [])) 
                               for round_data in rounds 
                               if round_data.get("is_accepted", False))
            
            # Count only transfers that were actually accepted (only full acceptances)
            accepted_swaps = 0
            for round_data in rounds:
                if round_data.get("is_accepted", False):
                    # For fully accepted rounds, all proposed transfers were accepted
                    accepted_swaps += len(round_data.get("proposed_transfers", []))
            
            # Get utility changes - correct to positive
            final_state = negotiation_data.get("final_state", {})
            utility_changes = final_state.get("utility_changes", {})
            utility_changes = {k: abs(v) for k, v in utility_changes.items()}
            
            avg_utility_gain = sum(utility_changes.values()) / len(utility_changes) if utility_changes else 0
            
            # Check if this is a Pareto improvement
            # A Pareto improvement means no agent is worse off and at least one is better off
            is_pareto_improvement = all(abs(utility_changes.get(agent_id, 0)) >= 0 for agent_id in utility_changes) and \
                                   any(abs(utility_changes.get(agent_id, 0)) > 0 for agent_id in utility_changes)
            
            results.append({
                "agent_count": agent_count,
                "case_number": case_number,
                "initial_total_cost": total_initial_cost,
                "final_total_cost": total_final_cost,
                "utility_gain": avg_utility_gain,
                "proposed_swaps": total_proposed_swaps,
                "accepted_swaps": accepted_swaps,
                "total_rounds": total_rounds,
                "exchange_approval_rate": exchange_approval_rate,
                "partial_approval_rate": partial_approval_rate,
                "is_pareto_improvement": is_pareto_improvement
            })
            
        except Exception as e:
            print(f"Error processing {dir_name}: {str(e)}")
    
    return pd.DataFrame(results)

def create_table2(df):
    """
    Create Table 2 with proper column names
    """
    if df.empty:
        return pd.DataFrame()
    
    # Calculate Pareto improvement count
    pareto_counts = df.groupby('agent_count')['is_pareto_improvement'].sum().to_dict()
    
    # Group by agent_count and calculate statistics
    stats = df.groupby('agent_count').agg({
        'initial_total_cost': 'mean',
        'final_total_cost': 'mean',
        'utility_gain': 'mean',
        'proposed_swaps': 'mean',
        'accepted_swaps': 'mean',
        'total_rounds': 'mean',
        'exchange_approval_rate': 'mean',
        'partial_approval_rate': 'mean',
        'is_pareto_improvement': 'count'
    })
    
    rows = []
    
    for agent_count in stats.index:
        # Calculate Pareto improvement percentage
        total_scenarios = stats.loc[agent_count, 'is_pareto_improvement']
        pareto_improvement_count = pareto_counts[agent_count]
        pareto_percentage = (pareto_improvement_count / total_scenarios) * 100 if total_scenarios > 0 else 0
        
        # No Negotiation row
        rows.append({
            '# Agents': agent_count,
            'Method': 'No Negotiation',
            'Avg. Total Cost': round(stats.loc[agent_count, 'initial_total_cost'], 1),
            'Avg. Agent Utility Gain': 'N/A',
            'Avg. Proposed Swaps': 'N/A',
            'Avg. Accepted Swaps': 'N/A',
            'Avg. Rounds': 'N/A',
            'Full Approval Rate': 'N/A',
            'Partial Approval Rate': 'N/A',
            'Pareto Improvement': 'N/A'
        })
        
        # With Negotiation row
        rows.append({
            '# Agents': agent_count,
            'Method': 'With Negotiation',
            'Avg. Total Cost': round(stats.loc[agent_count, 'final_total_cost'], 1),
            'Avg. Agent Utility Gain': round(stats.loc[agent_count, 'utility_gain'], 1),
            'Avg. Proposed Swaps': round(stats.loc[agent_count, 'proposed_swaps'], 1),
            'Avg. Accepted Swaps': round(stats.loc[agent_count, 'accepted_swaps'], 1),
            'Avg. Rounds': round(stats.loc[agent_count, 'total_rounds'], 1),
            'Full Approval Rate': f"{round(stats.loc[agent_count, 'exchange_approval_rate']*100, 1)}%",
            'Partial Approval Rate': f"{round(stats.loc[agent_count, 'partial_approval_rate']*100, 1)}%",
            'Pareto Improvement': f"{pareto_improvement_count}/{total_scenarios} ({round(pareto_percentage, 1)}%)"
        })
    
    table = pd.DataFrame(rows)
    
    # Column names are already properly formatted
    table = table[['# Agents', 'Method', 'Avg. Total Cost', 'Avg. Agent Utility Gain', 
                  'Avg. Proposed Swaps', 'Avg. Accepted Swaps', 'Avg. Rounds', 
                  'Full Approval Rate', 'Partial Approval Rate', 'Pareto Improvement']]
    
    return table

def analyze_cost_data(log_dirs):
    """
    Extract cost data for boxplot visualization
    
    Args:
        log_dirs (list): List of log directory paths
    
    Returns:
        pd.DataFrame: DataFrame with cost data
    """
    results = []
    
    for log_dir in log_dirs:
        dir_name = os.path.basename(log_dir)
        case_match = re.search(r'case_(\d+)', dir_name)
        case_number = int(case_match.group(1)) if case_match else 0
        
        routing_file = os.path.join(log_dir, "routing", f"routing_{dir_name}.json")
        if not os.path.exists(routing_file):
            continue
        
        try:
            with open(routing_file, 'r') as f:
                routing_data = json.load(f)
            
            # Derive agent_count from routing data
            agent_count = len(routing_data.get("initial_routes", {}))
            if agent_count == 0:
                continue
            
            initial_cost = sum(agent_data.get("cost", 0) for agent_id, agent_data 
                              in routing_data.get("initial_routes", {}).items())
            
            final_cost = sum(agent_data.get("cost", 0) for agent_id, agent_data 
                            in routing_data.get("final_routes", {}).items())
            
            results.append({
                "agent_count": agent_count,
                "case_number": case_number,
                "cost": initial_cost,
                "negotiation": "No Negotiation"
            })
            
            results.append({
                "agent_count": agent_count,
                "case_number": case_number,
                "cost": final_cost,
                "negotiation": "With Negotiation"
            })
            
        except Exception as e:
            print(f"Error processing {routing_file}: {str(e)}")
    
    return pd.DataFrame(results)

def create_cost_boxplot(df, output_dir="stats"):
    """
    Create boxplot visualization of total costs
    
    Args:
        df (pd.DataFrame): DataFrame with cost data
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Create boxplot using seaborn for better aesthetics
    ax = sns.boxplot(x="agent_count", y="cost", hue="negotiation", data=df,
                    palette={"No Negotiation": "cornflowerblue", "With Negotiation": "darkorange"})
    
    # Add individual data points for better visibility
    sns.stripplot(x="agent_count", y="cost", hue="negotiation", data=df, 
                 dodge=True, alpha=0.3, jitter=True, ax=ax,
                 palette={"No Negotiation": "darkblue", "With Negotiation": "darkred"})
    
    # Set labels and title
    plt.xlabel("Number of Agents", fontsize=14)
    plt.ylabel("Total System Cost", fontsize=14)
    plt.title("Distribution of Total System Costs With and Without Negotiation", fontsize=16)
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Method", fontsize=12)
    
    # Add grid for easier reading
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add statistics as text annotations
    stats = df.groupby(['agent_count', 'negotiation'])['cost'].agg(['mean', 'std'])
    
    for idx, agent_count in enumerate(sorted(df['agent_count'].unique())):
        y_pos = df['cost'].max() * 0.95
        for i, method in enumerate(["No Negotiation", "With Negotiation"]):
            if (agent_count, method) in stats.index:
                mean = stats.loc[(agent_count, method), 'mean']
                std = stats.loc[(agent_count, method), 'std']
                plt.text(idx + (-0.2 if i == 0 else 0.2), y_pos, 
                        f"Mean: {mean:.1f}\nStd: {std:.1f}",
                        fontsize=9, ha='center', va='top')
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, "cost_distribution_boxplot_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Boxplot saved to {output_path}")
    
    return output_path

def create_cost_line_plot(df, output_dir="stats"):
    """
    Create a line plot showing how total system cost changes with number of agents
    
    Args:
        df (pd.DataFrame): DataFrame with cost data from analyze_cost_data()
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group data by agent count and negotiation status to get means and std deviations
    cost_stats = df.groupby(['agent_count', 'negotiation'])['cost'].agg(['mean', 'std']).reset_index()
    
    # Create a pivot table for easier plotting
    pivot_df = cost_stats.pivot(index='agent_count', columns='negotiation', values=['mean', 'std'])
    
    # Flatten multi-index columns
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot line for No Negotiation with error bars
    plt.errorbar(pivot_df['agent_count'], pivot_df['No Negotiation_mean'], 
                yerr=pivot_df['No Negotiation_std'], 
                marker='o', markersize=8, linewidth=2, 
                capsize=6, capthick=2, color='#3274A1', 
                label='No Negotiation')
    
    # Plot line for With Negotiation with error bars
    plt.errorbar(pivot_df['agent_count'], pivot_df['With Negotiation_mean'], 
                yerr=pivot_df['With Negotiation_std'], 
                marker='s', markersize=8, linewidth=2, 
                capsize=6, capthick=2, color='#E1812C', 
                label='With Negotiation')
    
    # Add cost reduction percentages as annotations
    for i, row in pivot_df.iterrows():
        no_neg = row['No Negotiation_mean']
        with_neg = row['With Negotiation_mean']
        reduction = (no_neg - with_neg) / no_neg * 100 if no_neg > 0 else 0
        
        plt.annotate(f"{reduction:.1f}% reduction", 
                    xy=(row['agent_count'], with_neg), 
                    xytext=(0, -25), textcoords='offset points',
                    ha='center', va='top',
                    fontsize=10, color='#E1812C')
    
    # Add labels and title
    plt.xlabel('Number of Agents', fontsize=12)
    plt.ylabel('Total System Cost', fontsize=12)
    plt.title('Total System Cost vs. Number of Agents', fontsize=14)
    
    # Set x-axis ticks
    plt.xticks(pivot_df['agent_count'])
    
    # Add grid for readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(fontsize=11)
    
    # Annotate actual values on the points
    for i, row in pivot_df.iterrows():
        plt.annotate(f"{row['No Negotiation_mean']:.1f}", 
                    xy=(row['agent_count'], row['No Negotiation_mean']), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10)
        
        plt.annotate(f"{row['With Negotiation_mean']:.1f}", 
                    xy=(row['agent_count'], row['With Negotiation_mean']), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10)
    
    # Add explanation of error bars
    plt.figtext(0.5, 0.01, "Error bars show standard deviation across scenarios", 
               ha='center', fontsize=10, style='italic')
    
    # Tight layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    output_path = os.path.join(output_dir, "total_cost_vs_agents.png")
    plt.savefig(output_path, dpi=300)
    print(f"Line plot saved to {output_path}")
    
    return output_path

def get_log_directories():
    """
    Discover latest dataset from scenario generation data and return matching logs.
    
    Returns:
        list: List of log directory paths matching logs/<dataset>_case_*
    """
    import re
    ts_regex = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}$")
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_root = os.path.join(base_dir, "scenario_generation", "data")
    log_root = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir, "logs"))
    
    # Find latest timestamped dataset under data/
    candidate_dirs = [
        d for d in glob.glob(os.path.join(data_root, "*"))
        if os.path.isdir(d) and ts_regex.match(os.path.basename(d))
    ]
    
    if not candidate_dirs:
        print("No timestamped dataset folders found under scenario_generation/data.")
        return []
    
    latest_dataset = sorted(os.path.basename(d) for d in candidate_dirs)[-1]
    print(f"Using latest dataset from data/: {latest_dataset}")
    
    # Collect all logs for this dataset
    directories = sorted(glob.glob(os.path.join(log_root, f"{latest_dataset}_case_*")))
    print(f"Total directories: {len(directories)}")
    
    return directories

def calculate_avg_client_capacity():
    """
    Calculate the average client capacity for each agent count from scenario generation data
    
    Returns:
        dict: Dictionary mapping agent count to average client capacity
    """
    agent_folders = {
        3: "scenario_generation/data/2025_05_05_10_35_3agents",
        4: "scenario_generation/data/2025_05_05_14_04_4agents",
        5: "scenario_generation/data/2025_05_05_17_00_5agents",
        6: "scenario_generation/data/2025_05_05_20_49_6agents"
    }
    
    avg_client_capacities = {}
    
    for agent_count, folder_path in agent_folders.items():
        json_file = os.path.join(folder_path, "company_cases.json")
        
        if not os.path.exists(json_file):
            print(f"Warning: Company cases file not found for {agent_count} agents")
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            total_client_volume = 0
            total_clients = 0
            
            # Loop through all cases
            for case_name, case_data in data.items():
                # Loop through all companies
                for company_name, company_data in case_data.get("companies", {}).items():
                    # Get clients
                    clients = company_data.get("clients", [])
                    
                    # Sum volumes
                    for client in clients:
                        total_client_volume += client.get("volume", 0)
                        total_clients += 1
            
            # Calculate average
            if total_clients > 0:
                avg_client_capacities[agent_count] = total_client_volume / total_clients
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    return avg_client_capacities

def create_agent_scenario_table(log_dirs, output_dir="stats"):
    """
    Create a table showing agent and scenario statistics
    
    Args:
        log_dirs (list): List of log directory paths
        output_dir (str): Directory to save the results
    
    Returns:
        pd.DataFrame: DataFrame with agent and scenario statistics
    """
    # Get average client capacities from scenario generation data
    avg_client_capacities = calculate_avg_client_capacity()
    
    # Initialize data structures to collect statistics
    agent_stats = {}
    
    for log_dir in log_dirs:
        dir_name = os.path.basename(log_dir)
        routing_file = os.path.join(log_dir, "routing", f"routing_{dir_name}.json")
        
        if not os.path.exists(routing_file):
            continue
        
        try:
            with open(routing_file, 'r') as f:
                routing_data = json.load(f)
            
            # Derive agent_count from routing data
            agent_count = len(routing_data.get("initial_routes", {}))
            if agent_count == 0:
                continue

            # Initialize stats for this agent count if not already present
            if agent_count not in agent_stats:
                agent_stats[agent_count] = {
                    "scenario_count": 0,
                    "total_vehicles": 0,
                    "total_clients": 0,
                    "total_capacity": 0,  # Track total capacity
                    "total_load": 0,      # Track total max load
                    "vehicle_count": 0    # Track number of vehicles for averaging
                }
            
            # Count scenarios
            agent_stats[agent_count]["scenario_count"] += 1
            
            # Count vehicles, capacities per agent
            total_vehicles_in_scenario = 0
            total_capacity = 0
            total_load = 0
            vehicle_count = 0
            
            for agent_id, agent_data in routing_data.get("initial_routes", {}).items():
                agent_vehicles = agent_data.get("total_vehicles", 0)
                total_vehicles_in_scenario += agent_vehicles
                
                # Extract vehicle capacities and loads from routes
                for route in agent_data.get("routes", []):
                    if "capacity" in route:
                        total_capacity += route["capacity"]
                        # Also get max_load from the route
                        if "max_load" in route:
                            total_load += route["max_load"]
                        vehicle_count += 1
            
            avg_vehicles_per_agent = total_vehicles_in_scenario / agent_count
            agent_stats[agent_count]["total_vehicles"] += avg_vehicles_per_agent
            agent_stats[agent_count]["total_capacity"] += total_capacity
            agent_stats[agent_count]["total_load"] += total_load
            agent_stats[agent_count]["vehicle_count"] += vehicle_count
            
            # Count total clients in scenario
            all_clients = set()
            for agent_id, agent_data in routing_data.get("initial_routes", {}).items():
                clients = agent_data.get("clients_served", [])
                all_clients.update(clients)
            
            total_clients = len(all_clients)
            agent_stats[agent_count]["total_clients"] += total_clients
                
        except Exception as e:
            print(f"Error processing {routing_file}: {str(e)}")
    
    # Create DataFrame
    rows = []
    for agent_count, stats in agent_stats.items():
        scenario_count = stats["scenario_count"]
        avg_vehicles = stats["total_vehicles"] / scenario_count if scenario_count > 0 else 0
        avg_clients = stats["total_clients"] / scenario_count if scenario_count > 0 else 0
        
        # Calculate average vehicle capacity and load
        avg_capacity = stats["total_capacity"] / stats["vehicle_count"] if stats["vehicle_count"] > 0 else 0
        avg_load = stats["total_load"] / stats["vehicle_count"] if stats["vehicle_count"] > 0 else 0
        
        # Add average client capacity from scenario generation data
        avg_client_capacity = avg_client_capacities.get(agent_count, 0)
        
        rows.append({
            "Agent Count": agent_count,
            "Scenarios Run": scenario_count,
            "Avg Vehicles per Agent": round(avg_vehicles, 2),
            "Avg Vehicle Capacity": round(avg_capacity, 2),
            "Avg Vehicle Load": round(avg_load, 2),
            "Avg Client Capacity": round(avg_client_capacity, 2),
            "Avg Clients per Scenario": round(avg_clients, 2)
        })
    
    # Handle empty rows gracefully
    if not rows:
        print("No agent/scenario data available to build table.")
        return pd.DataFrame()
    
    # Sort by agent count
    df = pd.DataFrame(rows).sort_values("Agent Count")
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "agent_scenario_stats.csv"), 
              float_format="%.2f", index=False)
    
    # Print table
    print("\nAgent and Scenario Statistics:")
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    # Get all log directories
    log_dirs = get_log_directories()
    
    if not log_dirs:
        print("No log directories found matching the specified patterns!")
        exit(1)
    
    # Derive dataset name and output directory under stats/<dataset>_analyze
    first_session_name = os.path.basename(log_dirs[0])
    dataset_name = first_session_name.split("_case_")[0] if "_case_" in first_session_name else first_session_name
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "stats", f"{dataset_name}_analyze")
    os.makedirs(output_dir, exist_ok=True)
    
    # Diagnostic: report presence of routing/negotiation JSONs per session
    report_lines = [f"Dataset: {dataset_name}", f"Logs scanned: {len(log_dirs)}", ""]
    missing_routing = 0
    missing_negotiation = 0
    for log_dir in log_dirs:
        session = os.path.basename(log_dir)
        routing_json = os.path.join(log_dir, "routing", f"routing_{session}.json")
        negotiation_json = os.path.join(log_dir, "negotiation", f"negotiation_{session}.json")
        has_routing = os.path.exists(routing_json)
        has_neg = os.path.exists(negotiation_json)
        if not has_routing:
            missing_routing += 1
        if not has_neg:
            missing_negotiation += 1
        report_lines.append(f"{session} | routing: {'OK' if has_routing else 'MISSING'} | negotiation: {'OK' if has_neg else 'MISSING'}")
    report_lines.append("")
    report_lines.append(f"Missing routing files: {missing_routing}")
    report_lines.append(f"Missing negotiation files: {missing_negotiation}")
    with open(os.path.join(output_dir, "run_report.txt"), "w", encoding="utf-8") as rf:
        rf.write("\n".join(report_lines))
    print(f"Run report saved to {os.path.join(output_dir, 'run_report.txt')}")
    
    # Analyze costs from routing logs
    print("Analyzing costs from routing logs...")
    df = analyze_agent_costs(log_dirs)
    
    if df.empty:
        print("No data found for cost analysis.")
    else:
        # Generate statistics
        stats = generate_statistics(df)
        print("\nStatistics by Agent Count:")
        print(stats[['initial_total_cost_mean', 'final_total_cost_mean', 
                    'cost_reduction_mean', 'initial_total_cost_count']])
        
        # Generate plots
        generate_plots(df, stats, output_dir=output_dir)
        
        # Save results
        save_results(df, stats, output_dir=output_dir)
        
        print(f"\nCost analysis complete for {len(df)} scenarios.")
    
    # Analyze negotiation results
    print("\nAnalyzing negotiation results...")
    df_negotiation = analyze_negotiation_results(log_dirs)
    
    if df_negotiation.empty:
        print("No data found for negotiation analysis.")
    else:
        # Create Table 2
        table2 = create_table2(df_negotiation)
        
        # Print the table
        print("\nTable 2 — Averaged Results with Additional Metrics")
        print(table2.to_string(index=False))
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        table2.to_csv(os.path.join(output_dir, "table2_expanded_results.csv"), 
                      float_format="%.2f", index=False)
        print(f"\nTable saved to {os.path.join(output_dir, 'table2_expanded_results.csv')}")
    
    # Analyze cost data for boxplot
    print("\nCreating cost distribution boxplot...")
    df_cost_data = analyze_cost_data(log_dirs)
    
    if df_cost_data.empty:
        print("No data found for cost distribution boxplot.")
    else:
        create_cost_boxplot(df_cost_data, output_dir=output_dir)
        print(f"Boxplot created successfully.")
    
    # Create line plot of cost vs. agent count
    print("\nCreating line plot of total cost vs. number of agents...")
    if not df_cost_data.empty:
        create_cost_line_plot(df_cost_data, output_dir=output_dir)
        print(f"Line plot created successfully.")
    
    # Create agent and scenario statistics table
    print("\nCreating agent and scenario statistics table...")
    create_agent_scenario_table(log_dirs, output_dir=output_dir)
    
    print("\nAll analyses complete!")
