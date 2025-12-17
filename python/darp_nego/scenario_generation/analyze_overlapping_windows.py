#!/usr/bin/env python3
"""
Comprehensive Analysis and Visualization of DARP Cases with Overlapping Time Windows

This script analyzes the generated DARP scenarios with overlapping time windows
and creates various visualizations to understand the data characteristics.
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import glob
from collections import defaultdict
import math

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_latest_data():
    """Load the latest generated data file."""
    data_dir = "data"
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    subdirs.sort(reverse=True)
    
    if not subdirs:
        raise FileNotFoundError("No data directories found")
    
    latest_dir = subdirs[0]
    file_path = os.path.join(data_dir, latest_dir, "company_cases.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"company_cases.json not found in {latest_dir}")
    
    print(f"Loading data from: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data, latest_dir

def analyze_time_windows(data):
    """Analyze time window characteristics across all cases."""
    time_window_stats = []
    
    for case_name, case_data in data.items():
        companies = case_data["companies"]
        
        for company_name, company_data in companies.items():
            clients = company_data["clients"]
            
            for client in clients:
                pickup_window = client["late_pickup"] - client["early_pickup"]
                drop_window = client["late_drop_off"] - client["early_drop_off"]
                total_duration = client["late_drop_off"] - client["early_pickup"]
                
                time_window_stats.append({
                    "case": case_name,
                    "company": company_name,
                    "client_id": client["client_id"],
                    "pickup_window": pickup_window,
                    "drop_window": drop_window,
                    "total_duration": total_duration,
                    "early_pickup": client["early_pickup"],
                    "late_pickup": client["late_pickup"],
                    "early_drop": client["early_drop_off"],
                    "late_drop": client["late_drop_off"],
                    "volume": client["volume"]
                })
    
    return pd.DataFrame(time_window_stats)

def analyze_overlapping_pairs(data):
    """Analyze overlapping time window pairs within each company."""
    overlap_stats = []
    
    for case_name, case_data in data.items():
        companies = case_data["companies"]
        
        for company_name, company_data in companies.items():
            clients = company_data["clients"]
            overlapping_pairs = []
            
            # Check all pairs of clients for overlapping time windows
            for i in range(len(clients)):
                for j in range(i + 1, len(clients)):
                    client1 = clients[i]
                    client2 = clients[j]
                    
                    # Check if pickup windows overlap
                    pickup_overlap = (
                        client1["early_pickup"] <= client2["late_pickup"] and 
                        client2["early_pickup"] <= client1["late_pickup"]
                    )
                    
                    # Check if drop-off windows overlap
                    drop_overlap = (
                        client1["early_drop_off"] <= client2["late_drop_off"] and 
                        client2["early_drop_off"] <= client1["late_drop_off"]
                    )
                    
                    if pickup_overlap or drop_overlap:
                        # Calculate overlap duration
                        if pickup_overlap:
                            overlap_start = max(client1["early_pickup"], client2["early_pickup"])
                            overlap_end = min(client1["late_pickup"], client2["late_pickup"])
                            pickup_overlap_duration = overlap_end - overlap_start
                        else:
                            pickup_overlap_duration = 0
                        
                        if drop_overlap:
                            overlap_start = max(client1["early_drop_off"], client2["early_drop_off"])
                            overlap_end = min(client1["late_drop_off"], client2["late_drop_off"])
                            drop_overlap_duration = overlap_end - overlap_start
                        else:
                            drop_overlap_duration = 0
                        
                        overlapping_pairs.append({
                            "client1_id": client1["client_id"],
                            "client2_id": client2["client_id"],
                            "pickup_overlap": pickup_overlap,
                            "drop_overlap": drop_overlap,
                            "pickup_overlap_duration": pickup_overlap_duration,
                            "drop_overlap_duration": drop_overlap_duration,
                            "total_overlap_duration": pickup_overlap_duration + drop_overlap_duration
                        })
            
            overlap_stats.append({
                "case": case_name,
                "company": company_name,
                "total_clients": len(clients),
                "overlapping_pairs": len(overlapping_pairs),
                "overlap_ratio": len(overlapping_pairs) / (len(clients) * (len(clients) - 1) / 2) if len(clients) > 1 else 0,
                "avg_overlap_duration": np.mean([p["total_overlap_duration"] for p in overlapping_pairs]) if overlapping_pairs else 0,
                "max_overlap_duration": max([p["total_overlap_duration"] for p in overlapping_pairs]) if overlapping_pairs else 0
            })
    
    return pd.DataFrame(overlap_stats)

def analyze_case_complexity(data):
    """Analyze the complexity of each case based on various metrics."""
    case_stats = []
    
    for case_name, case_data in data.items():
        companies = case_data["companies"]
        total_clients = sum(len(company_data["clients"]) for company_data in companies.values())
        total_vehicles = sum(len(company_data["vehicles"]) for company_data in companies.values())
        num_companies = len(companies)
        
        # Calculate time window density
        all_clients = []
        for company_data in companies.values():
            all_clients.extend(company_data["clients"])
        
        # Calculate time span
        all_pickup_times = [c["early_pickup"] for c in all_clients] + [c["late_pickup"] for c in all_clients]
        all_drop_times = [c["early_drop_off"] for c in all_clients] + [c["late_drop_off"] for c in all_clients]
        
        time_span = max(all_pickup_times + all_drop_times) - min(all_pickup_times + all_drop_times)
        
        # Calculate average time window size
        avg_pickup_window = np.mean([c["late_pickup"] - c["early_pickup"] for c in all_clients])
        avg_drop_window = np.mean([c["late_drop_off"] - c["early_drop_off"] for c in all_clients])
        
        case_stats.append({
            "case": case_name,
            "num_companies": num_companies,
            "total_clients": total_clients,
            "total_vehicles": total_vehicles,
            "clients_per_company": total_clients / num_companies,
            "vehicles_per_company": total_vehicles / num_companies,
            "time_span": time_span,
            "avg_pickup_window": avg_pickup_window,
            "avg_drop_window": avg_drop_window,
            "complexity_score": (total_clients * num_companies * time_span) / 1000  # Simple complexity metric
        })
    
    return pd.DataFrame(case_stats)

def create_visualizations(time_window_df, overlap_df, case_df, data, output_dir="analysis_results"):
    """Create comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Time Window Distribution Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pickup window distribution
    axes[0, 0].hist(time_window_df['pickup_window'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Pickup Time Windows')
    axes[0, 0].set_xlabel('Pickup Window Duration')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Drop window distribution
    axes[0, 1].hist(time_window_df['drop_window'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Distribution of Drop-off Time Windows')
    axes[0, 1].set_xlabel('Drop-off Window Duration')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Total duration distribution
    axes[1, 0].hist(time_window_df['total_duration'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Distribution of Total Trip Duration')
    axes[1, 0].set_xlabel('Total Duration')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Volume distribution
    axes[1, 1].hist(time_window_df['volume'], bins=20, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_title('Distribution of Client Volumes')
    axes[1, 1].set_xlabel('Volume')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_window_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Overlap Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overlapping pairs per company
    overlap_counts = overlap_df['overlapping_pairs'].value_counts()
    axes[0, 0].bar(overlap_counts.index, overlap_counts.values, color='purple', alpha=0.7)
    axes[0, 0].set_title('Distribution of Overlapping Pairs per Company')
    axes[0, 0].set_xlabel('Number of Overlapping Pairs')
    axes[0, 0].set_ylabel('Number of Companies')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Overlap ratio vs number of clients
    axes[0, 1].scatter(overlap_df['total_clients'], overlap_df['overlap_ratio'], 
                      alpha=0.6, s=50, color='orange')
    axes[0, 1].set_title('Overlap Ratio vs Number of Clients')
    axes[0, 1].set_xlabel('Number of Clients')
    axes[0, 1].set_ylabel('Overlap Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average overlap duration
    axes[1, 0].hist(overlap_df['avg_overlap_duration'], bins=20, alpha=0.7, color='teal', edgecolor='black')
    axes[1, 0].set_title('Distribution of Average Overlap Duration')
    axes[1, 0].set_xlabel('Average Overlap Duration')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Max overlap duration
    axes[1, 1].hist(overlap_df['max_overlap_duration'], bins=20, alpha=0.7, color='brown', edgecolor='black')
    axes[1, 1].set_title('Distribution of Maximum Overlap Duration')
    axes[1, 1].set_xlabel('Maximum Overlap Duration')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overlap_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Case Complexity Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Complexity score distribution
    axes[0, 0].hist(case_df['complexity_score'], bins=20, alpha=0.7, color='darkblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Case Complexity Scores')
    axes[0, 0].set_xlabel('Complexity Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Clients vs Companies
    axes[0, 1].scatter(case_df['num_companies'], case_df['total_clients'], 
                      alpha=0.6, s=50, color='red')
    axes[0, 1].set_title('Total Clients vs Number of Companies')
    axes[0, 1].set_xlabel('Number of Companies')
    axes[0, 1].set_ylabel('Total Clients')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time span vs complexity
    axes[1, 0].scatter(case_df['time_span'], case_df['complexity_score'], 
                      alpha=0.6, s=50, color='green')
    axes[1, 0].set_title('Time Span vs Complexity Score')
    axes[1, 0].set_xlabel('Time Span')
    axes[1, 0].set_ylabel('Complexity Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Average time windows
    axes[1, 1].scatter(case_df['avg_pickup_window'], case_df['avg_drop_window'], 
                      alpha=0.6, s=50, color='purple')
    axes[1, 1].set_title('Average Pickup vs Drop-off Windows')
    axes[1, 1].set_xlabel('Average Pickup Window')
    axes[1, 1].set_ylabel('Average Drop-off Window')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'case_complexity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary Statistics Table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate summary statistics
    summary_stats = [
        ['Metric', 'Value'],
        ['Total Cases', len(case_df)],
        ['Total Companies', len(overlap_df)],
        ['Total Clients', len(time_window_df)],
        ['Average Clients per Company', f"{time_window_df.groupby('company').size().mean():.1f}"],
        ['Average Pickup Window', f"{time_window_df['pickup_window'].mean():.1f}"],
        ['Average Drop Window', f"{time_window_df['drop_window'].mean():.1f}"],
        ['Average Overlap Duration', f"{overlap_df['avg_overlap_duration'].mean():.1f}"],
        ['Companies with Overlaps', f"{len(overlap_df[overlap_df['overlapping_pairs'] > 0])}"],
        ['Average Complexity Score', f"{case_df['complexity_score'].mean():.1f}"],
        ['Average Time Span', f"{case_df['time_span'].mean():.1f}"],
        ['Average Volume', f"{time_window_df['volume'].mean():.1f}"]
    ]
    
    table = ax.table(cellText=summary_stats[1:], colLabels=summary_stats[0], 
                    cellLoc='center', loc='center', colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_stats)):
        for j in range(2):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#2E86AB')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                table[(i, j)].set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')
    
    plt.title('Overlapping Time Windows Analysis Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Time Window Timeline Visualization (for a sample case)
    sample_case = list(data.keys())[0]
    sample_data = data[sample_case]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(sample_data["companies"])))
    
    for i, (company_name, company_data) in enumerate(sample_data["companies"].items()):
        for client in company_data["clients"]:
            # Plot pickup window
            ax.barh(f"Client {client['client_id']} ({company_name})", 
                   client["late_pickup"] - client["early_pickup"], 
                   left=client["early_pickup"], 
                   color=colors[i], alpha=0.7, label=f"{company_name} Pickup" if client == company_data["clients"][0] else "")
            
            # Plot drop-off window
            ax.barh(f"Client {client['client_id']} ({company_name})", 
                   client["late_drop_off"] - client["early_drop_off"], 
                   left=client["early_drop_off"], 
                   color=colors[i], alpha=0.3, label=f"{company_name} Drop-off" if client == company_data["clients"][0] else "")
    
    ax.set_xlabel('Time')
    ax.set_title(f'Time Windows Timeline - {sample_case}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_window_timeline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}/")

def print_detailed_analysis(time_window_df, overlap_df, case_df):
    """Print detailed analysis results."""
    print("\n" + "="*80)
    print("DETAILED OVERLAPPING TIME WINDOWS ANALYSIS")
    print("="*80)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total Cases: {len(case_df)}")
    print(f"   Total Companies: {len(overlap_df)}")
    print(f"   Total Clients: {len(time_window_df)}")
    print(f"   Average Clients per Company: {time_window_df.groupby('company').size().mean():.1f}")
    
    print(f"\n⏰ TIME WINDOW CHARACTERISTICS:")
    print(f"   Average Pickup Window: {time_window_df['pickup_window'].mean():.1f} minutes")
    print(f"   Average Drop-off Window: {time_window_df['drop_window'].mean():.1f} minutes")
    print(f"   Average Total Duration: {time_window_df['total_duration'].mean():.1f} minutes")
    print(f"   Average Client Volume: {time_window_df['volume'].mean():.1f}")
    
    print(f"\n🔄 OVERLAP ANALYSIS:")
    companies_with_overlaps = len(overlap_df[overlap_df['overlapping_pairs'] > 0])
    print(f"   Companies with Overlapping Windows: {companies_with_overlaps}/{len(overlap_df)} ({companies_with_overlaps/len(overlap_df)*100:.1f}%)")
    print(f"   Average Overlapping Pairs per Company: {overlap_df['overlapping_pairs'].mean():.1f}")
    print(f"   Average Overlap Duration: {overlap_df['avg_overlap_duration'].mean():.1f} minutes")
    print(f"   Maximum Overlap Duration: {overlap_df['max_overlap_duration'].max():.1f} minutes")
    
    print(f"\n📈 CASE COMPLEXITY:")
    print(f"   Average Complexity Score: {case_df['complexity_score'].mean():.1f}")
    print(f"   Average Time Span: {case_df['time_span'].mean():.1f} minutes")
    print(f"   Average Companies per Case: {case_df['num_companies'].mean():.1f}")
    print(f"   Average Clients per Case: {case_df['total_clients'].mean():.1f}")
    
    print(f"\n🏆 TOP 5 MOST COMPLEX CASES:")
    top_complex = case_df.nlargest(5, 'complexity_score')
    for _, row in top_complex.iterrows():
        print(f"   {row['case']}: Score {row['complexity_score']:.1f} ({row['num_companies']} companies, {row['total_clients']} clients)")
    
    print(f"\n🔍 COMPANIES WITH MOST OVERLAPS:")
    top_overlaps = overlap_df.nlargest(5, 'overlapping_pairs')
    for _, row in top_overlaps.iterrows():
        print(f"   {row['company']}: {row['overlapping_pairs']} overlapping pairs ({row['total_clients']} clients)")

def main():
    """Main analysis function."""
    print("🔍 Analyzing DARP Cases with Overlapping Time Windows")
    print("=" * 60)
    
    try:
        # Load data
        data, data_dir = load_latest_data()
        print(f"✅ Loaded data from {data_dir}")
        
        # Perform analysis
        print("\n📊 Analyzing time windows...")
        time_window_df = analyze_time_windows(data)
        
        print("🔄 Analyzing overlapping pairs...")
        overlap_df = analyze_overlapping_pairs(data)
        
        print("📈 Analyzing case complexity...")
        case_df = analyze_case_complexity(data)
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        output_dir = f"analysis_results_{data_dir}"
        create_visualizations(time_window_df, overlap_df, case_df, data, output_dir)
        
        # Print detailed analysis
        print_detailed_analysis(time_window_df, overlap_df, case_df)
        
        # Save analysis data
        print(f"\n💾 Saving analysis data...")
        time_window_df.to_csv(f"{output_dir}/time_window_analysis.csv", index=False)
        overlap_df.to_csv(f"{output_dir}/overlap_analysis.csv", index=False)
        case_df.to_csv(f"{output_dir}/case_complexity_analysis.csv", index=False)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}/")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 