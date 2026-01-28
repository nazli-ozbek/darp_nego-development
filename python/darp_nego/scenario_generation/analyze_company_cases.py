#!/usr/bin/env python3

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import math

def load_company_cases(file_path):
    """Load company cases data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_company_cases(data):
    """Analyze company cases data and extract statistics."""
    analysis_results = []
    
    for case_name, case_data in data.items():
        if case_name == "time_matrix" or case_name == "coordinates":
            continue
            
        case_stats = {
            'case_name': case_name,
            'num_companies': len(case_data.get('companies', {})),
            'total_vehicles': 0,
            'total_clients': 0,
            'total_volume': 0,
            'avg_vehicle_capacity': 0,
            'time_windows': [],
            'volumes': [],
            'pickup_windows': [],
            'dropoff_windows': [],
            'companies': []
        }
        
        companies = case_data.get('companies', {})
        total_vehicle_capacity = 0
        
        for company_name, company_data in companies.items():
            company_id = int(company_name.split('_')[1])
            
            # Analyze vehicles
            vehicles = company_data.get('vehicles', [])
            company_vehicles = len(vehicles)
            company_capacity = sum(v.get('max_volume', 0) for v in vehicles)
            total_vehicle_capacity += company_capacity
            
            # Analyze clients
            clients = company_data.get('clients', [])
            company_clients = len(clients)
            company_volume = sum(c.get('volume', 0) for c in clients)
            
            # Collect time window data
            for client in clients:
                pickup_window = client.get('late_pickup', 0) - client.get('early_pickup', 0)
                dropoff_window = client.get('late_drop_off', 0) - client.get('early_drop_off', 0)
                
                case_stats['time_windows'].append(pickup_window + dropoff_window)
                case_stats['volumes'].append(client.get('volume', 0))
                case_stats['pickup_windows'].append(pickup_window)
                case_stats['dropoff_windows'].append(dropoff_window)
            
            case_stats['companies'].append({
                'company_id': company_id,
                'num_vehicles': company_vehicles,
                'num_clients': company_clients,
                'total_volume': company_volume,
                'vehicle_capacity': company_capacity
            })
            
            case_stats['total_vehicles'] += company_vehicles
            case_stats['total_clients'] += company_clients
            case_stats['total_volume'] += company_volume
        
        case_stats['avg_vehicle_capacity'] = total_vehicle_capacity / max(case_stats['total_vehicles'], 1)
        case_stats['avg_time_window'] = np.mean(case_stats['time_windows']) if case_stats['time_windows'] else 0
        case_stats['avg_volume'] = np.mean(case_stats['volumes']) if case_stats['volumes'] else 0
        case_stats['avg_pickup_window'] = np.mean(case_stats['pickup_windows']) if case_stats['pickup_windows'] else 0
        case_stats['avg_dropoff_window'] = np.mean(case_stats['dropoff_windows']) if case_stats['dropoff_windows'] else 0
        
        analysis_results.append(case_stats)
    
    return analysis_results

def create_visualizations(analysis_results, output_dir="company_cases_analysis"):
    """Create comprehensive visualizations of the company cases data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(analysis_results)
    
    # Set up plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. COMPANY DISTRIBUTION
    plt.figure(figsize=(12, 6))
    company_counts = df['num_companies'].value_counts().sort_index()
    bars = plt.bar(company_counts.index, company_counts.values, color='#339af0', alpha=0.8)
    plt.xlabel('Number of Companies')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of Cases by Number of Companies', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'company_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. VEHICLE AND CLIENT DISTRIBUTION
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vehicles per case
    ax1.hist(df['total_vehicles'], bins=range(0, max(df['total_vehicles'])+2, 1), 
             alpha=0.7, color='#339af0', edgecolor='black')
    ax1.set_xlabel('Total Vehicles per Case')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Total Vehicles per Case', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Clients per case
    ax2.hist(df['total_clients'], bins=range(0, max(df['total_clients'])+2, 1), 
             alpha=0.7, color='#51cf66', edgecolor='black')
    ax2.set_xlabel('Total Clients per Case')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Total Clients per Case', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vehicle_client_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. VOLUME ANALYSIS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Total volume per case
    ax1.scatter(df['total_clients'], df['total_volume'], alpha=0.7, color='#339af0', s=50)
    ax1.set_xlabel('Total Clients')
    ax1.set_ylabel('Total Volume')
    ax1.set_title('Total Volume vs Total Clients', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Average volume per client
    avg_volumes = [np.mean(volumes) if volumes else 0 for volumes in df['volumes']]
    ax2.hist(avg_volumes, bins=20, alpha=0.7, color='#51cf66', edgecolor='black')
    ax2.set_xlabel('Average Volume per Client')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Average Volume per Client', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volume_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. TIME WINDOW ANALYSIS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pickup windows
    all_pickup_windows = []
    for windows in df['pickup_windows']:
        all_pickup_windows.extend(windows)
    
    ax1.hist(all_pickup_windows, bins=30, alpha=0.7, color='#339af0', edgecolor='black')
    ax1.set_xlabel('Pickup Time Window (minutes)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Pickup Time Windows', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Dropoff windows
    all_dropoff_windows = []
    for windows in df['dropoff_windows']:
        all_dropoff_windows.extend(windows)
    
    ax2.hist(all_dropoff_windows, bins=30, alpha=0.7, color='#51cf66', edgecolor='black')
    ax2.set_xlabel('Dropoff Time Window (minutes)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Dropoff Time Windows', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_window_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. CORRELATION MATRIX
    numeric_cols = ['num_companies', 'total_vehicles', 'total_clients', 'total_volume', 
                   'avg_vehicle_capacity', 'avg_time_window', 'avg_volume']
    correlation_df = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Key Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. SUMMARY STATISTICS TABLE
    plt.figure(figsize=(14, 8))
    plt.axis('tight')
    plt.axis('off')
    
    # Calculate summary statistics
    summary_stats = {
        'Total Cases': len(df),
        'Average Companies per Case': f"{df['num_companies'].mean():.1f}",
        'Average Vehicles per Case': f"{df['total_vehicles'].mean():.1f}",
        'Average Clients per Case': f"{df['total_clients'].mean():.1f}",
        'Average Volume per Case': f"{df['total_volume'].mean():.1f}",
        'Average Vehicle Capacity': f"{df['avg_vehicle_capacity'].mean():.1f}",
        'Average Time Window': f"{df['avg_time_window'].mean():.1f} minutes",
        'Average Volume per Client': f"{df['avg_volume'].mean():.1f}",
        'Total Vehicles': f"{df['total_vehicles'].sum()}",
        'Total Clients': f"{df['total_clients'].sum()}",
        'Total Volume': f"{df['total_volume'].sum()}"
    }
    
    table_data = [[key, value] for key, value in summary_stats.items()]
    
    table = plt.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                     cellLoc='left', loc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(2):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#339af0')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                table[(i, j)].set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    plt.title('Company Cases Analysis Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. DETAILED BREAKDOWN BY COMPANY COUNT
    company_breakdown = df.groupby('num_companies').agg({
        'total_vehicles': ['count', 'mean', 'sum'],
        'total_clients': ['mean', 'sum'],
        'total_volume': ['mean', 'sum'],
        'avg_vehicle_capacity': 'mean',
        'avg_time_window': 'mean'
    }).round(2)
    
    # Flatten column names
    company_breakdown.columns = ['_'.join(col).strip() for col in company_breakdown.columns]
    company_breakdown = company_breakdown.reset_index()
    
    # Create table for company breakdown
    plt.figure(figsize=(16, 10))
    plt.axis('tight')
    plt.axis('off')
    
    # Prepare table data
    breakdown_data = [['Companies', 'Cases', 'Avg Vehicles', 'Total Vehicles', 'Avg Clients', 
                       'Total Clients', 'Avg Volume', 'Total Volume', 'Avg Capacity', 'Avg Time Window']]
    
    for _, row in company_breakdown.iterrows():
        breakdown_data.append([
            f"{row['num_companies']}",
            f"{row['total_vehicles_count']}",
            f"{row['total_vehicles_mean']:.1f}",
            f"{row['total_vehicles_sum']}",
            f"{row['total_clients_mean']:.1f}",
            f"{row['total_clients_sum']}",
            f"{row['total_volume_mean']:.1f}",
            f"{row['total_volume_sum']}",
            f"{row['avg_vehicle_capacity_mean']:.1f}",
            f"{row['avg_time_window_mean']:.1f}"
        ])
    
    table = plt.table(cellText=breakdown_data[1:], colLabels=breakdown_data[0], 
                     cellLoc='center', loc='center', colWidths=[0.1]*10)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(breakdown_data)):
        for j in range(10):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#339af0')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                table[(i, j)].set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    plt.title('Detailed Breakdown by Company Count', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'company_breakdown_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results to CSV
    df.to_csv(os.path.join(output_dir, 'detailed_analysis.csv'), index=False)
    
    return df, summary_stats

def main():
    """Main function to analyze and visualize company cases."""
    file_path = "python/darp_nego/scenario_generation/data/2025_11_06_21_52/company_cases.json"
    # Derive output folder similar to case_visuals: parent dir name of data file
    data_folder_name = os.path.basename(os.path.dirname(file_path))
    output_dir = os.path.join("company_cases_analysis", data_folder_name)
    
    print("Loading company cases data...")
    data = load_company_cases(file_path)
    
    print("Analyzing data...")
    analysis_results = analyze_company_cases(data)
    
    print("Creating visualizations...")
    df, summary_stats = create_visualizations(analysis_results, output_dir=output_dir)
    
    print("\n" + "="*60)
    print("COMPANY CASES ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total cases analyzed: {len(df)}")
    print(f"Output directory: {output_dir}/")
    print("\nGenerated visualizations:")
    print("- company_distribution.png")
    print("- vehicle_client_distribution.png")
    print("- volume_analysis.png")
    print("- time_window_analysis.png")
    print("- correlation_matrix.png")
    print("- summary_table.png")
    print("- company_breakdown_table.png")
    print("- detailed_analysis.csv")
    
    return df, summary_stats

if __name__ == "__main__":
    main() 