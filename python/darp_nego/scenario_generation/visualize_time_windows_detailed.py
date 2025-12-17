import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def load_data(file_path):
    """Load the time window outlier data"""
    with open(file_path, 'r') as f:
        return json.load(f)

def visualize_time_windows_detailed(data, case_name="case_1", save_plot=True):
    """
    Create a detailed visualization of time windows with clear outlier highlighting
    """
    case_data = data[case_name]
    companies = case_data['companies']
    
    # Calculate total number of clients for proper spacing
    total_clients = sum(len(company['clients']) for company in companies.values())
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    fig.suptitle(f'Detailed Time Windows Analysis - {case_name}\n(Red bars = Outliers with almost-overlapping windows)', 
                 fontsize=16, fontweight='bold')
    
    # Colors for different companies
    company_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    y_pos = 0
    outlier_info = []
    
    # Plot 1: Pickup Windows
    ax1.set_title('Pickup Time Windows', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Clients')
    ax1.grid(True, alpha=0.3)
    
    for company_idx, (company_id, company_data) in enumerate(companies.items()):
        clients = company_data['clients']
        color = company_colors[company_idx % len(company_colors)]
        
        # Calculate pickup windows
        pickup_windows = []
        for client in clients:
            pickup_window = client['late_pickup'] - client['early_pickup']
            pickup_windows.append(pickup_window)
        
        # Find outlier
        outlier_idx = np.argmin(pickup_windows)
        avg_pickup = np.mean(pickup_windows)
        
        for client_idx, client in enumerate(clients):
            pickup_window = client['late_pickup'] - client['early_pickup']
            
            # Determine if this is an outlier
            is_outlier = client_idx == outlier_idx
            
            if is_outlier:
                bar_color = 'red'
                edge_color = 'darkred'
                linewidth = 3
                alpha = 1.0
                outlier_info.append(f"{company_id} Client {client['client_id']}: {pickup_window} min (avg: {avg_pickup:.1f} min)")
            else:
                bar_color = color
                edge_color = 'black'
                linewidth = 1
                alpha = 0.7
            
            # Plot pickup window bar
            ax1.barh(y_pos, pickup_window, 
                    left=client['early_pickup'],
                    height=0.8, color=bar_color, alpha=alpha,
                    edgecolor=edge_color, linewidth=linewidth)
            
            # Add text label
            ax1.text(client['early_pickup'] + pickup_window/2, y_pos, 
                    f"{pickup_window}m", ha='center', va='center', 
                    fontweight='bold' if is_outlier else 'normal',
                    color='white' if is_outlier else 'black')
            
            y_pos += 1
    
    # Set y-axis labels
    y_pos = 0
    y_labels = []
    for company_id, company_data in companies.items():
        for client in company_data['clients']:
            y_labels.append(f"{company_id} C{client['client_id']}")
            y_pos += 1
    
    ax1.set_yticks(range(len(y_labels)))
    ax1.set_yticklabels(y_labels, fontsize=10)
    
    # Plot 2: Drop-off Windows
    ax2.set_title('Drop-off Time Windows', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Clients')
    ax2.grid(True, alpha=0.3)
    
    y_pos = 0
    for company_idx, (company_id, company_data) in enumerate(companies.items()):
        clients = company_data['clients']
        color = company_colors[company_idx % len(company_colors)]
        
        # Calculate drop windows
        drop_windows = []
        for client in clients:
            drop_window = client['late_drop_off'] - client['early_drop_off']
            drop_windows.append(drop_window)
        
        # Find outlier
        outlier_idx = np.argmin(drop_windows)
        avg_drop = np.mean(drop_windows)
        
        for client_idx, client in enumerate(clients):
            drop_window = client['late_drop_off'] - client['early_drop_off']
            
            # Determine if this is an outlier
            is_outlier = client_idx == outlier_idx
            
            if is_outlier:
                bar_color = 'red'
                edge_color = 'darkred'
                linewidth = 3
                alpha = 1.0
            else:
                bar_color = color
                edge_color = 'black'
                linewidth = 1
                alpha = 0.7
            
            # Plot drop window bar
            ax2.barh(y_pos, drop_window, 
                    left=client['early_drop_off'],
                    height=0.8, color=bar_color, alpha=alpha,
                    edgecolor=edge_color, linewidth=linewidth)
            
            # Add text label
            ax2.text(client['early_drop_off'] + drop_window/2, y_pos, 
                    f"{drop_window}m", ha='center', va='center', 
                    fontweight='bold' if is_outlier else 'normal',
                    color='white' if is_outlier else 'black')
            
            y_pos += 1
    
    ax2.set_yticks(range(len(y_labels)))
    ax2.set_yticklabels(y_labels, fontsize=10)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='red', label='Outliers (1-3 min windows)'),
        mpatches.Patch(color='#1f77b4', label='Normal clients (10-25 min windows)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Add outlier summary
    outlier_text = "OUTLIER SUMMARY:\n" + "\n".join(outlier_info[:6])  # Show first 6
    if len(outlier_info) > 6:
        outlier_text += f"\n... and {len(outlier_info) - 6} more outliers"
    
    fig.text(0.02, 0.02, outlier_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'time_windows_detailed_{case_name}.png', dpi=300, bbox_inches='tight')
        print(f"Detailed plot saved as: time_windows_detailed_{case_name}.png")
    
    plt.show()

def create_outlier_comparison_chart(data, case_name="case_1", save_plot=True):
    """
    Create a comparison chart showing outlier vs normal time windows
    """
    case_data = data[case_name]
    companies = case_data['companies']
    
    # Collect data for comparison
    normal_pickup_windows = []
    outlier_pickup_windows = []
    normal_drop_windows = []
    outlier_drop_windows = []
    company_names = []
    
    for company_id, company_data in companies.items():
        clients = company_data['clients']
        
        # Calculate windows
        pickup_windows = [client['late_pickup'] - client['early_pickup'] for client in clients]
        drop_windows = [client['late_drop_off'] - client['early_drop_off'] for client in clients]
        
        # Find outliers
        pickup_outlier_idx = np.argmin(pickup_windows)
        drop_outlier_idx = np.argmin(drop_windows)
        
        # Separate normal and outlier data
        for i, (pickup, drop) in enumerate(zip(pickup_windows, drop_windows)):
            if i == pickup_outlier_idx:
                outlier_pickup_windows.append(pickup)
                outlier_drop_windows.append(drop)
            else:
                normal_pickup_windows.append(pickup)
                normal_drop_windows.append(drop)
        
        company_names.append(company_id)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    fig.suptitle(f'Time Window Comparison - {case_name}\nOutliers vs Normal Clients', 
                 fontsize=16, fontweight='bold')
    
    # Pickup windows comparison
    x_pos = np.arange(len(company_names))
    width = 0.35
    
    # Calculate averages for normal clients per company
    normal_pickup_avg = []
    outlier_pickup_values = []
    
    client_idx = 0
    for company_id, company_data in companies.items():
        clients = company_data['clients']
        pickup_windows = [client['late_pickup'] - client['early_pickup'] for client in clients]
        outlier_idx = np.argmin(pickup_windows)
        
        # Average of normal clients (excluding outlier)
        normal_windows = [w for i, w in enumerate(pickup_windows) if i != outlier_idx]
        normal_pickup_avg.append(np.mean(normal_windows))
        outlier_pickup_values.append(pickup_windows[outlier_idx])
    
    bars1 = ax1.bar(x_pos - width/2, normal_pickup_avg, width, label='Normal Clients (Avg)', 
                    color='lightblue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, outlier_pickup_values, width, label='Outliers', 
                    color='red', alpha=0.8)
    
    ax1.set_xlabel('Companies')
    ax1.set_ylabel('Pickup Window (minutes)')
    ax1.set_title('Pickup Windows: Outliers vs Normal')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(company_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom', fontweight='bold')
    
    # Drop windows comparison
    normal_drop_avg = []
    outlier_drop_values = []
    
    for company_id, company_data in companies.items():
        clients = company_data['clients']
        drop_windows = [client['late_drop_off'] - client['early_drop_off'] for client in clients]
        outlier_idx = np.argmin(drop_windows)
        
        # Average of normal clients (excluding outlier)
        normal_windows = [w for i, w in enumerate(drop_windows) if i != outlier_idx]
        normal_drop_avg.append(np.mean(normal_windows))
        outlier_drop_values.append(drop_windows[outlier_idx])
    
    bars3 = ax2.bar(x_pos - width/2, normal_drop_avg, width, label='Normal Clients (Avg)', 
                    color='lightgreen', alpha=0.8)
    bars4 = ax2.bar(x_pos + width/2, outlier_drop_values, width, label='Outliers', 
                    color='red', alpha=0.8)
    
    ax2.set_xlabel('Companies')
    ax2.set_ylabel('Drop Window (minutes)')
    ax2.set_title('Drop Windows: Outliers vs Normal')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(company_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom')
    
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'time_windows_comparison_{case_name}.png', dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved as: time_windows_comparison_{case_name}.png")
    
    plt.show()

if __name__ == "__main__":
    # Load the time window outlier data
    data_path = "data/2025_08_08_14_00_time_windows/company_cases.json"
    
    try:
        data = load_data(data_path)
        print(f"Loaded data with {len(data)} cases")
        
        # Generate detailed visualization
        print(f"\nGenerating detailed visualization for case_1...")
        visualize_time_windows_detailed(data, "case_1", save_plot=True)
        
        # Generate comparison chart
        print(f"\nGenerating comparison chart for case_1...")
        create_outlier_comparison_chart(data, "case_1", save_plot=True)
        
    except FileNotFoundError:
        print(f"Error: Could not find data file at {data_path}")
        print("Please run generate_time_window_outliers.py first to generate the data.")
    except Exception as e:
        print(f"Error: {e}") 