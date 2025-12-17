import json
import matplotlib.pyplot as plt
import numpy as np
import random

def load_data(file_path):
    """Load the time window outlier data"""
    with open(file_path, 'r') as f:
        return json.load(f)

def visualize_time_windows(data, case_name="case_1", save_plot=True):
    """
    Visualize time windows for a specific case, highlighting outliers
    """
    case_data = data[case_name]
    companies = case_data['companies']
    
    # Set up the plot
    fig, axes = plt.subplots(len(companies), 1, figsize=(15, 4 * len(companies)))
    if len(companies) == 1:
        axes = [axes]
    
    fig.suptitle(f'Time Windows Visualization - {case_name}\n(Outliers have almost-overlapping windows)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for company_idx, (company_id, company_data) in enumerate(companies.items()):
        ax = axes[company_idx]
        clients = company_data['clients']
        
        # Calculate time window characteristics for outlier detection
        pickup_windows = []
        drop_windows = []
        total_service_times = []
        
        for client in clients:
            pickup_window = client['late_pickup'] - client['early_pickup']
            drop_window = client['late_drop_off'] - client['early_drop_off']
            total_service = client['late_drop_off'] - client['early_pickup']
            
            pickup_windows.append(pickup_window)
            drop_windows.append(drop_window)
            total_service_times.append(total_service)
        
        # Find the outlier (client with smallest time windows)
        outlier_idx = np.argmin(pickup_windows)
        
        # Plot each client's time windows
        for client_idx, client in enumerate(clients):
            # Determine color and style based on whether it's an outlier
            if client_idx == outlier_idx:
                color = 'red'
                alpha = 1.0
                linewidth = 3
                label = f"Client {client['client_id']} (OUTLIER)"
            else:
                color = colors[client_idx % len(colors)]
                alpha = 0.7
                linewidth = 2
                label = f"Client {client['client_id']}"
            
            # Plot pickup window
            ax.barh(f"Client {client['client_id']}", 
                   client['late_pickup'] - client['early_pickup'],
                   left=client['early_pickup'],
                   height=0.6, color=color, alpha=alpha, 
                   linewidth=linewidth, edgecolor='black',
                   label=f"Pickup: {client['early_pickup']}-{client['late_pickup']} ({client['late_pickup'] - client['early_pickup']} min)")
            
            # Plot drop-off window
            ax.barh(f"Client {client['client_id']}", 
                   client['late_drop_off'] - client['early_drop_off'],
                   left=client['early_drop_off'],
                   height=0.6, color=color, alpha=alpha*0.8,
                   linewidth=linewidth, edgecolor='black',
                   hatch='//', label=f"Drop: {client['early_drop_off']}-{client['late_drop_off']} ({client['late_drop_off'] - client['early_drop_off']} min)")
        
        # Customize the subplot
        ax.set_title(f'{company_id} - Time Windows', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Clients')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add statistics
        avg_pickup = np.mean(pickup_windows)
        avg_drop = np.mean(drop_windows)
        outlier_pickup = pickup_windows[outlier_idx]
        outlier_drop = drop_windows[outlier_idx]
        
        stats_text = f"Avg Pickup Window: {avg_pickup:.1f} min\n"
        stats_text += f"Avg Drop Window: {avg_drop:.1f} min\n"
        stats_text += f"Outlier Pickup: {outlier_pickup} min\n"
        stats_text += f"Outlier Drop: {outlier_drop} min"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'time_windows_{case_name}.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as: time_windows_{case_name}.png")
    
    plt.show()

def visualize_all_cases(data, max_cases=3):
    """
    Visualize time windows for multiple cases
    """
    cases = list(data.keys())[:max_cases]
    
    for case_name in cases:
        print(f"\nVisualizing {case_name}...")
        visualize_time_windows(data, case_name, save_plot=True)

def analyze_time_window_statistics(data):
    """
    Analyze and print statistics about time window outliers
    """
    print("=== TIME WINDOW OUTLIER ANALYSIS ===\n")
    
    for case_name, case_data in data.items():
        print(f"\n{case_name}:")
        companies = case_data['companies']
        
        for company_id, company_data in companies.items():
            clients = company_data['clients']
            
            # Calculate time window characteristics
            pickup_windows = []
            drop_windows = []
            
            for client in clients:
                pickup_window = client['late_pickup'] - client['early_pickup']
                drop_window = client['late_drop_off'] - client['early_drop_off']
                
                pickup_windows.append(pickup_window)
                drop_windows.append(drop_window)
            
            # Find outliers
            min_pickup_idx = np.argmin(pickup_windows)
            min_drop_idx = np.argmin(drop_windows)
            
            avg_pickup = np.mean(pickup_windows)
            avg_drop = np.mean(drop_windows)
            
            print(f"  {company_id}:")
            print(f"    Average pickup window: {avg_pickup:.1f} minutes")
            print(f"    Average drop window: {avg_drop:.1f} minutes")
            print(f"    Smallest pickup window: {pickup_windows[min_pickup_idx]} minutes (Client {clients[min_pickup_idx]['client_id']})")
            print(f"    Smallest drop window: {drop_windows[min_drop_idx]} minutes (Client {clients[min_drop_idx]['client_id']})")
            
            # Check if outliers are significantly different
            if pickup_windows[min_pickup_idx] < avg_pickup * 0.3:
                print(f"    ✓ Clear pickup window outlier detected!")
            if drop_windows[min_drop_idx] < avg_drop * 0.3:
                print(f"    ✓ Clear drop window outlier detected!")

if __name__ == "__main__":
    # Load the time window outlier data
    data_path = "data/2025_08_08_14_00_time_windows/company_cases.json"
    
    try:
        data = load_data(data_path)
        print(f"Loaded data with {len(data)} cases")
        
        # Analyze statistics
        analyze_time_window_statistics(data)
        
        # Visualize first case
        print(f"\nGenerating visualization for case_1...")
        visualize_time_windows(data, "case_1", save_plot=True)
        
        # Optionally visualize more cases
        # visualize_all_cases(data, max_cases=3)
        
    except FileNotFoundError:
        print(f"Error: Could not find data file at {data_path}")
        print("Please run generate_time_window_outliers.py first to generate the data.")
    except Exception as e:
        print(f"Error: {e}") 