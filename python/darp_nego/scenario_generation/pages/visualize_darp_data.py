import streamlit as st
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Try to import Plotly, but make it optional
USE_PLOTLY = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    pio.templates.default = "plotly"
    USE_PLOTLY = True
except Exception as e:
    print(f"Plotly not available: {e}")
    USE_PLOTLY = False

st.set_page_config(page_title="DARP Data Visualizer", layout="wide")

def load_case_data(file_path):
    """Load case data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_available_data_folders():
    """Get list of available data folders"""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    folders = [f.name for f in data_dir.iterdir() if f.is_dir()]
    return sorted(folders, reverse=True)  # Most recent first

def analyze_case_statistics(case_data):
    """Extract comprehensive statistics from a single case"""
    companies = case_data['companies']
    time_matrix = case_data['time_matrix']
    coordinates = case_data['coordinates']
    metadata = case_data.get('metadata', {})
    
    stats = {
        'num_companies': len(companies),
        'num_vehicles': 0,
        'num_clients': 0,
        'vehicle_capacities': [],
        'client_volumes': [],
        'client_time_windows': [],
        'pickup_flexibilities': [],
        'dropoff_flexibilities': [],
        'costly_clients': 0,
        'company_client_counts': [],
        'company_vehicle_counts': [],
        'company_total_capacities': [],
        'company_total_demands': [],
        'coordinates': coordinates,
        'time_matrix': time_matrix
    }
    
    for company_id, company_data in companies.items():
        vehicles = company_data['vehicles']
        clients = company_data['clients']
        
        stats['num_vehicles'] += len(vehicles)
        stats['num_clients'] += len(clients)
        stats['company_client_counts'].append(len(clients))
        stats['company_vehicle_counts'].append(len(vehicles))
        
        # Vehicle stats
        company_capacity = sum(v['max_volume'] for v in vehicles)
        stats['company_total_capacities'].append(company_capacity)
        stats['vehicle_capacities'].extend([v['max_volume'] for v in vehicles])
        
        # Client stats
        company_demand = sum(c['volume'] for c in clients)
        stats['company_total_demands'].append(company_demand)
        stats['client_volumes'].extend([c['volume'] for c in clients])
        
        for client in clients:
            # Time window analysis
            pickup_flexibility = client['late_pickup'] - client['early_pickup']
            dropoff_flexibility = client['late_drop_off'] - client['early_drop_off']
            stats['pickup_flexibilities'].append(pickup_flexibility)
            stats['dropoff_flexibilities'].append(dropoff_flexibility)
            
            stats['client_time_windows'].append({
                'early_pickup': client['early_pickup'],
                'late_pickup': client['late_pickup'],
                'early_drop': client['early_drop_off'],
                'late_drop': client['late_drop_off']
            })
            
            # Detect costly clients (tight windows or high volume)
            if pickup_flexibility < 35 or dropoff_flexibility < 35:
                stats['costly_clients'] += 1
    
    # Add metadata
    stats['metadata'] = metadata
    
    return stats

def plot_spatial_distribution(case_data, case_name):
    """Plot geographic distribution of clients, vehicles, and hospitals using matplotlib"""
    companies = case_data['companies']
    coordinates = case_data['coordinates']
    metadata = case_data.get('metadata', {})
    
    num_vehicles = metadata.get('num_vehicles', 0)
    num_hospitals = metadata.get('num_hospitals', 0)
    num_clients = metadata.get('num_clients', 0)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract location data
    vehicle_locs = []
    hospital_locs = []
    client_locs_by_company = {f"company_{i}": [] for i in range(len(companies))}
    
    for i in range(num_vehicles):
        if str(i) in coordinates:
            coord = coordinates[str(i)]
            vehicle_locs.append({'x': coord[0], 'y': coord[1]})
    
    for i in range(num_vehicles, num_vehicles + num_hospitals):
        if str(i) in coordinates:
            coord = coordinates[str(i)]
            hospital_locs.append({'x': coord[0], 'y': coord[1]})
    
    for company_id, company_data in companies.items():
        for client in company_data['clients']:
            loc_id = str(client['start_location'])
            if loc_id in coordinates:
                coord = coordinates[loc_id]
                client_locs_by_company[company_id].append({
                    'x': coord[0], 
                    'y': coord[1],
                    'is_costly': client.get('is_costly', False)
                })
    
    # Plot with matplotlib
    if vehicle_locs:
        ax.scatter([v['x'] for v in vehicle_locs], [v['y'] for v in vehicle_locs], 
                  marker='s', s=200, c='blue', label='Vehicles', zorder=3, edgecolors='black', linewidths=1)
    
    if hospital_locs:
        ax.scatter([h['x'] for h in hospital_locs], [h['y'] for h in hospital_locs], 
                  marker='X', s=300, c='red', label='Hospitals', zorder=4, edgecolors='black', linewidths=2)
    
    colors_mpl = plt.cm.Set2(np.linspace(0, 1, max(len(companies), 8)))
    for idx, (company_id, clients) in enumerate(client_locs_by_company.items()):
        if clients:
            normal = [c for c in clients if not c.get('is_costly', False)]
            costly = [c for c in clients if c.get('is_costly', False)]
            
            if normal:
                ax.scatter([c['x'] for c in normal], [c['y'] for c in normal], 
                         s=100, c=[colors_mpl[idx]], label=f'{company_id}', alpha=0.7, zorder=2)
            
            if costly:
                ax.scatter([c['x'] for c in costly], [c['y'] for c in costly], 
                         marker='*', s=300, c=[colors_mpl[idx]], 
                         label=f'{company_id} (COSTLY)', edgecolors='black', linewidths=1.5, zorder=3)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Spatial Distribution - {case_name}', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    return fig

def plot_time_window_distribution(stats):
    """Plot distribution of time windows"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pickup flexibility
    axes[0, 0].hist(stats['pickup_flexibilities'], bins=20, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Pickup Flexibility (minutes)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Pickup Time Window Flexibility')
    axes[0, 0].axvline(np.mean(stats['pickup_flexibilities']), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(stats["pickup_flexibilities"]):.1f}')
    axes[0, 0].legend()
    
    # Dropoff flexibility
    axes[0, 1].hist(stats['dropoff_flexibilities'], bins=20, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Dropoff Flexibility (minutes)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Dropoff Time Window Flexibility')
    axes[0, 1].axvline(np.mean(stats['dropoff_flexibilities']), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(stats["dropoff_flexibilities"]):.1f}')
    axes[0, 1].legend()
    
    # Early pickup times
    early_pickups = [tw['early_pickup'] for tw in stats['client_time_windows']]
    axes[1, 0].hist(early_pickups, bins=20, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Early Pickup Time (minutes from start)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Early Pickup Times')
    axes[1, 0].axvline(np.mean(early_pickups), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(early_pickups):.1f}')
    axes[1, 0].legend()
    
    # Time window span (early pickup to late dropoff)
    time_spans = [tw['late_drop'] - tw['early_pickup'] for tw in stats['client_time_windows']]
    axes[1, 1].hist(time_spans, bins=20, color='plum', edgecolor='black')
    axes[1, 1].set_xlabel('Total Time Window Span (minutes)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Total Service Time Window Span')
    axes[1, 1].axvline(np.mean(time_spans), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(time_spans):.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    return fig

def plot_capacity_vs_demand(stats):
    """Plot capacity vs demand by company"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    companies = [f"Company {i}" for i in range(len(stats['company_client_counts']))]
    
    # Bar chart: capacity vs demand
    x = np.arange(len(companies))
    width = 0.35
    
    ax1.bar(x - width/2, stats['company_total_capacities'], width, 
            label='Total Capacity', color='steelblue')
    ax1.bar(x + width/2, stats['company_total_demands'], width, 
            label='Total Demand', color='coral')
    
    ax1.set_xlabel('Company')
    ax1.set_ylabel('Volume')
    ax1.set_title('Capacity vs Demand by Company')
    ax1.set_xticks(x)
    ax1.set_xticklabels(companies)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Utilization percentage
    utilization = [d/c * 100 if c > 0 else 0 
                   for d, c in zip(stats['company_total_demands'], 
                                  stats['company_total_capacities'])]
    
    colors_util = ['red' if u > 100 else 'orange' if u > 80 else 'green' for u in utilization]
    ax2.bar(companies, utilization, color=colors_util, edgecolor='black')
    ax2.axhline(100, color='red', linestyle='--', linewidth=2, label='100% capacity')
    ax2.axhline(80, color='orange', linestyle='--', linewidth=1, label='80% capacity')
    ax2.set_xlabel('Company')
    ax2.set_ylabel('Capacity Utilization (%)')
    ax2.set_title('Company Capacity Utilization')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_volume_distribution(stats):
    """Plot distribution of client volumes and vehicle capacities"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Client volumes
    axes[0].hist(stats['client_volumes'], bins=range(1, max(stats['client_volumes'])+2), 
                 color='lightblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Client Volume')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Client Volumes')
    axes[0].axvline(np.mean(stats['client_volumes']), color='red', 
                    linestyle='--', label=f'Mean: {np.mean(stats["client_volumes"]):.2f}')
    axes[0].legend()
    
    # Vehicle capacities
    axes[1].hist(stats['vehicle_capacities'], bins=range(min(stats['vehicle_capacities']), 
                                                         max(stats['vehicle_capacities'])+2), 
                 color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Vehicle Capacity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Vehicle Capacities')
    axes[1].axvline(np.mean(stats['vehicle_capacities']), color='red', 
                    linestyle='--', label=f'Mean: {np.mean(stats["vehicle_capacities"]):.2f}')
    axes[1].legend()
    
    plt.tight_layout()
    return fig

def plot_distance_matrix_heatmap(stats, max_display=30):
    """Plot heatmap of travel times between locations"""
    time_matrix = stats['time_matrix']
    size = min(len(time_matrix), max_display)
    
    # Create a subset if matrix is too large
    matrix_subset = [row[:size] for row in time_matrix[:size]]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(matrix_subset, cmap='YlOrRd', aspect='auto')
    
    ax.set_xlabel('To Location')
    ax.set_ylabel('From Location')
    ax.set_title(f'Travel Time Matrix (first {size}x{size} locations)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Travel Time (minutes)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig

def plot_resource_balance(stats):
    """Plot clients and vehicles per company"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    companies = [f"Company {i}" for i in range(len(stats['company_client_counts']))]
    
    # Clients per company
    ax1.bar(companies, stats['company_client_counts'], color='cornflowerblue', edgecolor='black')
    ax1.set_xlabel('Company')
    ax1.set_ylabel('Number of Clients')
    ax1.set_title('Clients per Company')
    ax1.axhline(np.mean(stats['company_client_counts']), color='red', 
                linestyle='--', label=f'Mean: {np.mean(stats["company_client_counts"]):.1f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Vehicles per company
    ax2.bar(companies, stats['company_vehicle_counts'], color='seagreen', edgecolor='black')
    ax2.set_xlabel('Company')
    ax2.set_ylabel('Number of Vehicles')
    ax2.set_title('Vehicles per Company')
    ax2.axhline(np.mean(stats['company_vehicle_counts']), color='red', 
                linestyle='--', label=f'Mean: {np.mean(stats["company_vehicle_counts"]):.1f}')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def validate_case(case_data, case_name):
    """Perform validation checks on case data"""
    validation_results = []
    
    companies = case_data['companies']
    time_matrix = case_data['time_matrix']
    
    # Check 1: Each company has at least one vehicle
    for company_id, company_data in companies.items():
        if len(company_data['vehicles']) == 0:
            validation_results.append({
                'status': '❌',
                'check': 'Vehicle Assignment',
                'message': f'{company_id} has no vehicles'
            })
        else:
            validation_results.append({
                'status': '✅',
                'check': 'Vehicle Assignment',
                'message': f'{company_id} has {len(company_data["vehicles"])} vehicle(s)'
            })
    
    # Check 2: Time window feasibility
    infeasible_clients = []
    for company_id, company_data in companies.items():
        for client in company_data['clients']:
            # Check ordering
            if not (client['early_pickup'] <= client['late_pickup'] <= 
                   client['early_drop_off'] <= client['late_drop_off']):
                infeasible_clients.append(f"{company_id}/Client_{client['client_id']}")
    
    if infeasible_clients:
        validation_results.append({
            'status': '❌',
            'check': 'Time Window Ordering',
            'message': f'Invalid time windows: {", ".join(infeasible_clients[:5])}'
        })
    else:
        validation_results.append({
            'status': '✅',
            'check': 'Time Window Ordering',
            'message': 'All time windows properly ordered'
        })
    
    # Check 3: Capacity feasibility
    over_capacity = []
    for company_id, company_data in companies.items():
        total_capacity = sum(v['max_volume'] for v in company_data['vehicles'])
        max_client_volume = max([c['volume'] for c in company_data['clients']], default=0)
        
        if max_client_volume > total_capacity:
            over_capacity.append(f"{company_id} (need {max_client_volume}, have {total_capacity})")
    
    if over_capacity:
        validation_results.append({
            'status': '⚠️',
            'check': 'Capacity Feasibility',
            'message': f'Potential issues: {", ".join(over_capacity)}'
        })
    else:
        validation_results.append({
            'status': '✅',
            'check': 'Capacity Feasibility',
            'message': 'All clients fit in available vehicles'
        })
    
    # Check 4: Travel time vs time window
    tight_time_windows = []
    for company_id, company_data in companies.items():
        for client in company_data['clients']:
            start_loc = client['start_location']
            end_loc = client['end_location']
            
            if start_loc < len(time_matrix) and end_loc < len(time_matrix):
                travel_time = time_matrix[start_loc][end_loc]
                available_time = client['early_drop_off'] - client['late_pickup']
                
                if travel_time > available_time:
                    tight_time_windows.append(
                        f"{company_id}/Client_{client['client_id']} (need {travel_time}min, have {available_time}min)"
                    )
    
    if tight_time_windows:
        validation_results.append({
            'status': '❌',
            'check': 'Travel Time Feasibility',
            'message': f'Insufficient time: {", ".join(tight_time_windows[:3])}'
        })
    else:
        validation_results.append({
            'status': '✅',
            'check': 'Travel Time Feasibility',
            'message': 'All clients have sufficient travel time'
        })
    
    return validation_results

# ============= STREAMLIT APP =============

st.title("🔍 DARP Data Visualizer & Validator")
st.markdown("### Analyze and validate generated DARP cases")

# Sidebar for file selection
st.sidebar.header("📁 Data Selection")

# Get available folders
folders = get_available_data_folders()

if not folders:
    st.error("No data folders found in 'data/' directory. Please generate some cases first.")
    st.stop()

selected_folder = st.sidebar.selectbox("Select Data Folder", folders)
data_path = Path("data") / selected_folder / "company_cases.json"

if not data_path.exists():
    st.error(f"No company_cases.json found in {selected_folder}")
    st.stop()

# Load data
with st.spinner("Loading data..."):
    all_cases = load_case_data(data_path)

st.sidebar.success(f"✅ Loaded {len(all_cases)} cases")

# Case selection
case_names = list(all_cases.keys())
selected_case = st.sidebar.selectbox("Select Case to Analyze", case_names)

case_data = all_cases[selected_case]

# ============= MAIN CONTENT =============

# Summary metrics
st.header("📊 Case Summary")
stats = analyze_case_statistics(case_data)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Companies", stats['num_companies'])
col2.metric("Vehicles", stats['num_vehicles'])
col3.metric("Clients", stats['num_clients'])
col4.metric("Costly Clients", f"{stats['costly_clients']} ({stats['costly_clients']/stats['num_clients']*100:.1f}%)")
col5.metric("Avg Capacity", f"{np.mean(stats['vehicle_capacities']):.1f}")

# Validation
st.header("✅ Validation Results")
validation_results = validate_case(case_data, selected_case)

validation_df = pd.DataFrame(validation_results)
st.dataframe(validation_df, use_container_width=True, hide_index=True)

# Tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Spatial", 
    "⏰ Time Windows", 
    "📦 Capacity", 
    "📊 Volumes",
    "🔥 Distance Matrix",
    "⚖️ Resource Balance"
])

with tab1:
    st.subheader("Geographic Distribution")
    fig_spatial = plot_spatial_distribution(case_data, selected_case)
    st.pyplot(fig_spatial)
    
    st.markdown("""
    **Legend:**
    - 🟦 **Squares**: Vehicle start locations
    - ❌ **X marks**: Hospital locations
    - ⭕ **Circles**: Normal clients (by company)
    - ⭐ **Stars**: Costly clients (tight time windows or high volume)
    """)

with tab2:
    st.subheader("Time Window Distribution")
    fig_time = plot_time_window_distribution(stats)
    st.pyplot(fig_time)
    
    st.markdown(f"""
    **Insights:**
    - Average pickup flexibility: **{np.mean(stats['pickup_flexibilities']):.1f} minutes**
    - Average dropoff flexibility: **{np.mean(stats['dropoff_flexibilities']):.1f} minutes**
    - Tight windows (< 35 min): **{sum(1 for f in stats['pickup_flexibilities'] if f < 35)} clients**
    """)

with tab3:
    st.subheader("Capacity vs Demand Analysis")
    fig_capacity = plot_capacity_vs_demand(stats)
    st.pyplot(fig_capacity)
    
    # Capacity warnings
    for i, (cap, dem) in enumerate(zip(stats['company_total_capacities'], 
                                       stats['company_total_demands'])):
        util = dem / cap * 100 if cap > 0 else 0
        if util > 100:
            st.error(f"⚠️ Company {i}: Demand exceeds capacity! ({util:.1f}%)")
        elif util > 80:
            st.warning(f"⚠️ Company {i}: High utilization ({util:.1f}%)")

with tab4:
    st.subheader("Volume Distribution")
    fig_volume = plot_volume_distribution(stats)
    st.pyplot(fig_volume)
    
    col1, col2 = st.columns(2)
    col1.markdown(f"""
    **Client Volumes:**
    - Min: {min(stats['client_volumes'])}
    - Max: {max(stats['client_volumes'])}
    - Mean: {np.mean(stats['client_volumes']):.2f}
    - Std: {np.std(stats['client_volumes']):.2f}
    """)
    
    col2.markdown(f"""
    **Vehicle Capacities:**
    - Min: {min(stats['vehicle_capacities'])}
    - Max: {max(stats['vehicle_capacities'])}
    - Mean: {np.mean(stats['vehicle_capacities']):.2f}
    - Std: {np.std(stats['vehicle_capacities']):.2f}
    """)

with tab5:
    st.subheader("Travel Time Matrix")
    max_display = st.slider("Matrix size to display", 10, 50, 30)
    fig_matrix = plot_distance_matrix_heatmap(stats, max_display)
    st.pyplot(fig_matrix)
    
    st.markdown(f"""
    **Travel Time Statistics (minutes):**
    - Average: {np.mean([t for row in stats['time_matrix'] for t in row if t > 0]):.1f}
    - Max: {max([t for row in stats['time_matrix'] for t in row]):.1f}
    """)

with tab6:
    st.subheader("Resource Distribution")
    fig_resource = plot_resource_balance(stats)
    st.pyplot(fig_resource)
    
    # Balance analysis
    client_std = np.std(stats['company_client_counts'])
    vehicle_std = np.std(stats['company_vehicle_counts'])
    
    if client_std < 2:
        st.success("✅ Clients are well-balanced across companies")
    else:
        st.warning(f"⚠️ Client distribution has high variance (σ={client_std:.2f})")
    
    if vehicle_std < 1:
        st.success("✅ Vehicles are well-balanced across companies")
    else:
        st.warning(f"⚠️ Vehicle distribution has high variance (σ={vehicle_std:.2f})")

# ============= AGGREGATE ANALYSIS =============

st.header("📈 Aggregate Analysis (All Cases)")

if st.checkbox("Show aggregate statistics across all cases"):
    all_stats = []
    
    progress_bar = st.progress(0)
    for idx, (case_name, case_data) in enumerate(all_cases.items()):
        stats = analyze_case_statistics(case_data)
        all_stats.append({
            'case': case_name,
            'num_companies': stats['num_companies'],
            'num_vehicles': stats['num_vehicles'],
            'num_clients': stats['num_clients'],
            'costly_clients_pct': stats['costly_clients'] / stats['num_clients'] * 100,
            'avg_pickup_flex': np.mean(stats['pickup_flexibilities']),
            'avg_dropoff_flex': np.mean(stats['dropoff_flexibilities']),
            'avg_capacity_util': np.mean([d/c*100 for d, c in 
                                         zip(stats['company_total_demands'], 
                                            stats['company_total_capacities']) if c > 0])
        })
        progress_bar.progress((idx + 1) / len(all_cases))
    
    df_stats = pd.DataFrame(all_stats)
    
    st.subheader("Summary Statistics")
    st.dataframe(df_stats.describe(), use_container_width=True)
    
    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(df_stats['num_clients'], bins=20, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Number of Clients')
    axes[0, 0].set_title('Distribution of Client Counts Across Cases')
    
    axes[0, 1].hist(df_stats['costly_clients_pct'], bins=20, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Costly Clients (%)')
    axes[0, 1].set_title('Distribution of Costly Client Percentage')
    
    axes[1, 0].hist(df_stats['avg_pickup_flex'], bins=20, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Avg Pickup Flexibility (min)')
    axes[1, 0].set_title('Average Pickup Flexibility Across Cases')
    
    axes[1, 1].hist(df_stats['avg_capacity_util'], bins=20, color='plum', edgecolor='black')
    axes[1, 1].set_xlabel('Avg Capacity Utilization (%)')
    axes[1, 1].set_title('Average Capacity Utilization Across Cases')
    
    plt.tight_layout()
    st.pyplot(fig)

# ============= EXPORT =============

st.sidebar.markdown("---")
st.sidebar.header("📥 Export")

if st.sidebar.button("Export Current Case Analysis"):
    export_data = {
        'case_name': selected_case,
        'statistics': {
            'num_companies': stats['num_companies'],
            'num_vehicles': stats['num_vehicles'],
            'num_clients': stats['num_clients'],
            'costly_clients': stats['costly_clients'],
            'avg_pickup_flexibility': float(np.mean(stats['pickup_flexibilities'])),
            'avg_dropoff_flexibility': float(np.mean(stats['dropoff_flexibilities'])),
        },
        'validation_results': validation_results
    }
    
    export_json = json.dumps(export_data, indent=4)
    st.sidebar.download_button(
        "Download Analysis JSON",
        export_json,
        file_name=f"analysis_{selected_case}.json",
        mime="application/json"
    )