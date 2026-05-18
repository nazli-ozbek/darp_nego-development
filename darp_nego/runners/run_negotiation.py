from datetime import datetime
import os, json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import pandas as pd
import hashlib

from darp_nego.core.darp import (
    BasicDARPVehicle,
    BasicDARPClient,
    BasicDARPProblem,
    BasicDARPNegotiationProblem,
)
from darp_nego.core.negotiator import SingleTextBasicNegotiator
from darp_nego.protocols.heuristic.classic_single_mediated_text import ClassicSingleMediatedTextMechanism
from darp_nego.protocols.heuristic.classic_partial_single_mediated_text import ClassicPartialSingleMediatedTextMechanism
from darp_nego.protocols.learning.single_mediated_text import SingleMediatedTextMechanism
from darp_nego.runners.config import (
    DEBUG_SINGLE_SCENARIO,
    RUN_STRATEGY,
    FAIR_COMPARISON_MODE,
    FAIR_COMPARISON_SEEDS,
    ORTOOLS_NUM_SEARCH_WORKERS,
    USE_LATEST_SCENARIO,
    SCENARIO_JSON_PATH,
)
from metrics.negotiation_metrics import DARPNegotiationMetrics
from metrics.darp_routing_metrics import DARPRoutingMetrics

def generate_time_matrix(size, min_time=1, max_time=10, seed=None):
    """Generate a random time matrix for the road network."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create a matrix of random travel times
    matrix = np.random.randint(min_time, max_time, size=(size, size))
    
    # Ensure the diagonal is 0 (no travel time to same location)
    np.fill_diagonal(matrix, 0)
    
    return matrix.tolist()

def load_problems_from_json(json_path):
    """Load DARP problems from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    all_cases = {}

    for case_name, case_data in data.items():
        case_problems = {}
        time_matrix = case_data["time_matrix"]
        coordinates = case_data["coordinates"]

        for company_name, company_data in case_data["companies"].items():
            company_id = int(company_name.split("_")[1])

            problem = BasicDARPProblem(company_id)

            # Add vehicles
            for v in company_data.get("vehicles", []):
                vehicle = BasicDARPVehicle(
                    vehicle_id=v["vehicle_id"],
                    start_location=v["start_location"],
                    end_location=v["end_location"],
                    max_volume=v["max_volume"]
                )
                problem.add_vehicle(vehicle)

            # Add clients
            for c in company_data.get("clients", []):
                client = BasicDARPClient(
                    client_id=c["client_id"],
                    start_location=c["start_location"],
                    end_location=c["end_location"],
                    early_pickup=c["early_pickup"],
                    late_pickup=c["late_pickup"],
                    early_drop=c["early_drop_off"],
                    late_drop=c["late_drop_off"],
                    volume=c["volume"],
                    start_coordinates=coordinates[str(c["start_location"])],
                    end_coordinates=coordinates[str(c["end_location"])]
                )
                problem.add_client(client)

            # Set road network
            problem.road_network = time_matrix
            problem.coordinates = coordinates
            case_problems[company_id] = problem

        all_cases[case_name] = case_problems

    return all_cases

def find_latest_case_file(base_folder="scenario_generation/data/"):
    """Find the latest generated company_cases.json file.

    Resolves the data directory relative to this file so that it works
    regardless of the current working directory.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    resolved_base = os.path.join(repo_root, base_folder)

    if not os.path.isdir(resolved_base):
        raise FileNotFoundError(f"Data folder not found: {resolved_base}")

    # List only scenario folders that directly contain company_cases.json.
    subdirs = [
        d for d in os.listdir(resolved_base)
        if os.path.isdir(os.path.join(resolved_base, d))
        and os.path.isfile(os.path.join(resolved_base, d, "company_cases.json"))
    ]

    # Sort by datetime in folder name
    subdirs_sorted = sorted(subdirs, reverse=True)

    if not subdirs_sorted:
        raise FileNotFoundError(f"No scenario folders with company_cases.json found under {resolved_base}")

    latest_folder = subdirs_sorted[0]
    latest_path = os.path.join(resolved_base, latest_folder, "company_cases.json")

    if not os.path.isfile(latest_path):
        raise FileNotFoundError(f"'company_cases.json' not found at {latest_path}")
    return latest_path, latest_folder


def resolve_scenario_file():
    if USE_LATEST_SCENARIO:
        return find_latest_case_file()

    latest_case_file = SCENARIO_JSON_PATH
    if not os.path.isabs(latest_case_file):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        latest_case_file = os.path.join(repo_root, latest_case_file)
    latest_folder = os.path.splitext(os.path.basename(SCENARIO_JSON_PATH))[0]
    return latest_case_file, latest_folder


def main(strategy="heuristic", run_seed=42):
    latest_case_file, latest_folder = resolve_scenario_file()
    print(latest_case_file)
    problems_per_case = load_problems_from_json(latest_case_file)
    for case_name, case_problems in problems_per_case.items():
        print(f"\n=== Solving problems for {case_name} ===")

        time_matrix = next(iter(case_problems.values())).road_network

        company_ids = list(case_problems.keys())

        agents = list(case_problems.values())

        negotiation_problem = BasicDARPNegotiationProblem(
            problem_id=int(case_name.split("_")[1]),  # Use numeric part of case name as ID
            road_network=time_matrix,
            agents=agents
        )

        # Setup logging and metrics directories
        log_dir = "logs"
        metrics_dir = "metrics"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Generate a unique session ID for this negotiation run
        session_id = f"{latest_folder}_{case_name}_{strategy}_seed{run_seed}_{datetime.now().strftime('%H%M%S')}"
        print(f"Starting negotiation with session ID: {session_id}")
        
        print("\nPreparing pre-negotiation phase...\n")
        negotiators = []
        for company_id in company_ids:
            negotiator = SingleTextBasicNegotiator(
                agent_id=str(company_id),
                problem=case_problems[company_id]
            )
            negotiators.append(negotiator)
            print(f"Created negotiator for company {company_id} with {len(case_problems[company_id].clients)} clients\n")

        max_round = 200 if strategy == "learning" else 150

        # Select strategy from a single entrypoint:
        # heuristic -> classic mechanism, learning -> learning-enabled mechanism
        if strategy == "learning":
            mechanism = SingleMediatedTextMechanism(
                agents=negotiators,
                max_rounds=max_round,
                log_dir=log_dir,
                session_id=session_id,
                road_network=time_matrix
            )
        elif strategy == "heuristic_partial":
            mechanism = ClassicPartialSingleMediatedTextMechanism(
                agents=negotiators,
                max_rounds=max_round,
                log_dir=log_dir,
                session_id=session_id,
                road_network=time_matrix
            )
        else:
            mechanism = ClassicSingleMediatedTextMechanism(
                agents=negotiators,
                max_rounds=max_round,
                log_dir=log_dir,
                session_id=session_id,
                road_network=time_matrix
            )
        
        print("\nExecuting pre-negotiation phase...")
        mechanism.prenegotiation()
        
        print("\n=== Starting Negotiation ===")
        print("\nExecuting negotiation phase...")
        log_paths = mechanism.negotiation()
        if strategy == "learning":
            total_swaps = mechanism._total_buffer_size() if hasattr(mechanism, "_total_buffer_size") else 0
            print(f"Case {case_name} completed. Swap dataset size: {total_swaps}")
        json_log_path, txt_log_path = log_paths

        print("\n=== Analyzing Negotiation Results ===")
        metrics = DARPNegotiationMetrics(json_log_path)
        summary = metrics.calculate_metrics()
        print("\nNegotiation Metrics Summary:")
        print(summary)
        
        # Save metrics with the same session ID
        metrics_paths = metrics.save_metrics(output_dir=metrics_dir, run_id=session_id)
        
        # Calculate and save routing metrics
        routing_log_path = os.path.join(log_dir, session_id, "routing", f"routing_{session_id}.json")
        if os.path.exists(routing_log_path):
            print("\n=== Analyzing Routing Results ===")
            routing_metrics = DARPRoutingMetrics(routing_log_path)
            routing_summary = routing_metrics.calculate_metrics()
            print("\nRouting Metrics Summary:")
            for agent_id, phases in routing_summary.items():
                if "final" in phases:
                    print(f"\nAgent {agent_id} final metrics:")
                    print(phases["final"])
            
            # Save routing metrics with the same session ID
            routing_metrics_paths = routing_metrics.save_metrics(output_dir=metrics_dir, run_id=session_id)
        else:
            print(f"\nWarning: Routing log not found at expected path: {routing_log_path}")
        
        print(f"\nSession ID: {session_id}")
        print(f"Log files: {json_log_path}")
        print(f"Metrics: {metrics_paths['directory']}")

def debug_single_scenario():
        
    """
    BASE_DIR = "data/companies"

    scenario_folders = sorted([f for f in os.listdir(BASE_DIR) if f.isdigit()], key=int)

    for scenario in scenario_folders:
        scenario_path = os.path.join(BASE_DIR, scenario)

        clients_file = os.path.join(scenario_path, "clients.json")
        vehicles_file = os.path.join(scenario_path, "vehicles.json")

        if not os.path.exists(clients_file) or not os.path.exists(vehicles_file):
            print(f"Skipping scenario {scenario}: Missing clients.json or vehicles.json")
            continue

        with open(vehicles_file, "r") as f:
            vehicles_data = json.load(f)

        with open(clients_file, "r") as f:
            clients_data = json.load(f)

        vehicles = [BasicDARPVehicle.from_json(vehicle) for vehicle in vehicles_data]
        clients = [BasicDARPClient.from_json(client) for client in clients_data]

        problem = BasicDARPProblem(len(vehicles), vehicles=vehicles, clients=clients)
        network = {} #TODO
        # network = GISRoadNetwork()
        solution_cost = problem.solve_problem(road_network=network, config=None)

        if solution_cost is None:
            print(f"⚠️  Scenario {scenario} is infeasible!")
        else:
            print(f"✅ Scenario {scenario}: Solution Cost = {solution_cost}") 
    """

    # Set seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    
    # Create road network (time matrix)
    matrix_size = 15  # Ensure it's large enough for all location IDs
    time_matrix = generate_time_matrix(size=matrix_size, seed=random_seed)
    
    # ===== Company 1 =====
    vehicle1 = BasicDARPVehicle(0, start_location=0, end_location=0, max_volume=10)
    vehicle2 = BasicDARPVehicle(1, start_location=0, end_location=1, max_volume=5)
    vehicle3 = BasicDARPVehicle(2, start_location=1, end_location=1, max_volume=15)
    
    client1 = BasicDARPClient(0, start_location=7, end_location=5, early_pickup=150, late_pickup=300, early_drop=200, late_drop=800, volume=2)
    client2 = BasicDARPClient(1, start_location=10, end_location=4, early_pickup=10, late_pickup=400, early_drop=300, late_drop=760, volume=1)
    client3 = BasicDARPClient(2, start_location=9, end_location=3, early_pickup=2, late_pickup=10, early_drop=3, late_drop=10, volume=4)
    
    problem1 = BasicDARPProblem(0)
    problem1.add_vehicle(vehicle1)
    problem1.add_vehicle(vehicle2)
    problem1.add_vehicle(vehicle3)
    problem1.add_client(client1)
    problem1.add_client(client2)
    problem1.add_client(client3)
    problem1.road_network = time_matrix
    
    # ===== Company 2 =====
    vehicle4 = BasicDARPVehicle(3, start_location=2, end_location=2, max_volume=8)
    vehicle5 = BasicDARPVehicle(4, start_location=3, end_location=3, max_volume=6)
    
    client4 = BasicDARPClient(3, start_location=8, end_location=4, early_pickup=20, late_pickup=200, early_drop=30, late_drop=600, volume=2)
    client5 = BasicDARPClient(4, start_location=6, end_location=5, early_pickup=300, late_pickup=890, early_drop=670, late_drop=900, volume=1)
    client6 = BasicDARPClient(5, start_location=11, end_location=6, early_pickup=50, late_pickup=350, early_drop=100, late_drop=750, volume=3)
    
    problem2 = BasicDARPProblem(1)
    problem2.add_vehicle(vehicle4)
    problem2.add_vehicle(vehicle5)
    problem2.add_client(client4)
    problem2.add_client(client5)
    problem2.add_client(client6)
    problem2.road_network = time_matrix
    
    # ===== Company 3 =====

    vehicle6 = BasicDARPVehicle(5, start_location=4, end_location=4, max_volume=12)
    vehicle7 = BasicDARPVehicle(6, start_location=5, end_location=5, max_volume=7)
    vehicle8 = BasicDARPVehicle(7, start_location=3, end_location=4, max_volume=9)
    
    client7 = BasicDARPClient(6, start_location=12, end_location=7, early_pickup=30, late_pickup=250, early_drop=100, late_drop=550, volume=2)
    client8 = BasicDARPClient(7, start_location=5, end_location=13, early_pickup=150, late_pickup=450, early_drop=350, late_drop=700, volume=3)
    client9 = BasicDARPClient(8, start_location=14, end_location=8, early_pickup=200, late_pickup=500, early_drop=400, late_drop=850, volume=1)
    client10 = BasicDARPClient(9, start_location=4, end_location=9, early_pickup=180, late_pickup=370, early_drop=280, late_drop=720, volume=2)
    
    problem3 = BasicDARPProblem(2)
    problem3.add_vehicle(vehicle6)
    problem3.add_vehicle(vehicle7)
    problem3.add_vehicle(vehicle8)
    problem3.add_client(client7)
    problem3.add_client(client8)
    problem3.add_client(client9)
    problem3.add_client(client10)
    problem3.road_network = time_matrix
    
    # ===== Company 4 =====
    vehicle9 = BasicDARPVehicle(8, start_location=6, end_location=6, max_volume=10)
    vehicle10 = BasicDARPVehicle(9, start_location=7, end_location=7, max_volume=8)
    vehicle11 = BasicDARPVehicle(10, start_location=6, end_location=7, max_volume=11)
    
    client11 = BasicDARPClient(10, start_location=3, end_location=10, early_pickup=40, late_pickup=280, early_drop=120, late_drop=580, volume=2)
    client12 = BasicDARPClient(11, start_location=7, end_location=11, early_pickup=100, late_pickup=420, early_drop=250, late_drop=680, volume=3)
    client13 = BasicDARPClient(12, start_location=8, end_location=12, early_pickup=175, late_pickup=460, early_drop=300, late_drop=800, volume=1)
    client14 = BasicDARPClient(13, start_location=9, end_location=2, early_pickup=120, late_pickup=340, early_drop=220, late_drop=600, volume=2)
    
    problem4 = BasicDARPProblem(3)
    problem4.add_vehicle(vehicle9)
    problem4.add_vehicle(vehicle10)
    problem4.add_vehicle(vehicle11)
    problem4.add_client(client11)
    problem4.add_client(client12)
    problem4.add_client(client13)
    problem4.add_client(client14)
    problem4.road_network = time_matrix

    
    # ===== Company 5 =====
    vehicle12 = BasicDARPVehicle(11, start_location=8, end_location=8, max_volume=10)
    client15 = BasicDARPClient(14, start_location=11, end_location=11, early_pickup=100, late_pickup=400, early_drop=200, late_drop=600, volume=2)
    client16 = BasicDARPClient(15, start_location=12, end_location=12, early_pickup=150, late_pickup=450, early_drop=300, late_drop=700, volume=1)
    client17 = BasicDARPClient(16, start_location=13, end_location=13, early_pickup=200, late_pickup=500, early_drop=400, late_drop=800, volume=3)
    
    problem5 = BasicDARPProblem(4)
    problem5.add_vehicle(vehicle12)
    problem5.add_client(client15)
    problem5.add_client(client16)
    problem5.add_client(client17)
    problem5.road_network = time_matrix
    

    # Create the global problem
    company_ids = [0, 1, 2, 3]#, 4]       
    problems = {0: problem1, 1: problem2, 2: problem3, 3: problem4}#, 4: problem5}
    
    # Set road network for all problems
    for company_id in company_ids:
        problems[company_id].road_network = time_matrix
    
    # Create a negotiation problem
    negotiation_problem = BasicDARPNegotiationProblem(
        problem_id=0,
        road_network=time_matrix,
        agents=[problem1, problem2, problem3, problem4, problem5]
    )
    
    # Setup logging and metrics directories
    log_dir = "logs"
    metrics_dir = "metrics"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Generate a unique session ID for this negotiation run
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting negotiation with session ID: {session_id}")
    
    # Create negotiators
    negotiators = []
    for company_id in company_ids:
        negotiator = SingleTextBasicNegotiator(
            agent_id=str(company_id),
            problem=problems[company_id]
        )
        negotiators.append(negotiator)
        print(f"Created negotiator for company {company_id} with {len(problems[company_id].clients)} clients")
    
    # max_round = math.comb(len(company_ids), 2)
    max_round = 150

    # Create the mediation mechanism with the session ID
    mechanism = ClassicSingleMediatedTextMechanism(
        agents=negotiators,
        max_rounds=max_round,
        log_dir=log_dir,
        session_id=session_id,  # Pass session_id to mechanism
        road_network=time_matrix
    )
    
    print("\n=== Starting Negotiation ===")
    print("Executing pre-negotiation phase...")
    mechanism.prenegotiation()
    
    # Execute negotiation rounds
    print("\nExecuting negotiation phase...")
    log_paths = mechanism.negotiation()
    json_log_path, txt_log_path = log_paths
    
    # Calculate and display metri- From the example cs
    print("\n=== Analyzing Negotiation Results ===")
    metrics = DARPNegotiationMetrics(json_log_path)
    summary = metrics.calculate_metrics()
    print("\nNegotiation Metrics Summary:")
    print(summary)
    
    # Save metrics with the same session ID
    metrics_paths = metrics.save_metrics(output_dir=metrics_dir, run_id=session_id)
    
    # Calculate and save routing metrics
    routing_log_path = os.path.join(log_dir, session_id, "routing", f"routing_{session_id}.json")
    if os.path.exists(routing_log_path):
        print("\n=== Analyzing Routing Results ===")
        routing_metrics = DARPRoutingMetrics(routing_log_path)
        routing_summary = routing_metrics.calculate_metrics()
        print("\nRouting Metrics Summary:")
        for agent_id, phases in routing_summary.items():
            if "final" in phases:
                print(f"\nAgent {agent_id} final metrics:")
                print(phases["final"])
        
        # Save routing metrics with the same session ID
        routing_metrics_paths = routing_metrics.save_metrics(output_dir=metrics_dir, run_id=session_id)
    else:
        print(f"\nWarning: Routing log not found at expected path: {routing_log_path}")
    
    print(f"\nSession ID: {session_id}")
    print(f"Log files: {json_log_path}")
    print(f"Metrics: {metrics_paths['directory']}")
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

def analyze_negotiation_logs(session_pattern="2025_08_02_17_40_case_*"):
    """
    Analyze negotiation logs for a specific session pattern and generate statistics with visualizations.
    
    Args:
        session_pattern (str): Pattern to match session folders (e.g., "2025_08_02_17_40_case_*")
    
    Returns:
        dict: Statistics summary
    """
    log_dir = "logs"
    stats_data = []
    
    # Find all matching session folders
    session_folders = glob.glob(os.path.join(log_dir, session_pattern))
    session_folders.sort()
    
    print(f"Found {len(session_folders)} session folders matching pattern: {session_pattern}")
    
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
            
            # Extract statistics
            prenegotiation = log_data.get("prenegotiation", {})
            final_state = log_data.get("final_state", {})
            
            # Initial utilities
            initial_utilities = prenegotiation.get("initial_utilities", {})
            initial_total_cost = sum(initial_utilities.values())
            
            # Final utilities
            final_utilities = final_state.get("final_utilities", {})
            final_total_cost = sum(final_utilities.values())
            
            # Agent information
            participants = prenegotiation.get("participants", [])
            num_agents = len(participants)
            
            # Negotiation results
            agreement_reached = final_state.get("agreement_reached", False)
            total_rounds = final_state.get("total_rounds", 0)
            execution_time = final_state.get("execution_time", 0)
            
            # Utility changes
            utility_changes = final_state.get("utility_changes", {})
            avg_utility_change = final_state.get("avg_utility_change", 0)
            
            # Client information
            revealed_clients = prenegotiation.get("revealed_clients", {})
            total_revealed_clients = sum(len(clients) for clients in revealed_clients.values())
            
            # Store data
            stats_data.append({
                'case_number': case_number,
                'session_name': session_name,
                'num_agents': num_agents,
                'initial_total_cost': initial_total_cost,
                'final_total_cost': final_total_cost,
                'cost_change': final_total_cost - initial_total_cost,
                'cost_change_percentage': ((final_total_cost - initial_total_cost) / initial_total_cost * 100) if initial_total_cost > 0 else 0,
                'agreement_reached': agreement_reached,
                'total_rounds': total_rounds,
                'execution_time': execution_time,
                'avg_utility_change': avg_utility_change,
                'total_revealed_clients': total_revealed_clients,
                'initial_utilities': initial_utilities,
                'final_utilities': final_utilities,
                'utility_changes': utility_changes
            })
            
            print(f"Processed {session_name}: {num_agents} agents, "
                  f"Initial cost: {initial_total_cost}, Final cost: {final_total_cost}, "
                  f"Agreement: {agreement_reached}, Rounds: {total_rounds}")
            
        except Exception as e:
            print(f"Error processing {session_name}: {e}")
            continue
    
    if not stats_data:
        print("No valid log data found!")
        return None
    
    # Create DataFrame for analysis
    df = pd.DataFrame(stats_data)
    
    # Generate summary statistics
    summary_stats = {
        'total_cases': len(df),
        'cases_with_agreement': len(df[df['agreement_reached'] == True]),
        'agreement_rate': len(df[df['agreement_reached'] == True]) / len(df) * 100,
        'avg_agents_per_case': df['num_agents'].mean(),
        'avg_initial_cost': df['initial_total_cost'].mean(),
        'avg_final_cost': df['final_total_cost'].mean(),
        'avg_cost_change': df['cost_change'].mean(),
        'avg_cost_change_percentage': df['cost_change_percentage'].mean(),
        'avg_rounds': df['total_rounds'].mean(),
        'avg_execution_time': df['execution_time'].mean(),
        'avg_utility_change': df['avg_utility_change'].mean(),
        'avg_revealed_clients': df['total_revealed_clients'].mean()
    }
    
    # Print summary
    print("\n" + "="*60)
    print("NEGOTIATION LOGS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total cases analyzed: {summary_stats['total_cases']}")
    print(f"Cases with agreement: {summary_stats['cases_with_agreement']} ({summary_stats['agreement_rate']:.1f}%)")
    print(f"Average agents per case: {summary_stats['avg_agents_per_case']:.1f}")
    print(f"Average initial cost: {summary_stats['avg_initial_cost']:.1f}")
    print(f"Average final cost: {summary_stats['avg_final_cost']:.1f}")
    print(f"Average cost change: {summary_stats['avg_cost_change']:.1f} ({summary_stats['avg_cost_change_percentage']:.1f}%)")
    print(f"Average rounds: {summary_stats['avg_rounds']:.1f}")
    print(f"Average execution time: {summary_stats['avg_execution_time']:.1f} seconds")
    print(f"Average utility change: {summary_stats['avg_utility_change']:.1f}")
    print(f"Average revealed clients: {summary_stats['avg_revealed_clients']:.1f}")
    
    # Create output directory for visualizations
    output_dir = "negotiation_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. AGREEMENT SUCCESS RATE PIE CHART
    plt.figure(figsize=(10, 6))
    agreement_counts = df['agreement_reached'].value_counts()
    colors = ['#ff6b6b', '#51cf66'] if len(agreement_counts) == 2 else ['#ff6b6b', '#51cf66', '#868e96']
    plt.pie(agreement_counts.values, labels=['No Agreement' if not x else 'Agreement Reached' for x in agreement_counts.index], 
            autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Negotiation Agreement Success Rate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agreement_success_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. COST COMPARISON BAR CHART
    plt.figure(figsize=(12, 6))
    x = range(len(df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], df['initial_total_cost'], width, label='Initial Cost', 
            color='#339af0', alpha=0.8)
    plt.bar([i + width/2 for i in x], df['final_total_cost'], width, label='Final Cost', 
            color='#51cf66', alpha=0.8)
    
    plt.xlabel('Case Number')
    plt.ylabel('Total Cost')
    plt.title('Initial vs Final Total Costs by Case', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xticks(x, df['case_number'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. COST CHANGE PERCENTAGE SCATTER PLOT
    plt.figure(figsize=(10, 6))
    colors = ['#51cf66' if x else '#ff6b6b' for x in df['agreement_reached']]
    plt.scatter(df['num_agents'], df['cost_change_percentage'], c=colors, alpha=0.7, s=100)
    plt.xlabel('Number of Agents')
    plt.ylabel('Cost Change Percentage (%)')
    plt.title('Cost Change vs Number of Agents', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#51cf66', label='Agreement Reached'),
                      Patch(facecolor='#ff6b6b', label='No Agreement')]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_change_vs_agents.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. NEGOTIATION ROUNDS ANALYSIS
    plt.figure(figsize=(10, 6))
    plt.hist(df['total_rounds'], bins=range(0, max(df['total_rounds'])+2, 1), 
             alpha=0.7, color='#339af0', edgecolor='black')
    plt.xlabel('Number of Rounds')
    plt.ylabel('Frequency')
    plt.title('Distribution of Negotiation Rounds', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'negotiation_rounds_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. EXECUTION TIME ANALYSIS
    plt.figure(figsize=(10, 6))
    plt.scatter(df['num_agents'], df['execution_time'], c=df['agreement_reached'].map({True: '#51cf66', False: '#ff6b6b'}), 
                alpha=0.7, s=100)
    plt.xlabel('Number of Agents')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Number of Agents', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [Patch(facecolor='#51cf66', label='Agreement Reached'),
                      Patch(facecolor='#ff6b6b', label='No Agreement')]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_vs_agents.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. AGENT COUNT BREAKDOWN
    agent_counts = df['num_agents'].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    bars = plt.bar(agent_counts.index, agent_counts.values, color='#339af0', alpha=0.8)
    plt.xlabel('Number of Agents')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of Cases by Agent Count', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_count_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. COMPREHENSIVE SUMMARY TABLE
    plt.figure(figsize=(14, 8))
    plt.axis('tight')
    plt.axis('off')
    
    # Create summary table data
    table_data = [
        ['Metric', 'Value'],
        ['Total Cases', f"{summary_stats['total_cases']}"],
        ['Cases with Agreement', f"{summary_stats['cases_with_agreement']} ({summary_stats['agreement_rate']:.1f}%)"],
        ['Average Agents per Case', f"{summary_stats['avg_agents_per_case']:.1f}"],
        ['Average Initial Cost', f"{summary_stats['avg_initial_cost']:.1f}"],
        ['Average Final Cost', f"{summary_stats['avg_final_cost']:.1f}"],
        ['Average Cost Change', f"{summary_stats['avg_cost_change']:.1f} ({summary_stats['avg_cost_change_percentage']:.1f}%)"],
        ['Average Rounds', f"{summary_stats['avg_rounds']:.1f}"],
        ['Average Execution Time', f"{summary_stats['avg_execution_time']:.1f} seconds"],
        ['Average Utility Change', f"{summary_stats['avg_utility_change']:.1f}"],
        ['Average Revealed Clients', f"{summary_stats['avg_revealed_clients']:.1f}"]
    ]
    
    table = plt.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='left', loc='center', colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(2):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#339af0')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                table[(i, j)].set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    plt.title('Negotiation Analysis Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. DETAILED BREAKDOWN BY AGENT COUNT
    agent_breakdown = df.groupby('num_agents').agg({
        'agreement_reached': ['count', 'sum', 'mean'],
        'initial_total_cost': 'mean',
        'final_total_cost': 'mean',
        'cost_change_percentage': 'mean',
        'total_rounds': 'mean',
        'execution_time': 'mean'
    }).round(2)
    
    # Flatten column names
    agent_breakdown.columns = ['_'.join(col).strip() for col in agent_breakdown.columns]
    agent_breakdown = agent_breakdown.reset_index()
    
    # Create table for agent breakdown
    plt.figure(figsize=(16, 10))
    plt.axis('tight')
    plt.axis('off')
    
    # Prepare table data
    breakdown_data = [['Agents', 'Cases', 'Agreements', 'Success Rate', 'Avg Initial Cost', 
                       'Avg Final Cost', 'Avg Cost Change %', 'Avg Rounds', 'Avg Time (s)']]
    
    for _, row in agent_breakdown.iterrows():
        breakdown_data.append([
            f"{row['num_agents']}",
            f"{row['agreement_reached_count']}",
            f"{row['agreement_reached_sum']}",
            f"{row['agreement_reached_mean']*100:.1f}%",
            f"{row['initial_total_cost_mean']:.1f}",
            f"{row['final_total_cost_mean']:.1f}",
            f"{row['cost_change_percentage_mean']:.1f}%",
            f"{row['total_rounds_mean']:.1f}",
            f"{row['execution_time_mean']:.1f}"
        ])
    
    table = plt.table(cellText=breakdown_data[1:], colLabels=breakdown_data[0], 
                     cellLoc='center', loc='center', colWidths=[0.1]*9)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(breakdown_data)):
        for j in range(9):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#339af0')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                table[(i, j)].set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    plt.title('Detailed Breakdown by Agent Count', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_breakdown_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed breakdown
    print("\n" + "-"*60)
    print("BREAKDOWN BY AGENT COUNT")
    print("-"*60)
    print(agent_breakdown)
    
    # Save detailed results to CSV
    output_file = f"negotiation_stats_{session_pattern.replace('*', 'all')}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    print(f"Visualizations saved to: {output_dir}/")
    
    return {
        'summary': summary_stats,
        'detailed_data': df,
        'agent_breakdown': agent_breakdown
    }


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_fair_comparison_batch(strategy: str, seeds=None):
    seeds = seeds or FAIR_COMPARISON_SEEDS
    latest_case_file, latest_folder = resolve_scenario_file()
    scenario_hash = _file_sha256(latest_case_file)

    print("\n=== FAIR COMPARISON MODE ===")
    print(f"Strategy: {strategy}")
    print(f"Scenario folder: {latest_folder}")
    print(f"Scenario file: {latest_case_file}")
    print(f"Scenario SHA256: {scenario_hash}")
    print(f"Seeds: {seeds}")
    print(f"OR-Tools workers: {ORTOOLS_NUM_SEARCH_WORKERS}")

    os.environ["DARP_NEGO_ORTOOLS_NUM_SEARCH_WORKERS"] = str(ORTOOLS_NUM_SEARCH_WORKERS)

    for seed in seeds:
        print(f"\n--- Running seed={seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        os.environ["DARP_NEGO_ORTOOLS_SEED"] = str(seed)
        main(strategy=strategy, run_seed=seed)

if __name__ == "__main__":
    if DEBUG_SINGLE_SCENARIO:
        debug_single_scenario()
    elif FAIR_COMPARISON_MODE:
        run_fair_comparison_batch(strategy=RUN_STRATEGY, seeds=FAIR_COMPARISON_SEEDS)
    else:
        os.environ["DARP_NEGO_ORTOOLS_NUM_SEARCH_WORKERS"] = str(ORTOOLS_NUM_SEARCH_WORKERS)
        os.environ["DARP_NEGO_ORTOOLS_SEED"] = "42"
        main(strategy=RUN_STRATEGY, run_seed=42)
        # Analyze logs after running the main function
        print("\n" + "="*60)
        print("ANALYZING NEGOTIATION LOGS")
        print("="*60)
        analyze_negotiation_logs("llm_generated_case_*")
