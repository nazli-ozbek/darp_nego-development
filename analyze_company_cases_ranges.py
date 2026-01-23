#!/usr/bin/env python3
"""
Script to analyze company_cases.json case by case.
Finds ranges for all attributes within each case separately.
This helps identify which cases have balanced values that lead to full_acceptance.
"""

import json
from pathlib import Path
from collections import defaultdict


def analyze_case(case_id, case_data):
    """
    Analyze a single case and return its attribute ranges.
    
    Args:
        case_id: ID of the case
        case_data: Data for the case
        
    Returns:
        Dictionary with ranges for each attribute within this case
    """
    case_ranges = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'values': set()})
    
    companies = case_data.get("companies", {})
    num_companies = len(companies)
    
    # Track per-company statistics
    vehicles_per_company = []
    clients_per_company = []
    
    # Track all values across all companies in this case
    all_vehicle_max_volumes = []
    all_client_volumes = []
    all_client_start_locations = []
    all_client_end_locations = []
    all_early_pickups = []
    all_late_pickups = []
    all_early_drop_offs = []
    all_late_drop_offs = []
    
    # Iterate through companies
    for company_id, company_data in companies.items():
        # Analyze vehicles
        vehicles = company_data.get("vehicles", [])
        vehicles_per_company.append(len(vehicles))
        
        for vehicle in vehicles:
            if "max_volume" in vehicle:
                val = vehicle["max_volume"]
                all_vehicle_max_volumes.append(val)
                case_ranges["vehicle.max_volume"]["min"] = min(case_ranges["vehicle.max_volume"]["min"], val)
                case_ranges["vehicle.max_volume"]["max"] = max(case_ranges["vehicle.max_volume"]["max"], val)
                case_ranges["vehicle.max_volume"]["values"].add(val)
            if "start_location" in vehicle:
                val = vehicle["start_location"]
                case_ranges["vehicle.start_location"]["min"] = min(case_ranges["vehicle.start_location"]["min"], val)
                case_ranges["vehicle.start_location"]["max"] = max(case_ranges["vehicle.start_location"]["max"], val)
                case_ranges["vehicle.start_location"]["values"].add(val)
            if "end_location" in vehicle:
                val = vehicle["end_location"]
                case_ranges["vehicle.end_location"]["min"] = min(case_ranges["vehicle.end_location"]["min"], val)
                case_ranges["vehicle.end_location"]["max"] = max(case_ranges["vehicle.end_location"]["max"], val)
                case_ranges["vehicle.end_location"]["values"].add(val)
        
        # Analyze clients
        clients = company_data.get("clients", [])
        clients_per_company.append(len(clients))
        
        for client in clients:
            if "volume" in client:
                val = client["volume"]
                all_client_volumes.append(val)
                case_ranges["client.volume"]["min"] = min(case_ranges["client.volume"]["min"], val)
                case_ranges["client.volume"]["max"] = max(case_ranges["client.volume"]["max"], val)
                case_ranges["client.volume"]["values"].add(val)
            if "start_location" in client:
                val = client["start_location"]
                all_client_start_locations.append(val)
                case_ranges["client.start_location"]["min"] = min(case_ranges["client.start_location"]["min"], val)
                case_ranges["client.start_location"]["max"] = max(case_ranges["client.start_location"]["max"], val)
                case_ranges["client.start_location"]["values"].add(val)
            if "end_location" in client:
                val = client["end_location"]
                all_client_end_locations.append(val)
                case_ranges["client.end_location"]["min"] = min(case_ranges["client.end_location"]["min"], val)
                case_ranges["client.end_location"]["max"] = max(case_ranges["client.end_location"]["max"], val)
                case_ranges["client.end_location"]["values"].add(val)
            if "early_pickup" in client:
                val = client["early_pickup"]
                all_early_pickups.append(val)
                case_ranges["client.early_pickup"]["min"] = min(case_ranges["client.early_pickup"]["min"], val)
                case_ranges["client.early_pickup"]["max"] = max(case_ranges["client.early_pickup"]["max"], val)
                case_ranges["client.early_pickup"]["values"].add(val)
            if "late_pickup" in client:
                val = client["late_pickup"]
                all_late_pickups.append(val)
                case_ranges["client.late_pickup"]["min"] = min(case_ranges["client.late_pickup"]["min"], val)
                case_ranges["client.late_pickup"]["max"] = max(case_ranges["client.late_pickup"]["max"], val)
                case_ranges["client.late_pickup"]["values"].add(val)
            if "early_drop_off" in client:
                val = client["early_drop_off"]
                all_early_drop_offs.append(val)
                case_ranges["client.early_drop_off"]["min"] = min(case_ranges["client.early_drop_off"]["min"], val)
                case_ranges["client.early_drop_off"]["max"] = max(case_ranges["client.early_drop_off"]["max"], val)
                case_ranges["client.early_drop_off"]["values"].add(val)
            if "late_drop_off" in client:
                val = client["late_drop_off"]
                all_late_drop_offs.append(val)
                case_ranges["client.late_drop_off"]["min"] = min(case_ranges["client.late_drop_off"]["min"], val)
                case_ranges["client.late_drop_off"]["max"] = max(case_ranges["client.late_drop_off"]["max"], val)
                case_ranges["client.late_drop_off"]["values"].add(val)
    
    # Calculate case-level statistics
    case_ranges["statistics.num_companies"] = num_companies
    case_ranges["statistics.num_vehicles_per_company"] = {
        "min": min(vehicles_per_company) if vehicles_per_company else 0,
        "max": max(vehicles_per_company) if vehicles_per_company else 0,
        "values": vehicles_per_company
    }
    case_ranges["statistics.num_clients_per_company"] = {
        "min": min(clients_per_company) if clients_per_company else 0,
        "max": max(clients_per_company) if clients_per_company else 0,
        "values": clients_per_company
    }
    
    # Calculate aggregate statistics for the case
    if all_vehicle_max_volumes:
        case_ranges["aggregate.total_vehicle_capacity"] = sum(all_vehicle_max_volumes)
        case_ranges["aggregate.avg_vehicle_capacity"] = sum(all_vehicle_max_volumes) / len(all_vehicle_max_volumes)
    if all_client_volumes:
        case_ranges["aggregate.total_client_volume"] = sum(all_client_volumes)
        case_ranges["aggregate.avg_client_volume"] = sum(all_client_volumes) / len(all_client_volumes)
    if all_early_pickups:
        case_ranges["aggregate.early_pickup_range"] = max(all_early_pickups) - min(all_early_pickups)
    if all_late_drop_offs:
        case_ranges["aggregate.late_drop_off_range"] = max(all_late_drop_offs) - min(all_late_drop_offs)
    
    # Clean up ranges
    cleaned_ranges = {}
    for attr_name, attr_data in case_ranges.items():
        if attr_name.startswith("statistics.") or attr_name.startswith("aggregate."):
            cleaned_ranges[attr_name] = attr_data
        elif attr_data["min"] != float('inf'):
            cleaned_ranges[attr_name] = {
                "min": attr_data["min"],
                "max": attr_data["max"]
            }
            if len(attr_data["values"]) <= 20:
                cleaned_ranges[attr_name]["unique_values"] = sorted(attr_data["values"])
            else:
                cleaned_ranges[attr_name]["unique_values_count"] = len(attr_data["values"])
    
    return cleaned_ranges


def analyze_all_cases(json_file_path):
    """
    Analyze all cases in the JSON file case by case.
    
    Args:
        json_file_path: Path to company_cases.json file
        
    Returns:
        Dictionary mapping case_id to its ranges
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    case_analyses = {}
    for case_id, case_data in data.items():
        case_analyses[case_id] = analyze_case(case_id, case_data)
    
    return case_analyses


def print_case_analysis(case_id, case_ranges):
    """Print analysis for a single case"""
    print(f"\n{'='*80}")
    print(f"CASE: {case_id}")
    print(f"{'='*80}")
    
    # Statistics
    if "statistics.num_companies" in case_ranges:
        print(f"\nStatistics:")
        print(f"  Number of Companies: {case_ranges['statistics.num_companies']}")
        if "statistics.num_vehicles_per_company" in case_ranges:
            vpc = case_ranges["statistics.num_vehicles_per_company"]
            print(f"  Vehicles per Company: {vpc['min']}-{vpc['max']} (values: {vpc['values']})")
        if "statistics.num_clients_per_company" in case_ranges:
            cpc = case_ranges["statistics.num_clients_per_company"]
            print(f"  Clients per Company: {cpc['min']}-{cpc['max']} (values: {cpc['values']})")
    
    # Aggregate values
    if "aggregate" in str(case_ranges):
        print(f"\nAggregate Values:")
        for key, value in case_ranges.items():
            if key.startswith("aggregate."):
                attr_name = key.replace("aggregate.", "")
                if isinstance(value, (int, float)):
                    print(f"  {attr_name}: {value:.2f}" if isinstance(value, float) else f"  {attr_name}: {value}")
    
    # Vehicle attributes
    vehicle_attrs = [k for k in case_ranges.keys() if k.startswith("vehicle.")]
    if vehicle_attrs:
        print(f"\nVehicle Attributes:")
        for attr in sorted(vehicle_attrs):
            data = case_ranges[attr]
            if "min" in data:
                print(f"  {attr}:")
                print(f"    Range: [{data['min']}, {data['max']}]")
                if "unique_values" in data:
                    print(f"    Values: {data['unique_values']}")
                elif "unique_values_count" in data:
                    print(f"    Unique Values: {data['unique_values_count']}")
    
    # Client attributes
    client_attrs = [k for k in case_ranges.keys() if k.startswith("client.")]
    if client_attrs:
        print(f"\nClient Attributes:")
        for attr in sorted(client_attrs):
            data = case_ranges[attr]
            if "min" in data:
                print(f"  {attr}:")
                print(f"    Range: [{data['min']}, {data['max']}]")
                if "unique_values" in data:
                    print(f"    Values: {data['unique_values']}")
                elif "unique_values_count" in data:
                    print(f"    Unique Values: {data['unique_values_count']}")


def print_summary(case_analyses):
    """Print summary across all cases"""
    print(f"\n{'='*80}")
    print(f"SUMMARY ACROSS ALL CASES")
    print(f"{'='*80}")
    
    # Collect all values for each attribute across cases
    summary = defaultdict(list)
    
    for case_id, case_ranges in case_analyses.items():
        for attr_name, attr_data in case_ranges.items():
            if attr_name.startswith("statistics.") or attr_name.startswith("aggregate."):
                if isinstance(attr_data, dict) and "min" in attr_data:
                    summary[attr_name].append((attr_data["min"], attr_data["max"]))
                elif isinstance(attr_data, (int, float)):
                    summary[attr_name].append(attr_data)
            elif isinstance(attr_data, dict) and "min" in attr_data:
                summary[attr_name].append((attr_data["min"], attr_data["max"]))
    
    print(f"\nAttribute Ranges Across Cases:")
    for attr_name, values in sorted(summary.items()):
        if values and isinstance(values[0], tuple):
            all_mins = [v[0] for v in values]
            all_maxs = [v[1] for v in values]
            print(f"  {attr_name}:")
            print(f"    Min across cases: {min(all_mins)}")
            print(f"    Max across cases: {max(all_maxs)}")
            print(f"    Case ranges: {values}")
        elif values:
            print(f"  {attr_name}:")
            print(f"    Range: [{min(values):.2f}, {max(values):.2f}]" if isinstance(values[0], float) else f"    Range: [{min(values)}, {max(values)}]")


def main():
    """Main function"""
    json_file = Path("darp_nego/python/darp_nego/scenario_generation/data/llm_generated/company_cases.json")
    
    if not json_file.exists():
        print(f"Error: File not found: {json_file}")
        return
    
    print("Analyzing company_cases.json case by case...")
    case_analyses = analyze_all_cases(json_file)
    
    # Print each case
    for case_id in sorted(case_analyses.keys()):
        print_case_analysis(case_id, case_analyses[case_id])
    
    # Print summary
    print_summary(case_analyses)
    
    print(f"\n{'='*80}")
    print(f"Total Cases Analyzed: {len(case_analyses)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
