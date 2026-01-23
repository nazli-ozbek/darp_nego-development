#!/usr/bin/env python3
"""
Script to generate 10 cases targeting full_acceptance based on analyzed ranges.
Uses balanced values across companies to increase likelihood of successful negotiation.
"""

import json
import random
import math
from pathlib import Path


# Ranges optimized based on successful full_acceptance cases
# Data from: case_1, case_3, case_4, case_6, case_7 (all achieved full_acceptance)
RANGES = {
    "num_companies": (3, 5),  # Keep as requested, but successful ones are 4-5
    "vehicles_per_company": (1, 4),
    "clients_per_company": (2, 6),
    "vehicle_max_volume": (7, 20),  # Successful cases: 7-20 (minimum 7 for better capacity)
    "vehicle_start_location": (0, 7),
    "vehicle_end_location": (0, 7),
    "client_volume": (1, 5),
    "client_start_location": (9, 28),
    "client_end_location": (7, 11),
    # Time windows optimized from successful cases
    "early_pickup": (16, 439),  # Successful range: 16-439 (was 13-444)
    "late_pickup": (29, 453),   # Successful range: 29-453 (was 25-462)
    "early_drop_off": (46, 470), # Successful range: 46-470 (was 43-476)
    "late_drop_off": (68, 499),  # Successful range: 68-499 (was 62-505)
    "coordinate_range": (-10, 10),
    # Aggregate targets from successful cases
    "target_total_vehicle_capacity": (77, 172),
    "target_total_client_volume": (33, 70),
    "target_avg_vehicle_capacity": (11.0, 15.88),
    "target_avg_client_volume": (2.54, 3.59),
    "target_early_pickup_range": (353, 427),  # Range within case
    "target_late_drop_off_range": (358, 427)   # Range within case
}


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_time_matrix_from_coords(coordinates, speed_units_per_minute=0.5):
    """Generate time matrix from coordinates"""
    ids = sorted(coordinates.keys(), key=int)
    size = len(ids)
    time_matrix = [[0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            if i == j:
                time_matrix[i][j] = 0
            else:
                dist = euclidean_distance(coordinates[ids[i]], coordinates[ids[j]])
                time = int(dist / speed_units_per_minute)
                time_matrix[i][j] = max(1, time)  # Ensure minimum time of 1
    
    return time_matrix


def generate_coordinates(num_locations):
    """Generate coordinates for locations"""
    coordinates = {}
    for i in range(num_locations):
        x = round(random.uniform(RANGES["coordinate_range"][0], RANGES["coordinate_range"][1]), 2)
        y = round(random.uniform(RANGES["coordinate_range"][0], RANGES["coordinate_range"][1]), 2)
        coordinates[str(i)] = [x, y]
    return coordinates


def generate_vehicle(vehicle_id, max_location):
    """Generate a single vehicle"""
    start_loc = random.randint(RANGES["vehicle_start_location"][0], 
                               min(RANGES["vehicle_start_location"][1], max_location))
    end_loc = random.randint(RANGES["vehicle_end_location"][0], 
                            min(RANGES["vehicle_end_location"][1], max_location))
    max_volume = random.randint(RANGES["vehicle_max_volume"][0], RANGES["vehicle_max_volume"][1])
    
    return {
        "vehicle_id": vehicle_id,
        "start_location": start_loc,
        "end_location": end_loc,
        "max_volume": max_volume
    }


def generate_client(client_id, max_location):
    """Generate a single client with balanced time windows optimized for full_acceptance"""
    start_loc = random.randint(RANGES["client_start_location"][0], 
                              min(RANGES["client_start_location"][1], max_location))
    end_loc = random.randint(RANGES["client_end_location"][0], 
                            min(RANGES["client_end_location"][1], max_location))
    volume = random.randint(RANGES["client_volume"][0], RANGES["client_volume"][1])
    
    # Generate time windows optimized from successful cases
    # Use narrower ranges that worked better
    early_pickup = random.randint(RANGES["early_pickup"][0], RANGES["early_pickup"][1] - 50)
    late_pickup = early_pickup + random.randint(5, 20)  # Pickup window: 5-20 minutes
    service_time = random.randint(10, 30)  # Service time: 10-30 minutes
    early_drop_off = late_pickup + service_time
    late_drop_off = early_drop_off + random.randint(10, 40)  # Drop-off window: 10-40 minutes
    
    # Ensure within optimized ranges
    late_pickup = min(late_pickup, RANGES["late_pickup"][1])
    early_drop_off = max(early_drop_off, RANGES["early_drop_off"][0])
    early_drop_off = min(early_drop_off, RANGES["early_drop_off"][1] - 30)
    late_drop_off = min(late_drop_off, RANGES["late_drop_off"][1])
    
    return {
        "client_id": client_id,
        "start_location": start_loc,
        "end_location": end_loc,
        "early_pickup": early_pickup,
        "late_pickup": late_pickup,
        "early_drop_off": early_drop_off,
        "late_drop_off": late_drop_off,
        "volume": volume
    }


def generate_company(company_id, num_vehicles, num_clients, vehicle_id_start, client_id_start, max_location):
    """Generate a single company with balanced resources"""
    vehicles = []
    for i in range(num_vehicles):
        vehicles.append(generate_vehicle(vehicle_id_start + i, max_location))
    
    clients = []
    for i in range(num_clients):
        clients.append(generate_client(client_id_start + i, max_location))
    
    return {
        "vehicles": vehicles,
        "clients": clients
    }


def generate_case(case_id):
    """Generate a single case optimized for full_acceptance"""
    # Determine number of companies (3-5, but prefer 4-5 for better success rate)
    # Weight towards 4-5 companies (80% chance)
    if random.random() < 0.8:
        num_companies = random.randint(4, 5)
    else:
        num_companies = random.randint(3, 5)
    
    # Target aggregate values from successful cases
    target_total_capacity = random.randint(RANGES["target_total_vehicle_capacity"][0], 
                                          RANGES["target_total_vehicle_capacity"][1])
    target_total_volume = random.randint(RANGES["target_total_client_volume"][0], 
                                        RANGES["target_total_client_volume"][1])
    target_avg_capacity = random.uniform(RANGES["target_avg_vehicle_capacity"][0], 
                                        RANGES["target_avg_vehicle_capacity"][1])
    target_avg_volume = random.uniform(RANGES["target_avg_client_volume"][0], 
                                      RANGES["target_avg_client_volume"][1])
    
    # Calculate target number of vehicles and clients
    target_total_vehicles = max(7, min(15, int(target_total_capacity / target_avg_capacity)))
    target_total_clients = max(13, min(25, int(target_total_volume / target_avg_volume)))
    
    # Distribute vehicles and clients across companies (balanced)
    vehicles_per_company = []
    clients_per_company = []
    remaining_vehicles = target_total_vehicles
    remaining_clients = target_total_clients
    
    for i in range(num_companies):
        if i == num_companies - 1:
            # Last company gets remaining
            vehicles = remaining_vehicles
            clients = remaining_clients
        else:
            # Distribute proportionally
            vehicles = max(1, int(remaining_vehicles / (num_companies - i)))
            vehicles = min(vehicles, RANGES["vehicles_per_company"][1], remaining_vehicles)
            clients = max(2, int(remaining_clients / (num_companies - i)))
            clients = min(clients, RANGES["clients_per_company"][1], remaining_clients)
        
        vehicles_per_company.append(vehicles)
        clients_per_company.append(clients)
        remaining_vehicles -= vehicles
        remaining_clients -= clients
    
    # Determine max location needed
    max_location = max(RANGES["client_start_location"][1], RANGES["vehicle_start_location"][1])
    
    # Generate companies with capacity/volume targeting
    companies = {}
    vehicle_id_counter = 0
    client_id_counter = 0
    
    # Target time window range within case
    target_early_pickup_min = random.randint(RANGES["target_early_pickup_range"][0], 
                                            RANGES["target_early_pickup_range"][1] - 50)
    target_late_drop_off_max = target_early_pickup_min + random.randint(
        RANGES["target_late_drop_off_range"][0], 
        RANGES["target_late_drop_off_range"][1])
    
    for i in range(num_companies):
        company_id = f"company_{i}"
        
        # Generate vehicles with capacity targeting
        vehicles = []
        for j in range(vehicles_per_company[i]):
            # Target average capacity per vehicle
            capacity = int(target_avg_capacity + random.uniform(-3, 3))
            capacity = max(RANGES["vehicle_max_volume"][0], 
                          min(RANGES["vehicle_max_volume"][1], capacity))
            
            vehicle = generate_vehicle(vehicle_id_counter + j, max_location)
            vehicle["max_volume"] = capacity
            vehicles.append(vehicle)
        
        # Generate clients with volume and time window targeting
        clients = []
        for j in range(clients_per_company[i]):
            # Target average volume per client
            volume = int(target_avg_volume + random.uniform(-1, 1))
            volume = max(RANGES["client_volume"][0], 
                        min(RANGES["client_volume"][1], volume))
            
            client = generate_client(client_id_counter + j, max_location)
            client["volume"] = volume
            
            # Adjust time windows to fit within target range
            # Ensure early_pickup is within target range
            if client["early_pickup"] < target_early_pickup_min:
                offset = target_early_pickup_min - client["early_pickup"]
                client["early_pickup"] += offset
                client["late_pickup"] += offset
                client["early_drop_off"] += offset
                client["late_drop_off"] += offset
            
            # Ensure late_drop_off doesn't exceed target max
            if client["late_drop_off"] > target_late_drop_off_max:
                offset = client["late_drop_off"] - target_late_drop_off_max
                client["early_pickup"] = max(RANGES["early_pickup"][0], 
                                            client["early_pickup"] - offset)
                client["late_pickup"] = max(client["early_pickup"] + 5, 
                                           client["late_pickup"] - offset)
                client["early_drop_off"] = max(client["late_pickup"] + 10, 
                                              client["early_drop_off"] - offset)
                client["late_drop_off"] = target_late_drop_off_max
            
            clients.append(client)
        
        companies[company_id] = {
            "vehicles": vehicles,
            "clients": clients
        }
        
        vehicle_id_counter += vehicles_per_company[i]
        client_id_counter += clients_per_company[i]
    
    # Generate coordinates for all locations (0 to max_location)
    coordinates = generate_coordinates(max_location + 1)
    
    # Generate time matrix from coordinates
    time_matrix = generate_time_matrix_from_coords(coordinates)
    
    case = {
        "companies": companies,
        "coordinates": coordinates,
        "time_matrix": time_matrix
    }
    
    return case


def generate_dataset(num_cases=10, seed=42):
    """Generate a dataset of cases"""
    random.seed(seed)
    
    dataset = {}
    for i in range(1, num_cases + 1):
        case_id = f"case_{i}"
        print(f"Generating {case_id}...")
        dataset[case_id] = generate_case(case_id)
    
    return dataset


def main():
    """Main function"""
    output_file = Path("darp_nego/python/darp_nego/scenario_generation/data/llm_generated/company_cases_full_acceptance.json")
    
    print("Generating 10 cases targeting full_acceptance...")
    print("=" * 80)
    
    dataset = generate_dataset(num_cases=1000, seed=42)
    
    # Save to JSON file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Dataset generated successfully!")
    print(f"Output file: {output_file}")
    print(f"Total cases: {len(dataset)}")
    print(f"{'='*80}")
    
    # Print summary
    print("\nCase Summary:")
    for case_id, case_data in dataset.items():
        num_companies = len(case_data["companies"])
        total_vehicles = sum(len(c["vehicles"]) for c in case_data["companies"].values())
        total_clients = sum(len(c["clients"]) for c in case_data["companies"].values())
        total_capacity = sum(v["max_volume"] for c in case_data["companies"].values() 
                           for v in c["vehicles"])
        total_volume = sum(cl["volume"] for c in case_data["companies"].values() 
                          for cl in c["clients"])
        
        print(f"  {case_id}: {num_companies} companies, {total_vehicles} vehicles, "
              f"{total_clients} clients, capacity={total_capacity}, volume={total_volume}")


if __name__ == "__main__":
    main()

