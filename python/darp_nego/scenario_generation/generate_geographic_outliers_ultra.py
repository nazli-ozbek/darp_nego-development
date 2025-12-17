import random
import json
import os
from datetime import datetime
import math
import numpy as np

def save_json(folder, filename, data):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    return file_path

def generate_coordinates(num_locations, x_range=(-10, 10), y_range=(-10, 10), precision=2):
    coordinates = {}
    for loc_id in range(num_locations):
        x = round(random.uniform(*x_range), precision)
        y = round(random.uniform(*y_range), precision)
        coordinates[int(loc_id)] = [x, y]
    return coordinates

def generate_company_cluster_coordinates_ultra_outlier(company_id, num_clients, base_radius=0.5, outlier_radius=8.0, precision=2):
    """
    Generate coordinates for a company's clients with one ULTRA pronounced geographic outlier.
    All clients except the outlier are very tightly clustered together.
    ULTRA: Much tighter clustering and much larger outlier distance.
    
    Args:
        company_id: Company identifier
        num_clients: Number of clients for this company
        base_radius: Radius for clustered clients (ultra tight: 0.5)
        outlier_radius: Distance for outlier client (ultra large: 8.0)
        precision: Decimal precision for coordinates
    
    Returns:
        List of [x, y] coordinates
    """
    # Define company base locations in different quadrants (within bounds)
    company_bases = {
        0: (-6, 6),   # NW quadrant
        1: (6, 6),    # NE quadrant  
        2: (-6, -6),  # SW quadrant
        3: (6, -6),   # SE quadrant
        4: (0, 7),    # North
        5: (0, -7),   # South
        6: (-7, 0),   # West
        7: (7, 0),    # East
    }
    
    base_x, base_y = company_bases.get(company_id, (0, 0))
    coordinates = []
    
    # Generate ultra tightly clustered clients around the base
    for i in range(num_clients - 1):  # All but one client are very close
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, base_radius)  # Ultra small radius for tight clustering
        x = base_x + distance * math.cos(angle)
        y = base_y + distance * math.sin(angle)
        
        # Ensure coordinates stay within bounds
        x = max(-10, min(10, x))
        y = max(-10, min(10, y))
        
        coordinates.append([round(x, precision), round(y, precision)])
    
    # Generate one ultra outlier client very far from the cluster
    outlier_angle = random.uniform(0, 2 * math.pi)
    outlier_x = base_x + outlier_radius * math.cos(outlier_angle)
    outlier_y = base_y + outlier_radius * math.sin(outlier_angle)
    
    # Ensure outlier stays within bounds
    outlier_x = max(-10, min(10, outlier_x))
    outlier_y = max(-10, min(10, outlier_y))
    
    coordinates.append([round(outlier_x, precision), round(outlier_y, precision)])
    
    return coordinates

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_time_matrix_from_coords(coordinates, speed_units_per_minute=0.5):
    ids = sorted(coordinates.keys())
    size = len(ids)
    time_matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j:
                time_matrix[i][j] = 0
            else:
                dist = euclidean_distance(coordinates[ids[i]], coordinates[ids[j]])
                time = int(dist / speed_units_per_minute)
                time_matrix[i][j] = time
    return time_matrix

def generate_vehicles(num_vehicles):
    vehicles = []
    for i in range(num_vehicles):
        start_location = i
        end_location = i + 1 if i + 1 < num_vehicles else 0
        max_volume = random.randint(5, 20)
        vehicles.append({
            "vehicle_id": i,
            "start_location": start_location,
            "end_location": end_location,
            "max_volume": max_volume
        })
    return vehicles

def generate_clients_with_company_similarity(num_clients, num_vehicles, num_hospitals, min_vehicle_capacity=5):
    """
    Generate clients where each company's clients have similar characteristics,
    but one client per company will be geographically distant.
    """
    clients = []
    hospital_domain = list(range(num_vehicles, num_vehicles + num_hospitals))
    client_domain = list(range(num_vehicles + num_hospitals, num_vehicles + num_hospitals + num_clients))

    for i in range(num_clients):
        start_location = client_domain[i]
        end_location = random.choice(hospital_domain)

        # All clients have similar characteristics (no outliers in data)
        early_pickup = random.randint(50, 300)  # Normal range
        gap1 = random.randint(10, 25)  # Normal window
        late_pickup = early_pickup + gap1
        gap2 = random.randint(15, 35)  # Normal window
        early_drop = late_pickup + gap2
        gap3 = random.randint(10, 25)  # Normal window
        late_drop = early_drop + gap3
        
        volume = random.randint(2, min_vehicle_capacity - 1)  # Normal volume

        clients.append({
            "client_id": i,
            "start_location": start_location,
            "end_location": end_location,
            "early_pickup": early_pickup,
            "late_pickup": late_pickup,
            "early_drop_off": early_drop,
            "late_drop_off": late_drop,
            "volume": volume
        })
    
    return clients

def distribute_to_companies_with_ultra_outliers(vehicles, clients, num_companies, coordinates, num_vehicles, num_hospitals):
    """
    Distribute clients to companies ensuring each company has one ULTRA pronounced geographically distant outlier.
    All clients within a company have similar data characteristics.
    ULTRA: Much tighter clustering and much larger outlier distance.
    """
    companies = {f"company_{h_id}": {"vehicles": [], "clients": []} for h_id in range(num_companies)}
    
    # Assign vehicles randomly to companies
    random.shuffle(vehicles)
    for h_id in range(num_companies):
        companies[f"company_{h_id}"]["vehicles"].append(vehicles.pop())

    for vehicle in vehicles:
        h_id = random.choice(range(num_companies))
        companies[f"company_{h_id}"]["vehicles"].append(vehicle)
    
    # Calculate clients per company
    total_clients = len(clients)
    clients_per_company = total_clients // num_companies
    extra_clients = total_clients % num_companies
    
    # Distribute clients to companies
    client_index = 0
    for company_id in range(num_companies):
        # Calculate how many clients this company gets
        num_clients_for_company = clients_per_company
        if company_id < extra_clients:
            num_clients_for_company += 1
        
        # Get clients for this company
        company_clients = clients[client_index:client_index + num_clients_for_company]
        companies[f"company_{company_id}"]["clients"] = company_clients
        
        # Generate coordinates for this company's clients (ultra tightly clustered + one ultra distant outlier)
        company_coords = generate_company_cluster_coordinates_ultra_outlier(company_id, num_clients_for_company)
        
        # Assign coordinates to clients
        for i, client in enumerate(company_clients):
            coordinates[client["start_location"]] = company_coords[i]
        
        client_index += num_clients_for_company
    
    return companies

def generate_scenarios_with_ultra_outliers():
    """
    Generate 10 scenarios with ULTRA pronounced geographic outliers:
    - 3 scenarios with 3 companies
    - 4 scenarios with 4 companies  
    - 3 scenarios with 5 companies
    
    Each company has clients with similar data characteristics,
    but one client per company is ULTRA geographically distant.
    ULTRA: Much tighter clustering and much larger outlier distance.
    """
    scenarios = []
    
    # Define scenario configurations
    scenario_configs = [
        # 3 scenarios with 3 companies
        {"num_companies": 3, "num_clients": 12, "num_vehicles": 6, "num_hospitals": 3},
        {"num_companies": 3, "num_clients": 15, "num_vehicles": 7, "num_hospitals": 3},
        {"num_companies": 3, "num_clients": 18, "num_vehicles": 8, "num_hospitals": 4},
        
        # 4 scenarios with 4 companies
        {"num_companies": 4, "num_clients": 16, "num_vehicles": 8, "num_hospitals": 4},
        {"num_companies": 4, "num_clients": 20, "num_vehicles": 10, "num_hospitals": 4},
        {"num_companies": 4, "num_clients": 24, "num_vehicles": 12, "num_hospitals": 5},
        {"num_companies": 4, "num_clients": 28, "num_vehicles": 14, "num_hospitals": 5},
        
        # 3 scenarios with 5 companies
        {"num_companies": 5, "num_clients": 25, "num_vehicles": 12, "num_hospitals": 5},
        {"num_companies": 5, "num_clients": 30, "num_vehicles": 15, "num_hospitals": 6},
        {"num_companies": 5, "num_clients": 35, "num_vehicles": 18, "num_hospitals": 6},
    ]
    
    # Use consistent speed for all scenarios
    speed_units_per_minute = 0.5
    
    for i, config in enumerate(scenario_configs):
        num_companies = config["num_companies"]
        num_clients = config["num_clients"]
        num_vehicles = config["num_vehicles"]
        num_hospitals = config["num_hospitals"]
        
        # Generate all locations (vehicles and hospitals first)
        total_locations = num_vehicles + num_hospitals + num_clients
        coordinates = generate_coordinates(total_locations)
        
        # Generate clients and vehicles (similar characteristics within companies)
        clients = generate_clients_with_company_similarity(num_clients, num_vehicles, num_hospitals)
        vehicles = generate_vehicles(num_vehicles)
        
        # Distribute to companies with ULTRA geographic outlier positioning
        # This will update the coordinates for client locations
        companies = distribute_to_companies_with_ultra_outliers(vehicles, clients, num_companies, coordinates, num_vehicles, num_hospitals)
        
        # Generate time matrix AFTER all coordinates are finalized
        time_matrix = generate_time_matrix_from_coords(coordinates, speed_units_per_minute)
        
        scenarios.append({
            "companies": companies,
            "time_matrix": time_matrix,
            "coordinates": coordinates
        })
    
    return scenarios

if __name__ == "__main__":
    # Generate scenarios
    scenarios = generate_scenarios_with_ultra_outliers()
    
    # Create timestamp folder
    timestamp = "2025_08_08_14_00_ultra"
    folder_name = f"data/{timestamp}"
    
    # Save each scenario
    cases_data = {}
    for i, scenario in enumerate(scenarios, 1):
        cases_data[f"case_{i}"] = scenario
    
    file_path_cases = save_json(folder_name, "company_cases.json", cases_data)
    
    print(f"Generated {len(scenarios)} scenarios with ULTRA pronounced geographic outliers!")
    print(f"Cases saved in: {file_path_cases}")
    
    # Show summary
    print("\nScenario Summary:")
    for i, (case_name, case_data) in enumerate(cases_data.items()):
        num_companies = len(case_data["companies"])
        total_clients = sum(len(company["clients"]) for company in case_data["companies"].values())
        print(f"- {case_name}: {num_companies} companies, {total_clients} total clients")
    
    print(f"\nULTRA FIXES APPLIED:")
    print(f"- Coordinates now stay within -10 to +10 bounds")
    print(f"- Consistent speed of 0.5 units/minute for all scenarios")
    print(f"- Time matrix calculated from final coordinates")
    print(f"- ULTRA tight clustering (radius 0.5) for maximum outlier detection")
    print(f"- ULTRA large outliers (distance 8.0) for extremely clear patterns")
    print(f"- Each company has clients with similar data characteristics")
    print(f"- Each company has one client that is ULTRA geographically distant from the others")
    print(f"- This should DEFINITELY help the farthest distance algorithm find solutions in earlier rounds") 