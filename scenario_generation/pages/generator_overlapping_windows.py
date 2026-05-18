import streamlit as st
import random
import json
import os
from datetime import datetime
import math

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

def generate_quadrant_coordinate(quadrant, precision=2):
    # Define quadrant boundaries within -10 to 10 range
    if quadrant == 0:  # Longitude < 0, Latitude > 0 (NW)
        x_range = (-10, 0)
        y_range = (0, 10)
    elif quadrant == 1:  # Longitude > 0, Latitude > 0 (NE)
        x_range = (0, 10)
        y_range = (0, 10)
    elif quadrant == 2:  # Longitude < 0, Latitude < 0 (SW)
        x_range = (-10, 0)
        y_range = (-10, 0)
    else:  # quadrant 3: Longitude > 0, Latitude < 0 (SE)
        x_range = (0, 10)
        y_range = (-10, 0)
    
    x = round(random.uniform(*x_range), precision)
    y = round(random.uniform(*y_range), precision)
    return [x, y]

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

def generate_clients_with_overlapping_windows(num_clients, num_vehicles, num_hospitals, num_companies, min_vehicle_capacity=5):
    """
    Generate clients ensuring at least 2 clients per company have overlapping time windows.
    Returns a dictionary mapping company_id to their clients.
    """
    hospital_domain = list(range(num_vehicles, num_vehicles + num_hospitals))
    client_domain = list(range(num_vehicles + num_hospitals, num_vehicles + num_hospitals + num_clients))

    # Calculate how many clients each company will get
    clients_per_company = num_clients // num_companies
    remaining_clients = num_clients % num_companies
    
    client_id = 0
    companies_clients = {}  # Dictionary to store clients by company
    
    for company_id in range(num_companies):
        # Determine number of clients for this company
        company_clients = clients_per_company + (1 if company_id < remaining_clients else 0)
        
        # Ensure at least 2 clients for overlapping windows
        if company_clients < 2:
            company_clients = 2
        
        # Generate overlapping time windows for this company
        company_client_list = []
        
        # Create a base time window that will be shared by at least 2 clients
        base_early_pickup = random.randint(10, 400)
        base_late_pickup = base_early_pickup + random.randint(5, 30)
        base_early_drop = base_late_pickup + random.randint(10, 50)
        base_late_drop = base_early_drop + random.randint(5, 30)
        
        # Generate at least 2 clients with overlapping windows
        for i in range(min(2, company_clients)):
            start_location = client_domain[client_id]
            end_location = random.choice(hospital_domain)
            
            # Create overlapping time windows with some variation
            overlap_variation = random.randint(-20, 20)
            early_pickup = max(10, base_early_pickup + overlap_variation)
            late_pickup = early_pickup + random.randint(5, 25)
            early_drop = late_pickup + random.randint(10, 40)
            late_drop = early_drop + random.randint(5, 25)
            
            volume = random.randint(1, min_vehicle_capacity)
            
            company_client_list.append({
                "client_id": client_id,
                "start_location": start_location,
                "end_location": end_location,
                "early_pickup": early_pickup,
                "late_pickup": late_pickup,
                "early_drop_off": early_drop,
                "late_drop_off": late_drop,
                "volume": volume
            })
            client_id += 1
        
        # Generate remaining clients for this company (if any)
        for i in range(2, company_clients):
            start_location = client_domain[client_id]
            end_location = random.choice(hospital_domain)
            
            # These can have more varied time windows
            early_pickup = random.randint(10, 500)
            gap1 = random.randint(0, 20)
            late_pickup = early_pickup + gap1
            gap2 = random.randint(0, 30)
            early_drop = late_pickup + gap2
            gap3 = random.randint(0, 20)
            late_drop = early_drop + gap3
            
            volume = random.randint(1, min_vehicle_capacity)
            
            company_client_list.append({
                "client_id": client_id,
                "start_location": start_location,
                "end_location": end_location,
                "early_pickup": early_pickup,
                "late_pickup": late_pickup,
                "early_drop_off": early_drop,
                "late_drop_off": late_drop,
                "volume": volume
            })
            client_id += 1
        
        companies_clients[company_id] = company_client_list
    
    return companies_clients

def distribute_to_companies(vehicles, companies_clients, num_companies, coordinates, num_vehicles, num_hospitals):
    companies = {f"company_{h_id}": {"vehicles": [], "clients": []} for h_id in range(num_companies)}
    
    # Assign vehicles randomly to companies
    random.shuffle(vehicles)
    for h_id in range(num_companies):
        companies[f"company_{h_id}"]["vehicles"].append(vehicles.pop())

    for vehicle in vehicles:
        h_id = random.choice(range(num_companies))
        companies[f"company_{h_id}"]["vehicles"].append(vehicle)
    
    # Assign clients to companies (preserving the overlapping windows)
    for company_id, client_list in companies_clients.items():
        companies[f"company_{company_id}"]["clients"] = client_list
        
        # Update coordinates for each client's start location to be in the correct quadrant
        for client in client_list:
            quadrant = company_id % 4  # We'll cycle through 4 quadrants
            coordinates[client["start_location"]] = generate_quadrant_coordinate(quadrant)
    
    return companies

def verify_overlapping_windows(companies):
    """
    Verify that each company has at least 2 clients with overlapping time windows.
    Returns a dictionary with verification results.
    """
    verification_results = {}
    
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
                    overlapping_pairs.append((client1["client_id"], client2["client_id"]))
        
        verification_results[company_name] = {
            "total_clients": len(clients),
            "overlapping_pairs": overlapping_pairs,
            "has_overlap": len(overlapping_pairs) > 0,
            "overlap_count": len(overlapping_pairs)
        }
    
    return verification_results

# Streamlit UI
st.title("DARP Case Generator with Overlapping Time Windows")

st.markdown("""
This generator ensures that **at least 2 clients per company have overlapping time windows**, 
which creates more realistic negotiation scenarios where companies need to coordinate 
their schedules.
""")

num_cases = st.number_input("Number of Cases", min_value=1, max_value=100, value=50, step=1)

client_range = st.slider("Total Client Range per Case", 10, 30, (10, 15))
vehicle_range = st.slider("Total Vehicle Range per Case", 7, 15, (7, 8))
hospital_range = st.slider("Total Hospital Range per Case", 2, 10, (2, 10))
company_range = st.slider("Total Company Range per Case", 2, 10, (2, 4))

show_verification = st.checkbox("Show Time Window Verification", value=True)

if st.button("Generate Cases with Overlapping Windows"):
    cases_data = {}

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_name = f"data/{timestamp}"

    progress_bar = st.progress(0)
    status_text = st.empty()

    for case_index in range(1, num_cases + 1):
        status_text.text(f"Generating case {case_index}/{num_cases}...")
        
        num_clients = random.randint(*client_range)
        num_vehicles = random.randint(*vehicle_range)
        num_hospitals = random.randint(*hospital_range)
        num_companies = random.randint(*company_range)

        # Ensure minimum clients per company for overlapping windows
        min_clients_needed = num_companies * 2
        if num_clients < min_clients_needed:
            num_clients = min_clients_needed

        # First generate all locations (vehicles, hospitals, and clients)
        total_locations = num_vehicles + num_hospitals + num_clients
        coordinates = generate_coordinates(total_locations)
        
        # Generate clients with overlapping windows
        companies_clients = generate_clients_with_overlapping_windows(num_clients, num_vehicles, num_hospitals, num_companies)
        vehicles = generate_vehicles(num_vehicles)
        
        # Distribute to companies - this will update client start locations' coordinates
        companies = distribute_to_companies(vehicles, companies_clients, num_companies, coordinates, num_vehicles, num_hospitals)
        
        # Verify overlapping windows if requested
        if show_verification:
            verification = verify_overlapping_windows(companies)
            
            # Check if all companies have overlapping windows
            all_have_overlap = all(result["has_overlap"] for result in verification.values())
            
            if not all_have_overlap:
                st.warning(f"Case {case_index}: Some companies don't have overlapping windows!")
                for company, result in verification.items():
                    if not result["has_overlap"]:
                        st.write(f"  - {company}: {result['total_clients']} clients, no overlaps")
        
        # Generate time matrix
        time_matrix = generate_time_matrix_from_coords(coordinates)
        
        # Store case data
        case_data = {
            "time_matrix": time_matrix,
            "coordinates": coordinates,
            "companies": companies
        }
        
        cases_data[f"case_{case_index}"] = case_data
        
        progress_bar.progress(case_index / num_cases)

    # Save the generated data
    file_path = save_json(folder_name, "company_cases.json", cases_data)
    
    status_text.text("Generation complete!")
    st.success(f"Generated {num_cases} cases with overlapping time windows!")
    st.info(f"Data saved to: {file_path}")
    
    # Show verification summary
    if show_verification:
        st.subheader("Time Window Overlap Verification")
        
        # Sample verification for the last case
        last_case = list(cases_data.values())[-1]
        verification = verify_overlapping_windows(last_case["companies"])
        
        verification_df = []
        for company, result in verification.items():
            verification_df.append({
                "Company": company,
                "Total Clients": result["total_clients"],
                "Overlapping Pairs": result["overlap_count"],
                "Has Overlap": "✅" if result["has_overlap"] else "❌"
            })
        
        st.table(verification_df)
        
        # Show example overlapping pairs
        st.subheader("Example Overlapping Pairs (Last Case)")
        for company, result in verification.items():
            if result["overlapping_pairs"]:
                st.write(f"**{company}**: {result['overlapping_pairs'][:3]}...")  # Show first 3 pairs 