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

def generate_coordinates(num_locations, x_range=(0, 10), y_range=(0, 10), precision=2):
    coordinates = {}
    for loc_id in range(num_locations):
        x = round(random.uniform(*x_range), precision)
        y = round(random.uniform(*y_range), precision)
        coordinates[int(loc_id)] = [x, y]
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

def generate_clients(num_clients, num_vehicles, num_hospitals, min_vehicle_capacity=5):
    clients = []
    hospital_domain = list(range(num_vehicles, num_vehicles + num_hospitals))
    client_domain = list(range(num_vehicles + num_hospitals, num_vehicles + num_hospitals + num_clients))

    for i in range(num_clients):
        start_location = client_domain[i]
        end_location = random.choice(hospital_domain)

        early_pickup = random.randint(10, 500)
        gap1 = random.randint(0, 20)
        late_pickup = early_pickup + gap1
        gap2 = random.randint(0, 30)
        early_drop = late_pickup + gap2
        gap3 = random.randint(0, 20)
        late_drop = early_drop + gap3

        volume = random.randint(1, min_vehicle_capacity)

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

def distribute_to_companies(vehicles, clients, num_companies):
    companies = {f"company_{h_id}": {"vehicles": [], "clients": []} for h_id in range(num_companies)}

    random.shuffle(vehicles)
    for h_id in range(num_companies):
        companies[f"company_{h_id}"]["vehicles"].append(vehicles.pop())

    for vehicle in vehicles:
        h_id = random.choice(range(num_companies))
        companies[f"company_{h_id}"]["vehicles"].append(vehicle)

    random.shuffle(clients)
    for h_id in range(num_companies):
        companies[f"company_{h_id}"]["clients"].append(clients.pop())

    for client in clients:
        h_id = random.choice(range(num_companies))
        companies[f"company_{h_id}"]["clients"].append(client)

    return companies

# Streamlit UI
st.title("DARP Case Generator with Coordinates")

num_cases = st.number_input("Number of Cases", min_value=1, max_value=100, value=50, step=1)

client_range = st.slider("Total Client Range per Case", 10, 30, (10, 15))
vehicle_range = st.slider("Total Vehicle Range per Case", 7, 15, (7, 8))
hospital_range = st.slider("Total Hospital Range per Case", 2, 10, (2, 10))
company_range = st.slider("Total Company Range per Case", 2, 10, (2, 4))

if st.button("Generate Cases"):
    cases_data = {}

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_name = f"data/{timestamp}"

    for case_index in range(1, num_cases + 1):
        num_clients = random.randint(*client_range)
        num_vehicles = random.randint(*vehicle_range)
        num_hospitals = random.randint(*hospital_range)
        num_companies = random.randint(*company_range)

        total_locations = num_clients + num_vehicles + num_hospitals
        coordinates = generate_coordinates(total_locations, x_range=(-30, 40), y_range=(-40, 30), precision=2)
        speed_units_per_minute = random.uniform(0.4, 0.7)
        time_matrix = generate_time_matrix_from_coords(coordinates, speed_units_per_minute)

        vehicles = generate_vehicles(num_vehicles)
        clients = generate_clients(num_clients, num_vehicles, num_hospitals)

        companies = distribute_to_companies(vehicles, clients, num_companies)

        cases_data[f"case_{case_index}"] = {
            "companies": companies,
            "time_matrix": time_matrix,
            "coordinates": coordinates
        }

    file_path_cases = save_json(folder_name, "company_cases.json", cases_data)

    st.success(f"Cases saved in: {file_path_cases}")
    st.download_button("Download company_cases.json", json.dumps(cases_data, indent=4), file_name="company_cases.json")
    st.json(cases_data)
