import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

GENAI_BACKEND = None
try:  # New SDK
    import google.genai as genai  # type: ignore
    GENAI_BACKEND = "google.genai"
except Exception:  # pragma: no cover - optional dependency
    try:  # Legacy SDK (deprecated)
        import google.generativeai as genai  # type: ignore
        GENAI_BACKEND = "google.generativeai"
    except Exception:
        genai = None

from darp_nego.darp.basic_darp import BasicDARPClient, BasicDARPProblem, BasicDARPVehicle


DEFAULT_TIME_START_MIN = 0
DEFAULT_TIME_END_MIN = 12 * 60


@dataclass
class LLMScenarioConfig:
    num_companies: int
    clients_per_company: Tuple[int, int] = (20, 30)
    vehicles_per_company: Tuple[int, int] = (4, 5)
    num_hospitals: int = 3
    time_start_min: int = DEFAULT_TIME_START_MIN
    time_end_min: int = DEFAULT_TIME_END_MIN
    coordinate_range: Tuple[int, int] = (-10, 10)
    speed_units_per_minute: Tuple[float, float] = (0.4, 0.7)


def build_prompt(config: LLMScenarioConfig, include_matrix: bool = False) -> str:
    total_clients = config.num_companies * random.randint(*config.clients_per_company)
    total_vehicles = config.num_companies * random.randint(*config.vehicles_per_company)
    num_hospitals = max(2, config.num_hospitals)

    vehicle_loc_range = f"0..{total_vehicles - 1}"
    hospital_loc_range = (
        f"{total_vehicles}..{total_vehicles + num_hospitals - 1}"
    )
    client_loc_range = (
        f"{total_vehicles + num_hospitals}..{total_vehicles + num_hospitals + total_clients - 1}"
    )

    schema_extras = ""
    if include_matrix:
        schema_extras = (
            ',\n  "coordinates": { "0": [x,y], "1": [x,y], ... },'
            '\n  "time_matrix": [[...], [...], ...]'
        )

    prompt = f"""
You are generating a multi-company Dial-a-Ride (DARP) scenario for ambulance transportation.

Output must be valid JSON only. Do not wrap in markdown.

Time is measured in minutes from 08:00 to 20:00. Use the range {config.time_start_min}..{config.time_end_min}.

Hard constraints (from the thesis formulation):
- Each request r has pickup location p_r, delivery location d_r, pickup time window [e_p_r, l_p_r], delivery time window [e_d_r, l_d_r], and volume q_r.
- Each vehicle v has capacity Q_v, start depot s_v, end depot e_v (start and end depots must be the same).
- A feasible route must respect capacity constraints at all times.
- Time window constraints must be satisfied at all service points.
- Pickup must occur before its corresponding delivery for each request.
- Flow conservation: each request is picked up and delivered exactly once.
- The dataset must be fully feasible so that all requests can be served by the company’s own vehicles.
- Ensure there is room for negotiation: include cross-company pairs where swapping clients can reduce total cost for both companies (e.g., each company has at least 2 clients whose pickup locations are closer to the other company's depots and within overlapping time windows). This creates mutually beneficial exchanges with lower routing cost after reassignment.

Scenario size constraints:
- Companies: {config.num_companies}
- Per company clients: between {config.clients_per_company[0]} and {config.clients_per_company[1]}
- Per company vehicles: between {config.vehicles_per_company[0]} and {config.vehicles_per_company[1]}

Location ID ranges (global for the case):
- Vehicle depots: {vehicle_loc_range}
- Hospitals (delivery nodes): {hospital_loc_range}
- Client pickup nodes: {client_loc_range}

Rules for locations:
- Each vehicle start_location and end_location must be the same depot ID.
- Each client start_location must be a unique ID from the client pickup node range.
- Each client end_location must be a hospital ID from the hospital range.

Rules for time windows:
- 0 <= early_pickup <= late_pickup <= {config.time_end_min}
- 0 <= early_drop_off <= late_drop_off <= {config.time_end_min}
- late_pickup <= early_drop_off (pickup before delivery).

Output JSON schema:
{{
  "companies": {{
    "company_0": {{"vehicles": [{{...}}], "clients": [{{...}}]}},
    ...
  }}{schema_extras}
}}

Vehicle object:
{{"vehicle_id": int, "start_location": int, "end_location": int, "max_volume": int}}

Client object:
{{"client_id": int, "start_location": int, "end_location": int,
  "early_pickup": int, "late_pickup": int,
  "early_drop_off": int, "late_drop_off": int,
  "volume": int}}
""".strip()

    return prompt


def call_gemini(prompt: str, api_key: str, model_name: str = "gemini-2.0-flash", temperature: float = 0.4) -> str:
    if genai is None:
        raise RuntimeError(
            "Neither google-genai nor google-generativeai is installed. "
            "Install google-genai (preferred) or google-generativeai."
        )

    if GENAI_BACKEND == "google.genai":
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )
        return getattr(response, "text", "") or ""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    )
    return response.text or ""


def extract_json(text: str) -> Dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        return json.loads(match.group(0))


def generate_coordinates(num_locations: int, coord_range: Tuple[int, int], precision: int = 2) -> Dict[int, List[float]]:
    min_c, max_c = coord_range
    coordinates = {}
    for loc_id in range(num_locations):
        x = round(random.uniform(min_c, max_c), precision)
        y = round(random.uniform(min_c, max_c), precision)
        coordinates[int(loc_id)] = [x, y]
    return coordinates


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_time_matrix_from_coords(coordinates: Dict[int, List[float]], speed_units_per_minute: float) -> List[List[int]]:
    ids = sorted(coordinates.keys())
    size = len(ids)
    time_matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j:
                time_matrix[i][j] = 0
            else:
                dist = euclidean_distance(coordinates[ids[i]], coordinates[ids[j]])
                time = max(1, int(dist / speed_units_per_minute))
                time_matrix[i][j] = time
    return time_matrix


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def normalize_company(company: Dict, time_start: int, time_end: int) -> Dict:
    vehicles = company.get("vehicles", []) or []
    clients = company.get("clients", []) or []

    norm_vehicles = []
    for v in vehicles:
        start = _coerce_int(v.get("start_location", 0))
        end = _coerce_int(v.get("end_location", start))
        max_volume = max(1, _coerce_int(v.get("max_volume", 5)))
        norm_vehicles.append(
            {
                "vehicle_id": _coerce_int(v.get("vehicle_id", 0)),
                "start_location": start,
                "end_location": end,
                "max_volume": max_volume,
            }
        )

    norm_clients = []
    for c in clients:
        early_pickup = _clamp(_coerce_int(c.get("early_pickup", time_start)), time_start, time_end)
        late_pickup = _clamp(_coerce_int(c.get("late_pickup", early_pickup)), time_start, time_end)
        early_drop = _clamp(_coerce_int(c.get("early_drop_off", late_pickup)), time_start, time_end)
        late_drop = _clamp(_coerce_int(c.get("late_drop_off", early_drop)), time_start, time_end)

        if late_pickup < early_pickup:
            late_pickup = early_pickup
        if early_drop < late_pickup:
            early_drop = late_pickup
        if late_drop < early_drop:
            late_drop = early_drop

        volume = max(1, _coerce_int(c.get("volume", 1)))

        norm_clients.append(
            {
                "client_id": _coerce_int(c.get("client_id", 0)),
                "start_location": _coerce_int(c.get("start_location", 0)),
                "end_location": _coerce_int(c.get("end_location", 0)),
                "early_pickup": early_pickup,
                "late_pickup": late_pickup,
                "early_drop_off": early_drop,
                "late_drop_off": late_drop,
                "volume": volume,
            }
        )

    return {"vehicles": norm_vehicles, "clients": norm_clients}

def _random_time_window(time_start: int, time_end: int) -> Tuple[int, int, int, int]:
    early_pickup = random.randint(time_start, max(time_start, time_end - 10))
    late_pickup = random.randint(early_pickup, min(time_end, early_pickup + 120))
    early_drop = random.randint(late_pickup, min(time_end, late_pickup + 120))
    late_drop = random.randint(early_drop, min(time_end, early_drop + 120))
    return early_pickup, late_pickup, early_drop, late_drop


def _create_client_stub(client_id: int, time_start: int, time_end: int) -> Dict:
    e_p, l_p, e_d, l_d = _random_time_window(time_start, time_end)
    return {
        "client_id": client_id,
        "start_location": 0,
        "end_location": 0,
        "early_pickup": e_p,
        "late_pickup": l_p,
        "early_drop_off": e_d,
        "late_drop_off": l_d,
        "volume": random.randint(1, 3),
    }


def _create_vehicle_stub(vehicle_id: int) -> Dict:
    return {
        "vehicle_id": vehicle_id,
        "start_location": 0,
        "end_location": 0,
        "max_volume": random.randint(6, 12),
    }


def enforce_company_sizes(companies: Dict[str, Dict], config: LLMScenarioConfig) -> None:
    for company in companies.values():
        vehicles = company.get("vehicles", []) or []
        clients = company.get("clients", []) or []

        if len(vehicles) < config.vehicles_per_company[0]:
            for i in range(len(vehicles), config.vehicles_per_company[0]):
                vehicles.append(_create_vehicle_stub(i))
        if len(vehicles) > config.vehicles_per_company[1]:
            vehicles = vehicles[: config.vehicles_per_company[1]]

        if len(clients) < config.clients_per_company[0]:
            for i in range(len(clients), config.clients_per_company[0]):
                clients.append(_create_client_stub(i, config.time_start_min, config.time_end_min))
        if len(clients) > config.clients_per_company[1]:
            clients = clients[: config.clients_per_company[1]]

        company["vehicles"] = vehicles
        company["clients"] = clients


def reindex_company_entities(company: Dict) -> None:
    for idx, v in enumerate(company["vehicles"]):
        v["vehicle_id"] = idx
    for idx, c in enumerate(company["clients"]):
        c["client_id"] = idx


def enforce_company_constraints(company: Dict, time_start: int, time_end: int) -> Dict:
    vehicles = company["vehicles"]
    clients = company["clients"]

    # Enforce vehicle start/end equality
    for v in vehicles:
        v["end_location"] = v["start_location"]

    if not vehicles:
        return company

    max_vehicle_capacity = max(v["max_volume"] for v in vehicles)

    # Cap client volume to vehicle capacity if needed
    for c in clients:
        if c["volume"] > max_vehicle_capacity:
            c["volume"] = max_vehicle_capacity

    total_volume = sum(c["volume"] for c in clients)
    total_capacity = sum(v["max_volume"] for v in vehicles)

    if total_volume > total_capacity:
        needed = total_volume - total_capacity
        per_vehicle = math.ceil(needed / max(1, len(vehicles)))
        for v in vehicles:
            v["max_volume"] += per_vehicle

    # Ensure time windows are within the 12-hour window
    for c in clients:
        c["early_pickup"] = _clamp(c["early_pickup"], time_start, time_end)
        c["late_pickup"] = _clamp(c["late_pickup"], c["early_pickup"], time_end)
        c["early_drop_off"] = _clamp(c["early_drop_off"], c["late_pickup"], time_end)
        c["late_drop_off"] = _clamp(c["late_drop_off"], c["early_drop_off"], time_end)

    return company


def assign_unique_client_locations(companies: Dict[str, Dict], client_location_pool: List[int]) -> None:
    random.shuffle(client_location_pool)
    pool_iter = iter(client_location_pool)
    for company in companies.values():
        for client in company["clients"]:
            client["start_location"] = next(pool_iter)


def validate_company(company: Dict, time_start: int, time_end: int) -> List[str]:
    errors = []
    vehicles = company.get("vehicles", [])
    clients = company.get("clients", [])

    if not vehicles:
        errors.append("no_vehicles")
    if not clients:
        errors.append("no_clients")

    for v in vehicles:
        if v["start_location"] != v["end_location"]:
            errors.append("vehicle_start_end_mismatch")
        if v["max_volume"] <= 0:
            errors.append("vehicle_capacity_invalid")

    for c in clients:
        if c["early_pickup"] < time_start or c["late_pickup"] > time_end:
            errors.append("pickup_out_of_range")
        if c["early_drop_off"] < time_start or c["late_drop_off"] > time_end:
            errors.append("drop_out_of_range")
        if c["late_pickup"] < c["early_pickup"]:
            errors.append("pickup_window_invalid")
        if c["late_drop_off"] < c["early_drop_off"]:
            errors.append("drop_window_invalid")
        if c["late_pickup"] > c["early_drop_off"]:
            errors.append("pickup_after_drop")
    return errors


def ensure_company_feasible(company: Dict, road_network: List[List[int]]) -> bool:
    vehicles = [BasicDARPVehicle.from_json(v) for v in company["vehicles"]]
    clients = [
        BasicDARPClient(
            c["client_id"],
            c["start_location"],
            c["end_location"],
            c["early_pickup"],
            c["late_pickup"],
            c["early_drop_off"],
            c["late_drop_off"],
            c["volume"],
            start_coordinates=(0.0, 0.0),
            end_coordinates=(0.0, 0.0),
        )
        for c in company["clients"]
    ]
    problem = BasicDARPProblem(len(vehicles), vehicles=vehicles, clients=clients)
    problem.road_network = road_network
    solution_cost = problem.solve_problem()
    return solution_cost is not None


def relax_time_windows(company: Dict, time_start: int, time_end: int) -> None:
    for c in company["clients"]:
        c["early_pickup"] = time_start
        c["late_pickup"] = time_end
        c["early_drop_off"] = time_end
        c["late_drop_off"] = time_end


def generate_case_with_llm(
    api_key: str,
    config: LLMScenarioConfig,
    model_name: str = "gemini-2.0-flash",
    include_matrix: bool = False,
) -> Dict:
    prompt = build_prompt(config, include_matrix=include_matrix)
    response_text = call_gemini(prompt, api_key=api_key, model_name=model_name)
    raw = extract_json(response_text)

    companies = raw.get("companies", {})
    if not companies:
        raise ValueError("LLM output missing 'companies'.")

    normalized = {}
    for company_key, company_data in companies.items():
        normalized[company_key] = normalize_company(
            company_data, config.time_start_min, config.time_end_min
        )

    out = {"companies": normalized}
    if include_matrix:
        if "coordinates" in raw:
            out["coordinates"] = raw["coordinates"]
        if "time_matrix" in raw:
            out["time_matrix"] = raw["time_matrix"]
    return out

def normalize_company_keys(companies: Dict[str, Dict], num_companies: int) -> Dict[str, Dict]:
    ordered = [companies[k] for k in sorted(companies.keys())]
    normalized = {}
    for i in range(num_companies):
        if i < len(ordered):
            normalized[f"company_{i}"] = ordered[i]
        else:
            normalized[f"company_{i}"] = {"vehicles": [], "clients": []}
    return normalized


def _coerce_coordinates(raw_coords: Dict) -> Dict[int, List[float]]:
    coords = {}
    for k, v in raw_coords.items():
        try:
            key = int(k)
        except Exception:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            coords[key] = [float(v[0]), float(v[1])]
    return coords


def _validate_time_matrix(matrix: List[List[int]], size: int) -> bool:
    if not isinstance(matrix, list) or len(matrix) != size:
        return False
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            return False
        if any(not isinstance(x, (int, float)) for x in row):
            return False
    return True


def postprocess_case(raw_case: Dict, config: LLMScenarioConfig) -> Dict:
    companies = normalize_company_keys(raw_case["companies"], config.num_companies)
    enforce_company_sizes(companies, config)
    for company in companies.values():
        reindex_company_entities(company)

    total_vehicles = sum(len(c["vehicles"]) for c in companies.values())
    total_clients = sum(len(c["clients"]) for c in companies.values())
    num_hospitals = max(2, config.num_hospitals)

    total_locations = total_vehicles + num_hospitals + total_clients

    if "coordinates" in raw_case and "time_matrix" in raw_case:
        coordinates = _coerce_coordinates(raw_case["coordinates"])
        time_matrix = raw_case["time_matrix"]
        if len(coordinates) != total_locations or not _validate_time_matrix(time_matrix, total_locations):
            coordinates = generate_coordinates(total_locations, config.coordinate_range)
            speed = random.uniform(*config.speed_units_per_minute)
            time_matrix = generate_time_matrix_from_coords(coordinates, speed)
    else:
        coordinates = generate_coordinates(total_locations, config.coordinate_range)
        speed = random.uniform(*config.speed_units_per_minute)
        time_matrix = generate_time_matrix_from_coords(coordinates, speed)

    # Assign unique client pickup locations globally
    client_loc_start = total_vehicles + num_hospitals
    client_locations = list(range(client_loc_start, client_loc_start + total_clients))
    assign_unique_client_locations(companies, client_locations)

    # Assign vehicle depots globally
    vehicle_locations = list(range(0, total_vehicles))
    v_iter = iter(vehicle_locations)
    for company in companies.values():
        for v in company["vehicles"]:
            v["start_location"] = next(v_iter)
            v["end_location"] = v["start_location"]

    # Assign client drop-off locations from hospital range
    hospital_locations = list(range(total_vehicles, total_vehicles + num_hospitals))
    for company in companies.values():
        for c in company["clients"]:
            c["end_location"] = random.choice(hospital_locations)

    # Enforce constraints and feasibility per company
    for company in companies.values():
        enforce_company_constraints(company, config.time_start_min, config.time_end_min)

    # Try to ensure feasibility with the solver; relax time windows if needed
    for company in companies.values():
        if not ensure_company_feasible(company, time_matrix):
            relax_time_windows(company, config.time_start_min, config.time_end_min)

    return {
        "companies": companies,
        "coordinates": coordinates,
        "time_matrix": time_matrix,
    }


def save_json(folder: str, filename: str, data: Dict) -> str:
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    return file_path
