import json
import math
import os
import random
import re
import inspect
from dataclasses import dataclass
from typing import Dict, List, Tuple

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

from darp_nego.core.darp.basic_darp import BasicDARPClient, BasicDARPProblem, BasicDARPVehicle


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
    diversity_regime: str = ""


def build_prompt(config: LLMScenarioConfig) -> str:
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

    regime_key = (config.diversity_regime or "").strip().upper()
    regime_map = {
        "A": (
            "Dense & Overlapping (HIGH overlap between companies): "
            "companies serve similar pickup areas, balanced demand across companies, "
            "many pickups concentrated into a few hotspots. Reflect this via aligned time windows, "
            "shared hospitals, and overlapping pickup/drop patterns that enable swaps."
        ),
        "B": (
            "Sparse & Separated (LOW overlap): "
            "companies serve distinct pickup regions, pickups spread out with low clustering, "
            "balanced demand. Reflect this via staggered time windows, more distinct hospital usage, "
            "and fewer swap candidates."
        ),
        "C": (
            "Unbalanced Demand (HIGH inequality): "
            "one company has much higher demand than others. "
            "Even if counts are later clamped, encode imbalance via tighter time windows, "
            "higher volumes, and denser overlapping time windows for the heavy-demand company."
        ),
        "D": (
            "Hotspot Clustering: "
            "for at least one company, create 2+ distinct pickup hotspots (two clusters). "
            "Keep overlap medium. Reflect this via grouped pickup IDs, shared hospitals within clusters, "
            "and clustered time windows."
        ),
        "E": (
            "Geographic Outliers: "
            "most pickups in a main region, but a small fraction are outliers. "
            "Reflect this via a few clients with distinct time windows or hospital choices, "
            "and pickup IDs placed in a small separate band."
        ),
        "F": (
            "Tight Time Windows Stress: "
            "many clients have tight time windows (narrow intervals), still keeping pickup < dropoff "
            "and times within 0..720."
        ),
    }
    regime_text = regime_map.get(regime_key, "Balanced default: moderate overlap and moderate time window tightness.")

    prompt = f"""
You are generating a multi-company Dial-a-Ride (DARP) scenario for ambulance transportation.

The model MUST output valid JSON only.
Output must contain ONLY the "companies" object (no coordinates/time_matrix).
Use integer IDs for start_location/end_location.
Client fields: start_location, end_location, time_window [earliest, latest], volume (use early_pickup/late_pickup and early_drop_off/late_drop_off in JSON).
Vehicle fields: start_location, end_location, capacity (use max_volume in JSON).
Keep times in 0..720.

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

MANDATORY NEGOTIATION STRUCTURE (IMPORTANT):

Ensure that for at least ONE PAIR of companies (A, B), there exist at least TWO clients such that:
- Client c_A belongs to company A, client c_B belongs to company B.
- Swapping c_A and c_B between companies is FEASIBLE for both companies.
- After the swap, each company can serve all its assigned clients within time windows and capacity limits.
- The swap is beneficial in terms of routing effort:
  * c_A’s pickup location is closer to company B’s vehicle depots than to company A’s depots.
  * c_B’s pickup location is closer to company A’s vehicle depots than to company B’s depots.
- c_A and c_B have overlapping pickup time windows and similar volumes.

Diversity Regime:
- Selected regime: {regime_key or "NONE"}
- Instructions: {regime_text}
- Because coordinates are generated later, express diversity primarily via time window patterns, per-company demand patterns, hospital usage patterns, and pickup/drop pairing patterns that enable swaps.
- Use pickup ID groupings (bands or clusters) to indicate hotspots, separation, or outliers, even though coordinates are randomized later.

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
  }}
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


def _init_dspy_lm(api_key: str, model_name: str, temperature: float):
    try:
        import dspy  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "DSPy is not installed. Install dspy to use the OpenRouter backend."
        ) from exc

    lm_cls = None
    if hasattr(dspy, "LM"):
        lm_cls = dspy.LM
    elif hasattr(dspy, "OpenAI"):
        lm_cls = dspy.OpenAI
    else:
        raise RuntimeError("Unsupported DSPy version: missing LM/OpenAI class.")

    sig = inspect.signature(lm_cls)
    kwargs = {}
    normalized_model = model_name.strip()
    if normalized_model.startswith("openrouter/"):
        pass
    elif normalized_model.startswith("google/") or normalized_model.startswith("anthropic/") or normalized_model.startswith("openai/"):
        normalized_model = f"openrouter/{normalized_model}"
    else:
        normalized_model = f"openrouter/{normalized_model}"

    if "model" in sig.parameters:
        kwargs["model"] = normalized_model
    elif "model_name" in sig.parameters:
        kwargs["model_name"] = normalized_model

    # LM accepts **kwargs; always pass these so LiteLLM receives them.
    kwargs.setdefault("api_key", api_key)
    kwargs.setdefault("api_base", OPENROUTER_BASE_URL)

    if "temperature" in sig.parameters:
        kwargs["temperature"] = temperature

    # Optional OpenRouter headers (recommended)
    extra_headers = {}
    referer = os.getenv("OPENROUTER_REFERER") or os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_TITLE")
    if referer:
        extra_headers["HTTP-Referer"] = referer
    if title:
        extra_headers["X-Title"] = title

    if extra_headers:
        if "extra_headers" in sig.parameters:
            kwargs["extra_headers"] = extra_headers
        elif "headers" in sig.parameters:
            kwargs["headers"] = extra_headers

    return lm_cls(**kwargs)


def _coerce_lm_text(response) -> str:
    if isinstance(response, (list, tuple)):
        return str(response[0]) if response else ""
    if isinstance(response, dict):
        if "text" in response:
            return str(response["text"] or "")
        if "output" in response:
            return str(response["output"] or "")
        choices = response.get("choices")
        if choices and isinstance(choices, list):
            first = choices[0]
            if isinstance(first, dict):
                if "text" in first:
                    return str(first["text"] or "")
                message = first.get("message") or {}
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"] or "")
    return str(response or "")


def call_openrouter_dspy(
    prompt: str,
    api_key: str,
    model_name: str = "openrouter/google/gemini-3-flash-preview",
    temperature: float = 0.4,
) -> str:
    lm = _init_dspy_lm(api_key=api_key, model_name=model_name, temperature=temperature)
    response = lm(prompt)
    return _coerce_lm_text(response)


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
    model_name: str = "openrouter/google/gemini-3-flash-preview",
) -> Dict:
    prompt = build_prompt(config)
    response_text = call_openrouter_dspy(prompt, api_key=api_key, model_name=model_name)
    raw = extract_json(response_text)

    companies = raw.get("companies", {})
    if not companies:
        raise ValueError("LLM output missing 'companies'.")

    normalized = {}
    for company_key, company_data in companies.items():
        normalized[company_key] = normalize_company(
            company_data, config.time_start_min, config.time_end_min
        )

    return {"companies": normalized}

def normalize_company_keys(companies: Dict[str, Dict], num_companies: int) -> Dict[str, Dict]:
    ordered = [companies[k] for k in sorted(companies.keys())]
    normalized = {}
    for i in range(num_companies):
        if i < len(ordered):
            normalized[f"company_{i}"] = ordered[i]
        else:
            normalized[f"company_{i}"] = {"vehicles": [], "clients": []}
    return normalized


def postprocess_case(raw_case: Dict, config: LLMScenarioConfig) -> Dict:
    companies = normalize_company_keys(raw_case["companies"], config.num_companies)
    enforce_company_sizes(companies, config)
    for company in companies.values():
        reindex_company_entities(company)

    total_vehicles = sum(len(c["vehicles"]) for c in companies.values())
    total_clients = sum(len(c["clients"]) for c in companies.values())
    num_hospitals = max(2, config.num_hospitals)

    total_locations = total_vehicles + num_hospitals + total_clients

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
