from typing import Iterable, TypeVar, Optional
from .road_network import GISRoadNetwork
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from functools import partial
import random
import json

class BasicDARPClient:

    def __init__(self, client_id: int, start_location: int, end_location: int, early_pickup: float,
                 late_pickup: float,
                 early_drop: float, late_drop: float, volume: float, start_coordinates: Tuple[float, float], end_coordinates: Tuple[float, float]):
        """
        Creates a client for the Basic Negotiation DARP for health
        :param client_id: An identifier for the customer
        :param start_location: The identifier of a starting location
        :param end_location: The identifier of the end location
        :param early_pickup: The earliest pickup time in seconds
        :param late_pickup: The latest pickup time in seconds
        :param early_drop: The earliest drop off time in seconds
        :param late_drop: The latest drop off time in seconds
        :param volume: The space occupied by the client
        """
        self.client_id = client_id
        self.start_location = start_location
        self.end_location = end_location
        self.early_pickup = early_pickup
        self.late_pickup = late_pickup
        self.early_drop = early_drop
        self.late_drop = late_drop
        self.volume = volume
        self.start_coordinates = start_coordinates
        self.end_coordinates = end_coordinates

    @property
    def client_id(self) -> int:
        return self._client_id

    @client_id.setter
    def client_id(self, new_id: int):
        if new_id < 0:
            raise Exception("Customer IDs start from zero")
        else:
            self._client_id = new_id

    @property
    def start_location(self) -> int:
        return self._start_location

    @start_location.setter
    def start_location(self, value: int):
        if value < 0:
            raise Exception("Valid location IDs start from zero")
        else:
            self._start_location = value

    @property
    def end_location(self) -> int:
        return self._end_location

    @end_location.setter
    def end_location(self, value: int):
        if value < 0:
            raise Exception("Valid location IDs start from zero")
        else:
            self._end_location = value

    @property
    def early_pickup(self) -> float:
        return self._early_pickup

    @early_pickup.setter
    def early_pickup(self, value: float):
        if value >= 0:
            self._early_pickup = value
        else:
            raise Exception("Pickup time should be positive")

    @property
    def late_pickup(self) -> float:
        return self._late_pickup

    @late_pickup.setter
    def late_pickup(self, value: float):
        if value >= 0:
            self._late_pickup = value
        else:
            raise Exception("Pickup time should be positive")

    @property
    def early_drop(self) -> float:
        return self._early_drop

    @early_drop.setter
    def early_drop(self, value: float):
        if value >= 0:
            self._early_drop = value
        else:
            raise Exception("Drop off times should be positive")

    @property
    def late_drop(self) -> float:
        return self._late_drop

    @late_drop.setter
    def late_drop(self, value: float):
        if value >= 0:
            self._late_drop = value
        else:
            raise Exception("Drop off times should be positive")

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float):
        if value >= 0:
            self._volume = value
        else:
            raise Exception("The volume taken by the patient should be at least zero")

    def to_json(self) -> dict:
        """
        Transforms a BasicDARPClient into a JSON object
        :return: A JSON object equivalent to the current BasicDARPClient
        """
        return {
            "client_id": self.client_id,
            "start_location": self.start_location,
            "end_location": self.end_location,
            "early_pickup": self.early_pickup,
            "late_pickup": self.late_pickup,
            "early_drop_off": self.early_drop,
            "late_drop_off": self.late_drop,
            "volume": self.volume
        }

    @classmethod
    def from_json(cls, obj: dict) -> TypeVar('BasicDARPClient'):
        """
        Transforms a JSON object into a BasicDARPClient object
        :param obj: The JSON object
        :return: An equivalent BasicDARPClient
        """
        return BasicDARPClient(obj["client_id"], obj["start_location"],
                                 obj["end_location"],
                                 obj["early_pickup"],
                                 obj["late_pickup"],
                                 obj["early_drop_off"],
                                 obj["late_drop_off"],
                                 obj["volume"])


class BasicDARPVehicle:

    def __init__(self, vehicle_id: int, start_location: int, max_volume: float, end_location: int = None):
        """
        Create a basic vehicle for the negotiation DARP problem
        :param vehicle_id: An index representing the vehicle
        :param start_location: An index representing the starting location for the vehicle
        :param end_location: An index representing the end location for the vehicle. Normally it is the same than start_location
        :param max_volume: The maximum volume that can travel in the vehicle at the same time
        """
        self.vehicle_id = vehicle_id
        self.start_location = start_location
        if end_location is not None:
            self.end_location = end_location
        else:
            self.end_location = start_location
        self.max_volume = max_volume

    @property
    def vehicle_id(self) -> int:
        return self._vehicle_id

    @vehicle_id.setter
    def vehicle_id(self, value: int):
        if value >= 0:
            self._vehicle_id = value
        else:
            raise Exception("The vehicle id should be at least 0")

    @property
    def start_location(self) -> int:
        return self._start_location

    @start_location.setter
    def start_location(self, value: int):
        if value >= 0:
            self._start_location = value
        else:
            raise Exception("Valid location IDs should be positive")

    @property
    def end_location(self) -> int:
        return self._end_location

    @end_location.setter
    def end_location(self, value: int):
        if value >= 0:
            self._end_location = value
        else:
            raise Exception("Valid location IDs should be positive")

    @property
    def max_volume(self) -> float:
        return self._max_volume

    @max_volume.setter
    def max_volume(self, value: float):
        if value >= 0:
            self._max_volume = value
        else:
            raise Exception("Max volumes should be positive")

    def to_json(self) -> dict:
        """
        Transforms a BasicDARPVehicle into a JSON object
        :return: A JSON object equivalent to the current BasicDARPVehicle
        """
        return {
            "vehicle_id": self.vehicle_id,
            "start_location": self.start_location,
            "end_location": self.end_location,
            "max_volume": self.max_volume
        }

    @classmethod
    def from_json(cls, obj: dict) -> TypeVar('BasicDARPVehicle'):
        """
        Creates a BasicDARPVehicle from an equivalent JSON object
        :param obj: The JSON object to transform
        :return: An equivalent BasicDARPVehicle object
        """
        return BasicDARPVehicle(
            obj["vehicle_id"],
            obj["start_location"],
            obj["max_volume"],
            end_location=obj["end_location"]
        )


@dataclass
class SolverConfig:
    """Configuration parameters for the solver"""
    DEPOT_MAX_TIME_WIN: int = 24 * 60   # Default max time window for all nodes
    MAX_WAITING_TIME: int = 24 * 60     # Maximum allowed waiting time between stops
    MAX_ROUTE_TIME: int = 24 * 60       # Maximum time per vehicle route (24 hours)
    EARLY_ARRIVAL_PENALTY: int = 10     # Penalty for arriving before time window
    LATE_ARRIVAL_PENALTY: int = 100     # Penalty for arriving after time window
    SEARCH_TIME_LIMIT: int = 10         # Time limit for local search in seconds


@dataclass
class LocationInfo:
    """Class to store information about a location in the matrix"""

    matrix_index: int    # Index in the matrix
    actual_location: int # Original location ID
    entity_type: str     # 'client' or 'vehicle'
    entity_id: int       # ID of the client or vehicle
    location_type: str   # 'pickup', 'delivery', 'start', 'end'


class LocationMapper:
    """Class to manage mappings between actual locations and matrix indices"""

    def __init__(self, clients: Dict[int, BasicDARPClient], vehicles: Dict[int, BasicDARPVehicle]):
        self.location_map: Dict[int, LocationInfo] = {}  # matrix_index -> LocationInfo
        self.reverse_map: Dict[Tuple[int, str], int] = (
            {}
        )  # (actual_location, entity_type) -> matrix_index
        self._create_mapping(clients, vehicles)
        self.total_locations = len(self.location_map)

    def _create_mapping(self, clients: Dict[int, BasicDARPClient], vehicles: Dict[int, BasicDARPVehicle]) -> None:
        matrix_index = 0

        # Map client pickup locations
        for client_id, client in clients.items():
            self.location_map[matrix_index] = LocationInfo(
                matrix_index=matrix_index,
                actual_location=client.start_location,
                entity_type="client",
                entity_id=client_id,
                location_type="pickup",
            )
            self.reverse_map[(client.start_location, f"client_{client_id}_pickup")] = (
                matrix_index
            )
            matrix_index += 1

        # Map client delivery locations
        for client_id, client in clients.items():
            self.location_map[matrix_index] = LocationInfo(
                matrix_index=matrix_index,
                actual_location=client.end_location,
                entity_type="client",
                entity_id=client_id,
                location_type="delivery",
            )
            self.reverse_map[(client.end_location, f"client_{client_id}_delivery")] = (
                matrix_index
            )
            matrix_index += 1

        # Map vehicle start locations
        for vehicle_id, vehicle in vehicles.items():
            self.location_map[matrix_index] = LocationInfo(
                matrix_index=matrix_index,
                actual_location=vehicle.start_location,
                entity_type="vehicle",
                entity_id=vehicle_id,
                location_type="start",
            )
            self.reverse_map[
                (vehicle.start_location, f"vehicle_{vehicle_id}_start")
            ] = matrix_index
            matrix_index += 1

        # Map vehicle end locations
        for vehicle_id, vehicle in vehicles.items():
            self.location_map[matrix_index] = LocationInfo(
                matrix_index=matrix_index,
                actual_location=vehicle.end_location,
                entity_type="vehicle",
                entity_id=vehicle_id,
                location_type="end",
            )
            self.reverse_map[(vehicle.end_location, f"vehicle_{vehicle_id}_end")] = (
                matrix_index
            )
            matrix_index += 1

    def get_matrix_index(self, actual_location: int, entity_type: str) -> int:
        """
        Example usage:
            get_matrix_index(1, 'client_1_pickup')
            get_matrix_index(20, 'client_1_delivery')
        """
        return self.reverse_map.get((actual_location, entity_type))

    def get_location_info(self, matrix_index: int) -> Optional[LocationInfo]:
        return self.location_map.get(matrix_index)

    def print_mapping(self) -> None:
        """
        Print detailed view of the location mapping including:
        - Matrix indices and their corresponding locations
        - Grouping by entity type (clients and vehicles)
        - Summary statistics
        """
        print("\n=== Location Mapping Details ===")

        # Print Quick Reference Table
        print("\nQuick Reference Table:")
        print("--------------------")
        print(
            "Matrix Index | Entity Type | Entity ID | Location Type | Actual Location"
        )
        print("-" * 75)
        for matrix_idx in range(self.total_locations):
            info = self.location_map[matrix_idx]
            print(
                f"{matrix_idx:^11} | {info.entity_type:^11} | {info.entity_id:^9} | {info.location_type:^13} | {info.actual_location:^15}"
            )
        print()


class BasicDARPProblem:

    def __init__(self, agent: int | TypeVar('BasicDARPProblem'), vehicles: Optional[Iterable[BasicDARPVehicle]] = {},
                 clients: Optional[Iterable[BasicDARPClient]] = {}, road_network: dict = {}, coordinates: dict = {}):
        self._agent_id = 0
        self._road_network = road_network
        if isinstance(agent, int) or isinstance(agent, int):
            self.agent_id = agent
            self.vehicles = {v.vehicle_id: v for v in vehicles} if vehicles else dict()
            self.clients = {c.client_id: c for c in clients} if clients else dict()
            self.road_network = road_network
            self.coordinates = coordinates
        elif isinstance(agent, BasicDARPProblem):
            self.agent_id = agent.agent_id
            self.vehicles = dict(agent.vehicles)
            self.clients = dict(agent.clients)
            self.road_network = agent.road_network
            self.coordinates = agent.coordinates
    def add_vehicle(self, vehicle: BasicDARPVehicle):
        """
        Add vehicle to the DARP problem of the agent
        :param vehicle: The vehicle to be added
        :return:
        """
        self.vehicles[vehicle.vehicle_id] = vehicle

    def remove_vehicle(self, vehicle_id: int):
        """
        Remove a vehicle from the DARP problem of the agent
        :param vehicle_id: The vehicle id to remove
        :return:
        """
        del self.vehicles[vehicle_id]

    def has_vehicle(self, vehicle_id: int) -> bool:
        """
        Checks if the DARP problem contains a specific vehicle
        :param vehicle_id: The vehicle id to seek
        :return: True if the vehicle is associated to the problem, False otherwise
        """
        return vehicle_id in self.vehicles

    def get_vehicle(self, vehicle_id: int) -> BasicDARPVehicle:
        """
        Get the information associated to a specific vehicle
        :param vehicle_id: The specific vehicle to seek
        :return: The information of the vehicle
        """
        return self.vehicles[vehicle_id]

    def add_client(self, client: BasicDARPClient):
        """
        Add a client to the DARP problem
        :param client: The client to be added to the DARP problem
        :return:
        """
        self.clients[client.client_id] = client

    def remove_client(self, client_id: int):
        """
        Remove a client from the DARP problem
        :param client_id: The ID of the client to be removed
        :return:
        """
        if client_id in self.clients:
            print("Client is being removed:", client_id)
            del self.clients[client_id]
        else:
            print("Warning: Tried to remove non-existing client:", client_id)

    def has_client(self, client_id: int) -> bool:
        """
        Checks if the DARP problem as an associated client
        :param client_id: The ID of the client to check
        :return: True if the client is in the problem, False otherwise
        """
        return client_id in self.clients

    def get_client(self, client_id: int) -> BasicDARPClient:
        """
        Get the information associated to a specific client
        :param client_id: the id of the client to
        :return:
        """
        return self.clients[client_id]

    @property
    def agent_id(self):
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value):
        self._agent_id = value
    
    @property
    def road_network(self):
        return self._road_network
        
    @road_network.setter
    def road_network(self, value):
        self._road_network = value

    def to_json(self) -> dict:
        """
        Transform into an equivalent JSON object
        :return: An equivalent JSON object
        """
        return {
            "agent_id": self.agent_id,
            "problem_type": "basic_darp",
            "vehicles": sorted([v.to_json() for v in self.vehicles.values()], key=lambda vehicle: vehicle["vehicle_id"]),
            "clients": sorted([c.to_json() for c in self.clients.values()], key=lambda client: client["client_id"])
        }

    @classmethod
    def from_json(cls, obj: dict) -> TypeVar('BasicDARPProblem'):
        """
        Transforms a JSON object into a BasicDARPProblem
        :param obj: the JSON object
        :return: An equivalent BasicDARPProblem
        """
        agent_id = obj["agent_id"]
        vehicles = [BasicDARPVehicle.from_json(v) for v in obj["vehicles"]]
        clients = [BasicDARPClient.from_json(c) for c in obj["clients"]]
        return BasicDARPProblem(agent_id, vehicles=vehicles, clients=clients)

    def _get_time_matrix(self, mapper: LocationMapper, config: SolverConfig) -> List[List[int]]:
        """
        Create or-tools time matrix with defined road network
        - Same coordinates (x,x) are assigned 0
        """
        size = mapper.total_locations
        time_matrix = [[0 for _ in range(size)] for _ in range(size)]

        for i in range(size):
            for j in range(size):
                if i != j:
                    loc_i = mapper.get_location_info(i)
                    loc_j = mapper.get_location_info(j)

                    if loc_i.actual_location == loc_j.actual_location:
                        time_matrix[i][j] = 0
                    else:
                        time_matrix[i][j] = self.road_network[loc_i.actual_location][loc_j.actual_location]

        return time_matrix

    def _get_pickups_deliveries(self, mapper: LocationMapper) -> List[Tuple[int, int]]:
        """Create list of pickup-delivery pairs using matrix indices"""
        pairs = []
        for client_id in self.clients:
            pickup_idx = mapper.get_matrix_index(
                self.clients[client_id].start_location, f"client_{client_id}_pickup"
            )
            delivery_idx = mapper.get_matrix_index(
                self.clients[client_id].end_location, f"client_{client_id}_delivery"
            )
            pairs.append((pickup_idx, delivery_idx))
        return pairs

    def _get_demands(self, mapper: LocationMapper) -> List[int]:
        """Create list of demands for each location"""
        demands = [0] * mapper.total_locations

        for client_id, client in self.clients.items():
            # Set pickup demand (positive)
            pickup_idx = mapper.get_matrix_index(
                client.start_location, f"client_{client_id}_pickup"
            )
            demands[pickup_idx] = client.volume

            # Set delivery demand (negative)
            delivery_idx = mapper.get_matrix_index(
                client.end_location, f"client_{client_id}_delivery"
            )
            demands[delivery_idx] = -client.volume

        return demands

    def _get_time_windows(self, mapper: LocationMapper) -> List[Tuple[int, int]]:
        """Create list of time windows for each location"""
        # Use a large integer (e.g., 24 hours in minutes = 1440) instead of infinity
        MAX_TIME = 1440  # or any other suitable large integer
        
        # Initialize all time windows with default values
        time_windows = [(0, MAX_TIME)] * mapper.total_locations
        
        for client_id, client in self.clients.items():
            # Set pickup time window
            pickup_idx = mapper.get_matrix_index(
                client.start_location, 
                f'client_{client_id}_pickup'
            )
            time_windows[pickup_idx] = (
                int(client.early_pickup),
                int(client.late_pickup)
            )
            
            # Set delivery time window
            delivery_idx = mapper.get_matrix_index(
                client.end_location, 
                f'client_{client_id}_delivery'
            )
            time_windows[delivery_idx] = (
                int(client.early_drop),
                int(client.late_drop)
            )
        
        return time_windows

    def _get_vehicle_data(self, mapper: LocationMapper) -> Dict:
        """Get vehicle-related data"""
        vehicle_data = {"vehicle_capacities": [], "starts": [], "ends": []}

        for vehicle_id, vehicle in self.vehicles.items():
            vehicle_data["vehicle_capacities"].append(vehicle.max_volume)

            start_idx = mapper.get_matrix_index(
                vehicle.start_location, f"vehicle_{vehicle_id}_start"
            )
            vehicle_data["starts"].append(start_idx)

            end_idx = mapper.get_matrix_index(
                vehicle.end_location, f"vehicle_{vehicle_id}_end"
            )
            vehicle_data["ends"].append(end_idx)

        vehicle_data["num_vehicles"] = len(self.vehicles)

        return vehicle_data

    def _create_data_model(self, config: SolverConfig) -> Dict:
        """Convert DARP data structures into format needed for OR-Tools solver"""

        mapper = LocationMapper(self.clients, self.vehicles)

        data = {
            "location_mapper": mapper,
            "time_matrix": self._get_time_matrix(mapper, config),
            "pickups_deliveries": self._get_pickups_deliveries(mapper),
            "demands": self._get_demands(mapper),
            "time_windows": self._get_time_windows(mapper),
        }
        data.update(self._get_vehicle_data(mapper))

        return data

    def _create_time_evaluator(self, data: Dict):
        """Returns the travel time between the two nodes."""
        _time_matrix = data["time_matrix"]

        def time_callback(manager, from_index, to_index):
            """Convert from routing variable Index to time matrix NodeIndex."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return _time_matrix[from_node][to_node]

        return time_callback

    def _add_time_window_constraint(self, routing: pywrapcp.RoutingModel, 
                                manager: pywrapcp.RoutingIndexManager,
                                data: Dict,
                                time_evaluator,
                                config: SolverConfig) -> None:

        """Add Time windows constraint and define transportation requests"""
        time = "Time"
        routing.AddDimension(
            time_evaluator,
            config.MAX_WAITING_TIME, # allow waiting time
            config.MAX_ROUTE_TIME,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time,
        )
        time_dimension = routing.GetDimensionOrDie(time)

        # Add time window constraints for each location.
        # and 'copy' the slack var in the solution object (aka Assignment) to print it
        for location_idx, time_window in enumerate(data["time_windows"]):
            if location_idx in list(set(data["starts"] + data["ends"])):
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.SetCumulVarSoftLowerBound(index, time_window[0], config.EARLY_ARRIVAL_PENALTY)
            time_dimension.SetCumulVarSoftUpperBound(index, time_window[1], config.LATE_ARRIVAL_PENALTY)
            routing.AddToAssignment(time_dimension.SlackVar(index))

        # Add time window constraints for each vehicle start and end node
        # and 'copy' the slack var in the solution object (aka Assignment) to print it
        for vehicle_id, start_idx in enumerate(list(set(data["starts"]))):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data["time_windows"][start_idx][0], data["time_windows"][start_idx][1]
            )
            routing.AddToAssignment(time_dimension.SlackVar(index))
        for vehicle_id, end_idx in enumerate(list(set(data["ends"]))):
            index = routing.End(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data["time_windows"][end_idx][0], data["time_windows"][end_idx][1]
            )
            # Warning: Slack var is not defined for vehicle's end node

        # Instantiate route start and end times to produce feasible times.
        for i in range(data["num_vehicles"]):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i))
            )
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

        # Define Transportation Requests.
        for request in data["pickups_deliveries"]:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
            )
            routing.solver().Add(
                time_dimension.CumulVar(pickup_index)
                <= time_dimension.CumulVar(delivery_index)
            )

    def _create_capacity_evaluator(self, data: Dict):
        """Creates callback to get capacities at each location."""
        _demands = data["demands"]

        def capacity_evaluator(manager, from_node):
            """Returns the demand of the current node"""
            return _demands[manager.IndexToNode(from_node)]

        return capacity_evaluator

    def _add_capacity_constraints(self, routing: pywrapcp.RoutingModel,
                                manager: pywrapcp.RoutingIndexManager,
                                data: Dict,
                                demand_evaluator,
                                config: SolverConfig) -> None:
        """Adds capacity constraint"""
        vehicle_capacities = data["vehicle_capacities"]
        capacity = "Capacity"
        routing.AddDimensionWithVehicleCapacity(
            demand_evaluator,
            0, # null capacity slack
            vehicle_capacities,
            True,  # start cumul to zero
            capacity,
        )
        routing.GetDimensionOrDie(capacity)
                
    def solve_problem(self, config: SolverConfig = SolverConfig()) -> float:
        """
        TODO: Implementar heurística o metaheurística para resolver el problema
        Solve DARP problem with the agent's information
        :return: The cost of the BasicDARP problem
        """
        if len(self.vehicles) == 0:
            print("Empty vehicles list.")
            #TODO: negotiation must be still running, this agent should be removed from the negotiation
            return None

        data = self._create_data_model(config)

        manager = pywrapcp.RoutingIndexManager(
            len(data["time_matrix"]), data["num_vehicles"], data["starts"], data["ends"]
        )
        routing = pywrapcp.RoutingModel(manager)

        time_evaluator_index = routing.RegisterTransitCallback(
            partial(self._create_time_evaluator(data), manager)
        )
        routing.SetArcCostEvaluatorOfAllVehicles(time_evaluator_index)
        
        self._add_time_window_constraint(routing, manager, data, time_evaluator_index, config)

        capacity_evaluator_index = routing.RegisterUnaryTransitCallback(
            partial(self._create_capacity_evaluator(data), manager)
        )
        self._add_capacity_constraints(routing, manager, data, capacity_evaluator_index, config)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(config.SEARCH_TIME_LIMIT)
        # Allow more iterations in local search
        search_parameters.local_search_operators.use_path_lns = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_tsp_lns = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_inactive_lns = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_relocate = pywrapcp.BOOL_TRUE
        
        solution = routing.SolveWithParameters(search_parameters)

        # Store the solution and related objects for later access by the logger
        self._data_model = data
        self._manager = manager
        self._routing = routing
        self._last_solution = solution

        if solution:
            #self._print_solution(data, manager, routing, solution)
            print(f"Cost: {solution.ObjectiveValue()}, Clients: {list(self.clients.keys())}")
            return solution.ObjectiveValue()
        else:
            #TODO: negotiation must be still running
            return None

    def estimate_client_cost(self, client_id: int) -> float:
        """
        Estimate the cost contribution of a specific client by looking only
        at arcs directly connected to the client's pickup and delivery nodes.
        """
        # Make sure we have a solution to analyze
        if not hasattr(self, '_last_solution') or not self._last_solution:
            print("No solution available. Need to solve the problem first.")
            return None
        
        # Make sure the client exists
        if client_id not in self.clients:
            print(f"Client {client_id} not found in problem")
            return 0
        
        solution = self._last_solution
        manager = self._manager
        routing = self._routing
        data = self._data_model
        
        # Find the matrix indices for client pickup and delivery
        mapper = data['location_mapper']
        pickup_key = f'client_{client_id}_pickup'
        delivery_key = f'client_{client_id}_delivery'
        
        pickup_idx = mapper.get_matrix_index(
            self.clients[client_id].start_location, 
            pickup_key
        )
        delivery_idx = mapper.get_matrix_index(
            self.clients[client_id].end_location, 
            delivery_key
        )
        
        if pickup_idx is None or delivery_idx is None:
            print(f"Could not find matrix indices for client {client_id}")
            return 0
        
        # Find which vehicle serves this client
        assigned_vehicle = None
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                if node_idx == pickup_idx:
                    assigned_vehicle = vehicle_id
                    break
                index = solution.Value(routing.NextVar(index))
            if assigned_vehicle is not None:
                break
            
        if assigned_vehicle is None:
            print(f"Client {client_id} is not served in the current solution")
            return 0
        
        # Find route nodes
        route_nodes = []
        index = routing.Start(assigned_vehicle)
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            route_nodes.append(node_idx)
            index = solution.Value(routing.NextVar(index))
        route_nodes.append(manager.IndexToNode(index))  # Add end node
        
        # Find positions
        try:
            pickup_pos = route_nodes.index(pickup_idx)
            delivery_pos = route_nodes.index(delivery_idx)
        except ValueError:
            print(f"Could not find pickup or delivery node in route for client {client_id}")
            return 0
        
        # Only consider arcs directly connected to client nodes
        client_cost = 0
        
        # Arc to pickup node (if not at start)
        if pickup_pos > 0:
            prev_node = route_nodes[pickup_pos-1]
            from_idx = manager.NodeToIndex(prev_node)
            to_idx = manager.NodeToIndex(pickup_idx)
            try:
                client_cost += routing.GetArcCostForVehicle(from_idx, to_idx, assigned_vehicle)
            except Exception:
                # Just skip if arc doesn't exist
                pass
        
        # Arc from pickup node (if not at end)
        if pickup_pos < len(route_nodes)-1:
            next_node = route_nodes[pickup_pos+1]
            from_idx = manager.NodeToIndex(pickup_idx)
            to_idx = manager.NodeToIndex(next_node)
            try:
                client_cost += routing.GetArcCostForVehicle(from_idx, to_idx, assigned_vehicle)
            except Exception:
                # Just skip if arc doesn't exist
                pass
        
        # Arc to delivery node (if not at start)
        if delivery_pos > 0:
            prev_node = route_nodes[delivery_pos-1]
            from_idx = manager.NodeToIndex(prev_node)
            to_idx = manager.NodeToIndex(delivery_idx)
            try:
                client_cost += routing.GetArcCostForVehicle(from_idx, to_idx, assigned_vehicle)
            except Exception:
                # Just skip if arc doesn't exist
                pass
        
        # Arc from delivery node (if not at end)
        if delivery_pos < len(route_nodes)-1:
            next_node = route_nodes[delivery_pos+1]
            from_idx = manager.NodeToIndex(delivery_idx)
            to_idx = manager.NodeToIndex(next_node)
            try:
                client_cost += routing.GetArcCostForVehicle(from_idx, to_idx, assigned_vehicle)
            except Exception:
                # Just skip if arc doesn't exist
                pass
        
        return client_cost

    def copy_with_solution(self) -> 'BasicDARPProblem':
        """
        Create a deep copy of this BasicDARPProblem instance that includes
        solution-related attributes needed for cost estimation.
        
        Returns:
            A new BasicDARPProblem instance with all solution data preserved
        """
        new_problem = BasicDARPProblem(self)        
        new_problem._last_solution = self._last_solution    
        new_problem._manager = self._manager    
        new_problem._routing = self._routing    
        new_problem._data_model = self._data_model        
        return new_problem

    def _print_solution_summary_table(self, data, manager, routing, solution) -> None:
        """Print solution on console"""
        print('\n=== Solution ===')
        
        time_dimension = routing.GetDimensionOrDie('Time')
        total_time = 0
        total_load = 0
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = f'Route for vehicle {vehicle_id}:\n'
            route_load = 0
            
            # Track the actual route
            actual_route = []
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                location_info = data['location_mapper'].get_location_info(node_index)
                
                # Get time information
                time_var = time_dimension.CumulVar(index)
                time_min = solution.Min(time_var)
                
                # Get load information
                route_load += data['demands'][node_index]
                
                # Add to actual route
                actual_route.append({
                    'actual_loc': location_info.actual_location,
                    'action': f"{location_info.entity_type} {location_info.entity_id} {location_info.location_type}",
                    'time': time_min,
                    'load': route_load
                })
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
            
            # Handle arrival at vehicle end location
            node_index = manager.IndexToNode(index)
            location_info = data['location_mapper'].get_location_info(node_index)
            time_var = time_dimension.CumulVar(index)
            time_min = solution.Min(time_var)
            
            actual_route.append({
                'actual_loc': location_info.actual_location,
                'action': f"{location_info.entity_type} {location_info.entity_id} {location_info.location_type}",
                'time': time_min,
                'load': route_load
            })
            
            # Print the actual route with location numbers
            print(f"\nVehicle {vehicle_id} Route:")
            print("Location | Action | Arrival Time | Load")
            print("-" * 50)
            
            for stop in actual_route:
                print(f"   {stop['actual_loc']:<6} | {stop['action']:<20} | {stop['time']:<11} | {stop['load']}")
            
            # Calculate route statistics
            route_time = actual_route[-1]['time']
            total_time += route_time
            total_load += max(stop['load'] for stop in actual_route)
            
            print(f'\nRoute duration: {route_time} minutes')
            print(f'Route max load: {max(stop["load"] for stop in actual_route)}')
            print("-" * 50)
        
        print(f'\nTotal time of all routes: {total_time} minutes')
        print(f'Total max load of all routes: {total_load}')

    def _print_solution(self, data, manager, routing, solution):
        """Prints solution on console."""
        print("\n=== Solution ===\n")
        mapper = data['location_mapper']
        time_dimension = routing.GetDimensionOrDie("Time")
        total_time = 0
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            info = mapper.get_location_info(index)
            plan_output = f"Route for vehicle {info.entity_id}:\n"
            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                plan_output += (
                    f"{manager.IndexToNode(index)}"
                    f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
                    " -> "
                )
                index = solution.Value(routing.NextVar(index))
            time_var = time_dimension.CumulVar(index)
            plan_output += (
                f"{manager.IndexToNode(index)}"
                f" Time({solution.Min(time_var)},{solution.Max(time_var)})\n"
            )
            plan_output += f"Time of the route: {solution.Min(time_var)}min\n"
            print(plan_output)
            total_time += solution.Min(time_var)
        print(f"Total time of all routes: {total_time}min")
        print(f"Cost: {solution.ObjectiveValue()}")

    def _print_data_model(self, data) -> None:
        """Print the contents of the data model in a readable format"""
        mapper = data['location_mapper']
        
        print("\n=== Data Model Contents ===\n")
        
        print("Pickup-Delivery Pairs:")
        for pickup, delivery in data['pickups_deliveries']:
            pickup_info = mapper.get_location_info(pickup)
            delivery_info = mapper.get_location_info(delivery)
            print(f"Client {pickup_info.entity_id}: {pickup_info.actual_location} → {delivery_info.actual_location} | index({pickup_info.matrix_index},{delivery_info.matrix_index})")
        
        print("\nDemands:")
        for idx, demand in enumerate(data['demands']):
            location_info = mapper.get_location_info(idx)
            print(f"Location {location_info.actual_location} ({location_info.location_type}): {demand}")
        
        print("\nTime Windows:")
        for idx, (early, late) in enumerate(data['time_windows']):
            location_info = mapper.get_location_info(idx)
            print(f"Location {location_info.actual_location} ({location_info.location_type}): [{early}, {late}]")
        
        print("\nVehicle Information:")
        print(f"Number of vehicles: {data['num_vehicles']}")
        print("Vehicle capacities:", data['vehicle_capacities'])
        print("Start locations:", [mapper.get_location_info(idx).actual_location for idx in data['starts']])
        print("End locations:", [mapper.get_location_info(idx).actual_location for idx in data['ends']])

        print("\nTime Matrix:")
        print("Matrix Index | Location Info", end="")
        for j in range(len(data['time_matrix'])):
            print(f" | {j:^5}", end="")
        print()  
        print("-" * (20 + 7 * len(data['time_matrix'])))
        
        for i in range(len(data['time_matrix'])):
            info = mapper.get_location_info(i)
            location_info = f"{info.entity_type[:3]}_{info.entity_id}_{info.location_type[:3]}"
            print(f"{i:^11} | {location_info:<12}", end="")
            
            for j in range(len(data['time_matrix'][i])):
                print(f" | {data['time_matrix'][i][j]:^5}", end="")
            print()
            
        data['location_mapper'].print_mapping()

class BasicDARPNegotiationProblem:

    def __init__(self, problem_id, road_network: Optional[GISRoadNetwork], agents: Optional[Iterable[BasicDARPProblem]]):
        self.problem_id = problem_id
        self.road_network = road_network if road_network else None
        self.agents = {agent.agent_id: agent for agent in agents} if agents else dict()

    @property
    def problem_id(self) -> int:
        return self._problem_id

    @problem_id.setter
    def problem_id(self, value):
        if value >= 0:
            self._problem_id = value
        else:
            raise Exception("The problem ID should be at least zero")

    @property
    def road_network(self) -> GISRoadNetwork:
        return self._road_network if hasattr(self, '_road_network') else None

    @road_network.setter
    def road_network(self, value: GISRoadNetwork):
        if value:
            self._road_network = value
        elif hasattr(self, '_road_network'):
            del self._road_network

    @road_network.deleter
    def road_network(self):
        if hasattr(self, '_road_network'):
            del self._road_network

    def add_basic_darp_problem(self, problem: BasicDARPProblem):
        self.agents[problem.agent_id] = problem

    def has_basic_darp_problem(self, agent_id:int) -> bool:
        return agent_id in self.agents

    def get_basic_darp_problem(self, agent_id: int) -> BasicDARPProblem:
        return self.agents[agent_id]

    def remove_basic_darp_problem(self, agent_id: int):
        del self.agents[agent_id]

    def to_json(self) -> dict:
        """
        Transforms the problem into an equivalent JSON object
        :return:
        """
        return {
            "problem_id": self.problem_id,
            "problem_type": "basic_darp_negotiation",
            "road_network": self.road_network.to_json(),
            "agents": sorted([a.to_json() for a in self.agents.values()], key=lambda a: a["agent_id"])
        }

    @classmethod
    def from_json(cls, obj: dict) -> TypeVar('BasicDARPNegotiationProblem'):
        """
        Creates a BasicDARPNegotiationProblem from a JSON object
        :param obj: The JSON object to parse
        :return: An equivalent BasicDARPProblem
        """
        problem_id = obj["problem_id"]
        network = obj["road_network"]
        agents = [BasicDARPProblem.from_json(a) for a in obj["agents"]]
        problem = BasicDARPNegotiationProblem(problem_id, road_network=network, agents=agents)
        return problem
