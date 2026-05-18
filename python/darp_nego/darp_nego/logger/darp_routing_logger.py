import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RouteInfo:
    """Information about a single vehicle route"""
    vehicle_id: int
    start_time: float
    end_time: float
    duration: float
    waiting_time: float
    capacity: int
    max_load: int
    client_pickups: List[int]
    client_deliveries: List[int]
    stops: int

class DARPRoutingLogger:
    """
    Logger for tracking DARP routing solutions throughout the negotiation process.
    Captures initial, per-round, and final routing solutions.
    """
    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None, log_subdir: str = ""):
        self.start_time = datetime.now()
        
        # Use provided session ID or generate a new one
        self.session_id = session_id or self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Create session-specific directory path
        self.log_dir = os.path.join(log_dir, self.session_id, log_subdir)
        
        self.log_data = {
            "session_id": self.session_id,
            "timestamp": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "initial_routes": {},
            "round_routes": {},
            "final_routes": {}
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def log_darp_solution(self, agent_id: str, phase: str, darp_problem, 
                          round_number: Optional[int] = None) -> Dict:
        """
        Log a solution from a BasicDARPProblem instance.
        
        Args:
            agent_id: ID of the agent who owns the problem
            phase: 'initial', 'round', or 'final'
            darp_problem: BasicDARPProblem instance with a solution
            round_number: Round number (only required for phase='round')
            
        Returns:
            Dictionary with the route data
        """            
        # Extract route data from the DARP problem
        routes_data = self._extract_route_data(agent_id, darp_problem)
        
        # Store in the appropriate section based on phase
        if phase == 'initial':
            self.log_data["initial_routes"][agent_id] = routes_data
        elif phase == 'final':
            self.log_data["final_routes"][agent_id] = routes_data
        elif phase == 'round':
            if round_number is None:
                raise ValueError("Round number is required when phase='round'")
                
            round_key = f"round_{round_number}"
            if round_key not in self.log_data["round_routes"]:
                self.log_data["round_routes"][round_key] = {}
                
            self.log_data["round_routes"][round_key][agent_id] = routes_data
        
        return routes_data
    
    def _extract_route_data(self, agent_id: str, darp_problem) -> Dict:
        """Extract detailed route data from a solved BasicDARPProblem"""

        # Get solution components
        data = darp_problem._data_model
        manager = darp_problem._manager
        routing = darp_problem._routing
        solution = darp_problem._last_solution
        
        # Get time dimension
        time_dimension = routing.GetDimensionOrDie("Time")
        capacity_dimension = routing.GetDimensionOrDie("Capacity")
        
        # Initialize summary metrics
        routes_summary = {
            "agent_id": agent_id,
            "total_vehicles": data["num_vehicles"],
            "vehicles_used": 0,
            "total_time": 0,
            "total_waiting_time": 0,
            "clients_served": set(),
            "routes": [],
            "late_arrivals": [],  # Track late arrivals
            "delayed_clients": {
                "pickup": [],    # List of client IDs with pickup delays
                "delivery": []   # List of client IDs with delivery delays
            },
            "cost": solution.ObjectiveValue()  # Add objective value as cost
        }
                
        # Extract data for each vehicle
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            
            # Skip empty routes
            if solution.Value(routing.NextVar(index)) == routing.End(vehicle_id):
                continue
                
            routes_summary["vehicles_used"] += 1
            
            # Initialize route data
            waiting_time = 0
            stops = 0
            max_load = 0
            client_pickups = []
            client_deliveries = []
            route_late_arrivals = []  # Track late arrivals for this route
            
            # Get start time
            time_var = time_dimension.CumulVar(index)
            start_time = solution.Min(time_var)
            prev_time = start_time  # Track the previous node's departure time
            
            # Process each node in the route
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                stops += 1
                
                # Get time info
                time_var = time_dimension.CumulVar(index)
                current_time_min = solution.Min(time_var)
                current_time_max = solution.Max(time_var)
                
                # Track capacity if available
                if capacity_dimension:
                    capacity_var = capacity_dimension.CumulVar(index)
                    current_load = solution.Max(capacity_var)
                    max_load = max(max_load, current_load)
                
                # Extract client info for this node
                client_id = None
                location_type = None
                if "location_mapper" in data:
                    loc_info = data["location_mapper"].get_location_info(node_index)
                    if loc_info and loc_info.entity_type == "client":
                        client_id = loc_info.entity_id
                        location_type = loc_info.location_type
                        if location_type == "pickup":
                            client_pickups.append(client_id)
                            routes_summary["clients_served"].add(client_id)
                        elif location_type == "delivery":
                            client_deliveries.append(client_id)
                
                # Calculate waiting and late arrival times
                if "time_windows" in data and node_index < len(data["time_windows"]):
                    earliest_service = data["time_windows"][node_index][0]
                    latest_service = data["time_windows"][node_index][1]
                    
                    # If this is not the start node, calculate travel time from previous node
                    if index != routing.Start(vehicle_id):
                        prev_node_index = manager.IndexToNode(prev_index)
                        # Get travel time from time matrix
                        travel_time = data["time_matrix"][prev_node_index][node_index]
                        # Calculate actual arrival time
                        arrival_time = prev_time + travel_time
                        
                        # If vehicle arrives before time window, it waits
                        if arrival_time < earliest_service:
                            node_waiting = earliest_service - arrival_time
                            waiting_time += node_waiting
                        
                        # If vehicle arrives after time window closes, it's a late arrival
                        if arrival_time > latest_service:
                            late_by = arrival_time - latest_service
                            
                            # Create detailed late arrival info
                            client_info = {
                                "node": node_index,
                                "late_by": late_by,
                                "client_id": client_id,
                                "location_type": location_type,
                                "vehicle_id": vehicle_id,
                                "scheduled_arrival": latest_service,
                                "actual_arrival": arrival_time
                            }
                            
                            route_late_arrivals.append(client_info)
                            
                            # Add client ID to the appropriate delayed clients list
                            if client_id is not None:
                                if location_type == "pickup" and client_id not in routes_summary["delayed_clients"]["pickup"]:
                                    routes_summary["delayed_clients"]["pickup"].append(client_id)
                                elif location_type == "delivery" and client_id not in routes_summary["delayed_clients"]["delivery"]:
                                    routes_summary["delayed_clients"]["delivery"].append(client_id)
                
                # Store current time and index for next iteration
                prev_time = solution.Max(time_var)  # Departure time is the max of the time var
                prev_index = index
                
                # Move to next node
                index = solution.Value(routing.NextVar(index))
            
            # Get end time from final node
            time_var = time_dimension.CumulVar(index)
            end_time = solution.Max(time_var)
            
            # Get vehicle capacity from data model
            vehicle_capacity = 0
            if "vehicle_capacities" in data and vehicle_id < len(data["vehicle_capacities"]):
                vehicle_capacity = data["vehicle_capacities"][vehicle_id]
            
            # Calculate route duration based on the Time dimension
            route_duration = end_time - start_time
            
            # Create route info
            route_info = RouteInfo(
                vehicle_id=vehicle_id,
                start_time=start_time,
                end_time=end_time,
                duration=route_duration,
                waiting_time=waiting_time,
                capacity=vehicle_capacity,
                max_load=max_load,
                client_pickups=client_pickups,
                client_deliveries=client_deliveries,
                stops=stops
            )
            
            # Add to summary
            routes_summary["routes"].append(vars(route_info))
            routes_summary["total_time"] += route_duration
            routes_summary["total_waiting_time"] += waiting_time
            
            # Add route's late arrivals to the overall summary
            routes_summary["late_arrivals"].extend(route_late_arrivals)
        
        # Convert client set to list
        routes_summary["clients_served"] = list(routes_summary["clients_served"])
        routes_summary["client_count"] = len(routes_summary["clients_served"])
        
        # Count late arrivals by type
        pickup_delays = 0
        delivery_delays = 0
        for late in routes_summary["late_arrivals"]:
            if late.get("location_type") == "pickup":
                pickup_delays += 1
            elif late.get("location_type") == "delivery":
                delivery_delays += 1
        
        routes_summary["pickup_delays"] = pickup_delays
        routes_summary["delivery_delays"] = delivery_delays
        routes_summary["total_delays"] = len(routes_summary["late_arrivals"])
        
        return routes_summary
    
    def save_logs(self) -> str:
        """Save logs to JSON file and return the file path"""
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create filename with session ID
        json_path = os.path.join(self.log_dir, f"routing_{self.session_id}.json")
        
        # Save JSON log
        with open(json_path, 'w') as f:
            json.dump(self.log_data, f, indent=2)
            
        print(f"Routing logs saved to: {os.path.abspath(json_path)}")
        return json_path 