import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RoutingMetricsSummary:
    """Summary of routing metrics for a specific phase (initial, round, final)"""
    # Core metrics
    total_time: float  # Total time across all routes
    total_waiting_time: float  # Total waiting time across all routes
    vehicles_used: int  # Number of vehicles used
    max_route_time: float  # Longest route time
    avg_route_time: float  # Average route time per vehicle used
    cost: float  # Objective value of the solution
    
    # Client metrics
    clients_served: int  # Total number of clients served
    avg_client_wait_time: float  # Average client waiting time
    
    # Delay metrics
    pickup_delays: int  # Number of late arrivals at pickup locations
    delivery_delays: int  # Number of late arrivals at delivery locations
    total_delays: int  # Total number of late arrivals
    delayed_pickup_clients: List[int]  # List of client IDs with pickup delays
    delayed_delivery_clients: List[int]  # List of client IDs with delivery delays
    
    # Efficiency metrics
    vehicle_utilization: float  # Average vehicle utilization (0-1)
    time_utilization: float  # Time spent with clients / total time
    capacity_utilization: float  # Average load / capacity ratio
    
    # Comparison metrics (for final vs initial)
    time_change: float  # Percent change in total time
    vehicles_change: float  # Percent change in vehicles used
    delays_change: float  # Percent change in total delays
    cost_change: float  # Percent change in cost
    
    # New client satisfaction metrics
    client_satisfaction_score: float  # Overall satisfaction score (0-100%)
    time_window_adherence: float  # Percentage of arrivals within time windows
    avg_pickup_delay: float  # Average delay time for pickups
    avg_delivery_delay: float  # Average delay time for deliveries
    avg_ride_duration: float  # Average time clients spend in vehicles
    max_ride_duration: float  # Maximum time any client spends in vehicle
    direct_ride_ratio: float  # Ratio of actual ride times to direct travel times
    
    # New operational metrics
    empty_travel_ratio: float  # Ratio of travel without clients to total travel
    peak_vehicle_usage: float  # Highest percentage of fleet in use at once
    avg_detour_factor: float  # Average detour factor for all clients
    
    def __repr__(self) -> str:
        """Print-friendly representation of metrics summary"""
        delayed_pickups = ", ".join(map(str, self.delayed_pickup_clients)) if self.delayed_pickup_clients else "None"
        delayed_deliveries = ", ".join(map(str, self.delayed_delivery_clients)) if self.delayed_delivery_clients else "None"
        
        return (
            f"Total time: {self.total_time:.1f} minutes\n"
            f"Vehicles used: {self.vehicles_used}\n"
            f"Clients served: {self.clients_served}\n"
            f"Solution cost: {self.cost:.1f}\n"
            f"Max route time: {self.max_route_time:.1f} minutes\n"
            f"Avg route time: {self.avg_route_time:.1f} minutes\n"
            f"Waiting time: {self.total_waiting_time:.1f} minutes\n"
            f"Late arrivals: {self.total_delays} (Pickup: {self.pickup_delays}, Delivery: {self.delivery_delays})\n"
            f"Delayed pickup clients: {delayed_pickups}\n"
            f"Delayed delivery clients: {delayed_deliveries}\n"
            f"Time utilization: {self.time_utilization:.1%}\n"
            f"Capacity utilization: {self.capacity_utilization:.1%}\n"
            f"Client satisfaction score: {self.client_satisfaction_score:.1f}%\n"
            f"Time window adherence: {self.time_window_adherence:.1f}%\n"
            f"Avg pickup delay: {self.avg_pickup_delay:.1f} minutes\n"
            f"Avg delivery delay: {self.avg_delivery_delay:.1f} minutes\n"
            f"Avg ride duration: {self.avg_ride_duration:.1f} minutes\n"
            f"Direct ride ratio: {self.direct_ride_ratio:.2f}\n"
            f"Empty travel ratio: {self.empty_travel_ratio:.1%}\n"
            f"Avg detour factor: {self.avg_detour_factor:.2f}\n"
            f"Time change: {self.time_change:+.1%}\n"
            f"Vehicles change: {self.vehicles_change:+.1%}\n"
            f"Delays change: {self.delays_change:+.1%}\n"
            f"Cost change: {self.cost_change:+.1%}"
        )

class DARPRoutingMetrics:
    """
    Calculates and analyzes metrics from DARP routing solutions.
    
    This class processes routing logs to calculate:
    1. Service efficiency metrics
    2. Vehicle utilization metrics
    3. Client service quality metrics
    4. Comparison between initial and final solutions
    """
    
    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize metrics calculator, optionally with a log file.
        
        Args:
            log_path: Path to routing log JSON file (optional)
        """
        self.log_data = None
        self.agent_metrics = {}
        if log_path:
            self.load_log(log_path)
    
    def load_log(self, log_path: str) -> None:
        """
        Load routing log data from a JSON file.
        
        Args:
            log_path: Path to routing log JSON file
        """
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        with open(log_path, 'r') as f:
            self.log_data = json.load(f)
    
    def calculate_metrics(self) -> Dict[str, Dict[str, RoutingMetricsSummary]]:
        """
        Calculate comprehensive metrics from loaded routing log.
        
        Returns:
            Dictionary of agent metrics by phase
        """
        if not self.log_data:
            raise ValueError("No log data loaded. Call load_log() first.")
        
        # Initialize metrics storage for each agent
        agent_metrics = {}
        
        # Calculate metrics for initial routes
        for agent_id, initial_data in self.log_data["initial_routes"].items():
            if agent_id not in agent_metrics:
                agent_metrics[agent_id] = {}
            
            # Calculate initial metrics (without comparison fields)
            initial_metrics = self._calculate_phase_metrics(initial_data)
            agent_metrics[agent_id]["initial"] = initial_metrics
        
        # Calculate metrics for each round (optional)
        if "round_routes" in self.log_data:
            for round_key, round_data in self.log_data["round_routes"].items():
                for agent_id, agent_round_data in round_data.items():
                    if agent_id not in agent_metrics:
                        agent_metrics[agent_id] = {}
                    
                    if "rounds" not in agent_metrics[agent_id]:
                        agent_metrics[agent_id]["rounds"] = {}
                    
                    # Calculate round metrics with comparison to initial
                    initial_data = self.log_data["initial_routes"].get(agent_id, {})
                    round_metrics = self._calculate_phase_metrics(
                        agent_round_data, 
                        initial_data if initial_data else None
                    )
                    agent_metrics[agent_id]["rounds"][round_key] = round_metrics
        
        # Calculate metrics for final routes
        for agent_id, final_data in self.log_data["final_routes"].items():
            if agent_id not in agent_metrics:
                agent_metrics[agent_id] = {}
            
            # Calculate final metrics with comparison to initial
            initial_data = self.log_data["initial_routes"].get(agent_id, {})
            final_metrics = self._calculate_phase_metrics(
                final_data, 
                initial_data if initial_data else None
            )
            agent_metrics[agent_id]["final"] = final_metrics
        
        self.agent_metrics = agent_metrics
        return agent_metrics
    
    def _calculate_phase_metrics(self, data: Dict, initial_data: Optional[Dict] = None) -> RoutingMetricsSummary:
        """
        Calculate metrics for a specific phase (initial, round, or final).
        
        Args:
            data: Route data for the phase
            initial_data: Optional initial route data for comparison
            
        Returns:
            RoutingMetricsSummary object with calculated metrics
        """
        # Extract core metrics from data
        total_time = data.get("total_time", 0)
        total_waiting_time = data.get("total_waiting_time", 0)
        cost = data.get("cost", 0)  # Extract cost from data
        
        # Count vehicles actually used (with routes)
        vehicles_used = data.get("vehicles_used", 0)
        
        # Calculate route times from the routes data structure
        route_times = [r.get("duration", 0) for r in data.get("routes", []) if r.get("stops", 0) > 0]
        max_route_time = max(route_times) if route_times else 0
        avg_route_time = sum(route_times) / len(route_times) if route_times else 0
        
        # Get the client count
        client_count = data.get("client_count", 0)
        
        # Get delay metrics
        pickup_delays = data.get("pickup_delays", 0)
        delivery_delays = data.get("delivery_delays", 0)
        total_delays = data.get("total_delays", 0)
        
        # Get delayed client IDs
        delayed_clients = data.get("delayed_clients", {})
        delayed_pickup_clients = delayed_clients.get("pickup", [])
        delayed_delivery_clients = delayed_clients.get("delivery", [])
        
        # Calculate client wait time (if needed)
        route_client_wait_times = [r.get("waiting_time", 0) for r in data.get("routes", [])]
        total_client_wait_time = sum(route_client_wait_times)
        avg_client_wait_time = total_client_wait_time / client_count if client_count > 0 else 0
        
        # Vehicle utilization: used vehicles / total vehicles
        vehicle_utilization = vehicles_used / data.get("total_vehicles", 1) if data.get("total_vehicles", 0) > 0 else 0
        
        # Time utilization: productive time / total time
        # Productive time = total time - waiting time
        time_utilization = 1.0
        if total_time > 0:
            time_utilization = (total_time - total_waiting_time) / total_time
        
        # Calculate capacity utilization
        total_capacity = 0
        total_load = 0
        for route in data.get("routes", []):
            if route.get("stops", 0) > 0:
                total_capacity += route.get("capacity", 0)
                total_load += route.get("max_load", 0)
        
        capacity_utilization = total_load / total_capacity if total_capacity > 0 else 0
        
        # Default comparison metrics
        time_change = 0.0
        vehicles_change = 0.0
        delays_change = 0.0
        cost_change = 0.0  # Add cost change default
        
        # Calculate comparison metrics if initial data is provided
        if initial_data:
            initial_time = initial_data.get("total_time", 0)
            initial_vehicles = initial_data.get("vehicles_used", 0)
            initial_delays = initial_data.get("total_delays", 0)
            initial_cost = initial_data.get("cost", 0)  # Get initial cost
            
            # Calculate change - positive values indicate improvement
            if initial_time > 0:
                time_change = (initial_time - total_time) / initial_time
            if initial_vehicles > 0:
                vehicles_change = (initial_vehicles - vehicles_used) / initial_vehicles
            if initial_delays > 0:
                delays_change = (initial_delays - total_delays) / initial_delays
            if initial_cost > 0:
                cost_change = (initial_cost - cost) / initial_cost  # Calculate cost change
        
        # Calculate client satisfaction metrics
        
        # Time window adherence (percentage of arrivals within time windows)
        total_service_points = pickup_delays + delivery_delays + len(data.get("clients_served", [])) * 2
        on_time_services = total_service_points - total_delays
        time_window_adherence = (on_time_services / total_service_points * 100) if total_service_points > 0 else 100
        
        # Calculate average delays
        pickup_delay_times = [late.get("late_by", 0) for late in data.get("late_arrivals", []) 
                             if late.get("location_type") == "pickup"]
        delivery_delay_times = [late.get("late_by", 0) for late in data.get("late_arrivals", []) 
                               if late.get("location_type") == "delivery"]
        
        avg_pickup_delay = sum(pickup_delay_times) / len(pickup_delay_times) if pickup_delay_times else 0
        avg_delivery_delay = sum(delivery_delay_times) / len(delivery_delay_times) if delivery_delay_times else 0
        
        # Client ride durations and direct ride ratios
        client_ride_times = {}
        direct_ride_ratios = {}
        
        # Calculate client ride times from routes
        for route in data.get("routes", []):
            route_pickups = {c: None for c in route.get("client_pickups", [])}
            route_deliveries = {}
            
            # Extract pickup and delivery times from the route
            for stop_idx, stop in enumerate(route.get("stops_detail", [])):
                client_id = stop.get("client_id")
                if not client_id:
                    continue
                    
                if stop.get("location_type") == "pickup" and client_id in route_pickups:
                    route_pickups[client_id] = stop.get("departure_time", 0)
                elif stop.get("location_type") == "delivery":
                    route_deliveries[client_id] = stop.get("arrival_time", 0)
            
            # Calculate ride times for each client
            for client_id, pickup_time in route_pickups.items():
                if client_id in route_deliveries and pickup_time is not None:
                    delivery_time = route_deliveries[client_id]
                    ride_time = delivery_time - pickup_time
                    client_ride_times[client_id] = ride_time
                    
                    # Calculate direct travel time if we have a time matrix
                    if "direct_travel_times" in data:
                        direct_time = data["direct_travel_times"].get(client_id, ride_time)
                        direct_ride_ratios[client_id] = ride_time / direct_time if direct_time > 0 else 1
        
        avg_ride_duration = sum(client_ride_times.values()) / len(client_ride_times) if client_ride_times else 0
        max_ride_duration = max(client_ride_times.values()) if client_ride_times else 0
        direct_ride_ratio = sum(direct_ride_ratios.values()) / len(direct_ride_ratios) if direct_ride_ratios else 1
        
        # Calculate empty travel ratio
        total_travel_time = sum([r.get("duration", 0) - r.get("waiting_time", 0) for r in data.get("routes", [])])
        client_travel_time = sum(client_ride_times.values())
        empty_travel_ratio = 1 - (client_travel_time / total_travel_time) if total_travel_time > 0 else 0
        
        # Calculate detour factor (increase over direct path)
        detour_factors = [ratio - 1 for ratio in direct_ride_ratios.values()]
        avg_detour_factor = sum(detour_factors) / len(detour_factors) if detour_factors else 0
        
        # Calculate overall client satisfaction score (weighted average of metrics)
        # Higher is better (0-100%)
        time_weight = 0.4
        delay_weight = 0.4
        detour_weight = 0.2
        
        time_score = time_window_adherence
        delay_score = 100 - min(100, (avg_pickup_delay + avg_delivery_delay))
        detour_score = 100 - min(100, avg_detour_factor * 100)
        
        client_satisfaction_score = (
            time_weight * time_score +
            delay_weight * delay_score +
            detour_weight * detour_score
        )
        
        # Calculate peak vehicle usage
        peak_vehicle_usage = vehicles_used / data.get("total_vehicles", 1) if data.get("total_vehicles", 0) > 0 else 0
        
        # Create metrics summary with the new metrics
        return RoutingMetricsSummary(
            total_time=total_time,
            total_waiting_time=total_waiting_time,
            vehicles_used=vehicles_used,
            max_route_time=max_route_time,
            avg_route_time=avg_route_time,
            cost=cost,  # Include cost in metrics
            clients_served=client_count,
            avg_client_wait_time=avg_client_wait_time,
            pickup_delays=pickup_delays,
            delivery_delays=delivery_delays,
            total_delays=total_delays,
            delayed_pickup_clients=delayed_pickup_clients,
            delayed_delivery_clients=delayed_delivery_clients,
            vehicle_utilization=vehicle_utilization,
            time_utilization=time_utilization,
            capacity_utilization=capacity_utilization,
            time_change=time_change,
            vehicles_change=vehicles_change,
            delays_change=delays_change,
            cost_change=cost_change,  # Include cost change in metrics
            client_satisfaction_score=client_satisfaction_score,
            time_window_adherence=time_window_adherence,
            avg_pickup_delay=avg_pickup_delay,
            avg_delivery_delay=avg_delivery_delay,
            avg_ride_duration=avg_ride_duration,
            max_ride_duration=max_ride_duration,
            direct_ride_ratio=direct_ride_ratio,
            empty_travel_ratio=empty_travel_ratio,
            peak_vehicle_usage=peak_vehicle_usage,
            avg_detour_factor=avg_detour_factor
        )
    
    def plot_time_comparison(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Create a bar chart comparing total times across agents.
        
        Args:
            save_path: Path to save the plot, if None the plot won't be saved
            show: Whether to show the plot
        """
        agents = list(self.agent_metrics.keys())
        
        initial_times = []
        final_times = []
        
        for agent_id in agents:
            phases = self.agent_metrics[agent_id]
            if "initial" in phases:
                initial_times.append(phases["initial"].total_time)
            else:
                initial_times.append(0)
            
            if "final" in phases:
                final_times.append(phases["final"].total_time)
            else:
                final_times.append(0)
        
        x = np.arange(len(agents))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        initial_bars = ax.bar(x - width/2, initial_times, width, label='Initial', color='#3274A1')
        final_bars = ax.bar(x + width/2, final_times, width, label='Final', color='#E1812C')
        
        ax.set_title('Total Time Before and After Negotiation')
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Time (minutes)')
        ax.set_xticks(x)
        ax.set_xticklabels(agents)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Add improvement labels
        for i, (initial, final) in enumerate(zip(initial_times, final_times)):
            if initial > 0:
                change = (initial - final) / initial * 100
                if abs(change) > 0.1:  # Only show label if change is meaningful
                    ax.annotate(f"{change:.1f}%", xy=(i, max(initial, final) + 20),
                                ha='center', va='bottom', fontweight='bold',
                                color='green' if change > 0 else 'red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
    
    def plot_vehicles_used(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Plot comparison of vehicles used before and after negotiation.
        
        Args:
            save_path: Optional path to save the plot image
            show: Whether to show the plot
        """
        if not self.agent_metrics:
            self.calculate_metrics()
            
        agent_ids = list(self.agent_metrics.keys())
        initial_vehicles = []
        final_vehicles = []
        
        for agent_id in agent_ids:
            metrics = self.agent_metrics[agent_id]
            if "initial" in metrics:
                initial_vehicles.append(metrics["initial"].vehicles_used)
            else:
                initial_vehicles.append(0)
                
            if "final" in metrics:
                final_vehicles.append(metrics["final"].vehicles_used)
            else:
                final_vehicles.append(0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(agent_ids))
        width = 0.35
        
        ax.bar(x - width/2, initial_vehicles, width, label='Initial')
        ax.bar(x + width/2, final_vehicles, width, label='Final')
        
        ax.set_title('Vehicles Used Before and After Negotiation')
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Number of Vehicles')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_ids)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
            
    def plot_utilization_comparison(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Plot comparison of vehicle utilization before and after negotiation.
        
        Args:
            save_path: Optional path to save the plot image
            show: Whether to show the plot
        """
        if not self.agent_metrics:
            self.calculate_metrics()
            
        agent_ids = list(self.agent_metrics.keys())
        initial_util = []
        final_util = []
        
        for agent_id in agent_ids:
            metrics = self.agent_metrics[agent_id]
            if "initial" in metrics:
                initial_util.append(metrics["initial"].capacity_utilization * 100)  # To percentage
            else:
                initial_util.append(0)
                
            if "final" in metrics:
                final_util.append(metrics["final"].capacity_utilization * 100)  # To percentage
            else:
                final_util.append(0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(agent_ids))
        width = 0.35
        
        ax.bar(x - width/2, initial_util, width, label='Initial')
        ax.bar(x + width/2, final_util, width, label='Final')
        
        ax.set_title('Capacity Utilization Before and After Negotiation')
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Capacity Utilization (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_ids)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics_over_rounds(self, agent_id: str, metric: str, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Plot the evolution of a specific metric over negotiation rounds.
        
        Args:
            agent_id: ID of the agent to plot
            metric: Metric to plot ('total_time', 'vehicles_used', etc.)
            save_path: Optional path to save the plot image
            show: Whether to show the plot
        """
        if not self.agent_metrics:
            self.calculate_metrics()
            
        if agent_id not in self.agent_metrics:
            raise ValueError(f"No data for agent {agent_id}")
            
        metrics = self.agent_metrics[agent_id]
            
        # Get valid attributes from RoutingMetricsSummary
        valid_metrics = [attr for attr in dir(RoutingMetricsSummary) 
                        if not attr.startswith('_') and not callable(getattr(RoutingMetricsSummary, attr))]
        
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Valid options are: {valid_metrics}")
        
        # Extract round numbers and metric values
        rounds = []
        values = []
        
        # Add initial value if available
        if "initial" in metrics:
            rounds.append(0)
            values.append(getattr(metrics["initial"], metric))
        
        # Add final value if available
        if "final" in metrics:
            rounds.append(max(rounds) + 1 if rounds else 1)
            values.append(getattr(metrics["final"], metric))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, values, marker='o')
        plt.title(f'{metric} Over Negotiation Rounds for Agent {agent_id}')
        plt.xlabel('Round')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
    
    def plot_delays_comparison(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Create a bar chart comparing late arrivals across agents.
        
        Args:
            save_path: Path to save the plot, if None the plot won't be saved
            show: Whether to show the plot
        """
        agents = list(self.agent_metrics.keys())
        
        initial_delays = []
        final_delays = []
        
        for agent_id in agents:
            phases = self.agent_metrics[agent_id]
            if "initial" in phases:
                initial_delays.append(phases["initial"].total_delays)
            else:
                initial_delays.append(0)
            
            if "final" in phases:
                final_delays.append(phases["final"].total_delays)
            else:
                final_delays.append(0)
        
        x = np.arange(len(agents))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, initial_delays, width, label='Initial', color='#ff9999')
        ax.bar(x + width/2, final_delays, width, label='Final', color='#66b3ff')
        
        ax.set_title('Late Arrivals Comparison by Agent')
        ax.set_xlabel('Agent')
        ax.set_ylabel('Number of Late Arrivals')
        ax.set_xticks(x)
        ax.set_xticklabels(agents)
        ax.legend()
        
        # Add improvement labels
        for i, (initial, final) in enumerate(zip(initial_delays, final_delays)):
            if initial > 0:
                change = (initial - final) / initial * 100
                ax.annotate(f"{change:.1f}%", xy=(i, max(initial, final) + 0.5),
                            ha='center', va='bottom', fontweight='bold',
                            color='green' if change > 0 else 'red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
    
    def plot_cost_comparison(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Create a bar chart comparing solution costs across agents.
        
        Args:
            save_path: Path to save the plot, if None the plot won't be saved
            show: Whether to show the plot
        """
        agents = list(self.agent_metrics.keys())
        
        initial_costs = []
        final_costs = []
        
        for agent_id in agents:
            phases = self.agent_metrics[agent_id]
            if "initial" in phases:
                initial_costs.append(phases["initial"].cost)
            else:
                initial_costs.append(0)
                
            if "final" in phases:
                final_costs.append(phases["final"].cost)
            else:
                final_costs.append(0)
        
        x = np.arange(len(agents))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, initial_costs, width, label='Initial', color='#3274A1')
        ax.bar(x + width/2, final_costs, width, label='Final', color='#E1812C')
        
        ax.set_title('Solution Cost Before and After Negotiation')
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Cost')
        ax.set_xticks(x)
        ax.set_xticklabels(agents)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Add improvement labels
        for i, (initial, final) in enumerate(zip(initial_costs, final_costs)):
            if initial > 0:
                change = (initial - final) / initial * 100
                if abs(change) > 0.1:  # Only show label if change is meaningful
                    ax.annotate(f"{change:.1f}%", xy=(i, max(initial, final) + (max(initial, final) * 0.05)),
                                ha='center', va='bottom', fontweight='bold',
                                color='green' if change > 0 else 'red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
    
    def plot_client_satisfaction(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Create a radar chart comparing client satisfaction metrics across agents.
        
        Args:
            save_path: Path to save the plot, if None the plot won't be saved
            show: Whether to show the plot
        """
        agents = list(self.agent_metrics.keys())
        
        # Extract satisfaction metrics for each agent
        time_window_scores = []
        delay_scores = []
        ride_time_scores = []
        detour_scores = []
        
        for agent_id in agents:
            final_metrics = self.agent_metrics.get(agent_id, {}).get("final")
            if not final_metrics:
                continue
            
            time_window_scores.append(final_metrics.time_window_adherence)
            
            # Invert delays so lower is better (0-100 scale)
            delay_score = 100 - min(100, (final_metrics.avg_pickup_delay + final_metrics.avg_delivery_delay))
            delay_scores.append(delay_score)
            
            # Ride time score based on ride duration
            ride_score = 100 - min(100, (final_metrics.avg_ride_duration / 60) * 10)  # Scale to percentage
            ride_time_scores.append(ride_score)
            
            # Detour score based on deviation from direct path
            detour_score = 100 - min(100, final_metrics.avg_detour_factor * 100)
            detour_scores.append(detour_score)
        
        # Create the radar chart
        labels = ['Time Window\nAdherence', 'On-Time\nService', 'Ride\nDuration', 'Direct Path\nEfficiency']
        num_agents = len(agents)
        
        # Create angle list
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        for i, agent_id in enumerate(agents):
            values = [time_window_scores[i], delay_scores[i], ride_time_scores[i], detour_scores[i]]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Agent {agent_id}")
            ax.fill(angles, values, alpha=0.1)
        
        # Set chart properties
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        
        # Add radial grid lines
        ax.set_rticks([0, 20, 40, 60, 80, 100])
        ax.set_rlabel_position(0)
        ax.set_title("Client Satisfaction Metrics Comparison")
        
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
    
    def save_metrics(self, output_dir: str = "metrics", run_id: Optional[str] = None) -> Dict:
        """
        Save metrics to files and generate plots.
        
        Args:
            output_dir: Directory to save metrics in
            run_id: ID for this metrics run, defaults to timestamp
            
        Returns:
            Dictionary with paths to saved files
        """
        if not self.agent_metrics:
            self.calculate_metrics()
            
        # Create unique run ID if not provided
        if not run_id:
            from datetime import datetime
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create output directory - use run_id as the session folder
        run_dir = os.path.join(output_dir, run_id, "routing")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save summary to JSON
        summary_path = os.path.join(run_dir, "routing_metrics.json")
        with open(summary_path, 'w') as f:
            # Convert dataclass objects to dictionaries for JSON serialization
            json_data = {}
            for agent_id, phases in self.agent_metrics.items():
                json_data[agent_id] = {}
                for phase, metrics in phases.items():
                    if phase == "rounds":
                        json_data[agent_id][phase] = {}
                        for round_key, round_metrics in metrics.items():
                            json_data[agent_id][phase][round_key] = vars(round_metrics)
                    else:
                        json_data[agent_id][phase] = vars(metrics)
            
            json.dump(json_data, f, indent=2)
        
        # Save summary to text
        summary_txt_path = os.path.join(run_dir, "routing_metrics.txt")
        with open(summary_txt_path, 'w') as f:
            f.write("=== DARP Routing Metrics ===\n\n")
            
            for agent_id, phases in self.agent_metrics.items():
                f.write(f"Agent: {agent_id}\n")
                f.write("=" * 50 + "\n")
                
                if "initial" in phases:
                    f.write("Initial Routes:\n")
                    f.write(str(phases["initial"]) + "\n\n")
                
                if "final" in phases:
                    f.write("Final Routes:\n")
                    f.write(str(phases["final"]) + "\n\n")
                
                f.write("-" * 50 + "\n\n")
        
        # Save plots
        time_plot_path = os.path.join(run_dir, "time_comparison.png")
        self.plot_time_comparison(save_path=time_plot_path, show=False)
        
        vehicles_plot_path = os.path.join(run_dir, "vehicles_used.png")
        self.plot_vehicles_used(save_path=vehicles_plot_path, show=False)
        
        utilization_plot_path = os.path.join(run_dir, "utilization_comparison.png")
        self.plot_utilization_comparison(save_path=utilization_plot_path, show=False)
        
        delays_plot_path = os.path.join(run_dir, "delays_comparison.png")
        self.plot_delays_comparison(save_path=delays_plot_path, show=False)
        
        cost_plot_path = os.path.join(run_dir, "cost_comparison.png")
        self.plot_cost_comparison(save_path=cost_plot_path, show=False)
        
        # Create round metrics plots for each agent
        round_plots = {}
        for agent_id in self.agent_metrics:
            # Plot time over rounds
            time_rounds_path = os.path.join(run_dir, f"{agent_id}_time_rounds.png")
            try:
                self.plot_metrics_over_rounds(agent_id, "total_time", save_path=time_rounds_path, show=False)
                round_plots[f"{agent_id}_time"] = time_rounds_path
            except Exception as e:
                print(f"Could not create time rounds plot for {agent_id}: {e}")
            
            # Plot vehicle utilization over rounds
            util_rounds_path = os.path.join(run_dir, f"{agent_id}_utilization_rounds.png")
            try:
                self.plot_metrics_over_rounds(agent_id, "capacity_utilization", save_path=util_rounds_path, show=False)
                round_plots[f"{agent_id}_utilization"] = util_rounds_path
            except Exception as e:
                print(f"Could not create utilization rounds plot for {agent_id}: {e}")
        
        # Add new plot
        client_satisfaction_path = os.path.join(run_dir, "client_satisfaction.png")
        self.plot_client_satisfaction(save_path=client_satisfaction_path, show=False)
        
        print(f"\nRouting metrics saved to: {os.path.abspath(run_dir)}")
        
        return {
            "directory": run_dir,
            "summary_json": summary_path,
            "summary_txt": summary_txt_path,
            "time_plot": time_plot_path,
            "vehicles_plot": vehicles_plot_path,
            "utilization_plot": utilization_plot_path,
            "delays_plot": delays_plot_path,
            "cost_plot": cost_plot_path,
            **round_plots,
            "client_satisfaction_plot": client_satisfaction_path
        } 