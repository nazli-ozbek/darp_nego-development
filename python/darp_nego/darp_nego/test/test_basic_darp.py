import random
import unittest
from darp_nego.darp.basic_darp import BasicDARPClient, BasicDARPProblem, BasicDARPVehicle, SolverConfig

class TestBasicDARPSolver(unittest.TestCase):
    
    def generate_time_matrix(self, size=20, min_time=1, max_time=20):
        random.seed(42)
        time_matrix = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                travel_time = random.randint(min_time, max_time)
                time_matrix[i][j] = travel_time
                time_matrix[j][i] = travel_time
        return time_matrix
    
    def test_basic_darp_solver(self):
        # Create a simple problem with 2 vehicles and 3 clients
        problem = BasicDARPProblem(agent=1)
        
        problem.add_vehicle(BasicDARPVehicle(0, start_location=0, max_volume=10, end_location=0))  
        problem.add_vehicle(BasicDARPVehicle(1, start_location=0, max_volume=15, end_location=0)) 
        
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=1,
            end_location=2,
            early_pickup=0,    # Can be picked up starting at time 0
            late_pickup=30,    # Must be picked up by time 30
            early_drop=10,     # Can be dropped off starting at time 10
            late_drop=40,      # Must be dropped off by time 40
            volume=5
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=1,
            start_location=3,
            end_location=4,
            early_pickup=10,
            late_pickup=40,
            early_drop=20,
            late_drop=50,
            volume=3
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=2,
            start_location=5,
            end_location=6,
            early_pickup=15,
            late_pickup=45,
            early_drop=25,
            late_drop=55,
            volume=4
        ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=120,        # 2 hours in minutes
            MAX_WAITING_TIME=30,       # 30 minutes max wait
            MAX_ROUTE_TIME=120,        # 2 hours max route time
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10       # 10 seconds search time
        )

        road_network = self.generate_time_matrix(max_time=30)
        objective_value = problem.solve_problem(road_network, config)        
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_tight_time_windows(self):
        """Test with very tight time windows between pickups and deliveries"""
        problem = BasicDARPProblem(agent=1)
        
        # Single vehicle with standard capacity
        problem.add_vehicle(BasicDARPVehicle(0, 0, 20, 0))
        
        # Add clients with tight, overlapping time windows
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=1,
            end_location=2,
            early_pickup=0,
            late_pickup=3,    # Very tight pickup window, late pickup penalty
            early_drop=6,     # Minimal time between pickup and delivery
            late_drop=7,      # Late drop penalty
            volume=5
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=1,
            start_location=3,
            end_location=4,
            early_pickup=4,    # Overlaps with first client's delivery
            late_pickup=5,
            early_drop=7,
            late_drop=8,
            volume=5
        ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=60,
            MAX_WAITING_TIME=5,        # Short waiting time allowed
            MAX_ROUTE_TIME=60,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )
        
        road_network = self.generate_time_matrix(min_time=3) 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_max_capacity(self):
        """Test with clients that exactly match vehicle capacity"""
        problem = BasicDARPProblem(agent=1)
        
        # Vehicle with exact capacity for all clients
        problem.add_vehicle(BasicDARPVehicle(0, 0, 10, 0))
        
        # Clients whose total volume equals vehicle capacity
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=1,
            end_location=2,
            early_pickup=0,
            late_pickup=30,
            early_drop=10,
            late_drop=40,
            volume=6        # Large volume
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=1,
            start_location=3,
            end_location=4,
            early_pickup=0,
            late_pickup=30,
            early_drop=10,
            late_drop=40,
            volume=4       # Together with first client = vehicle capacity
        ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=60,
            MAX_WAITING_TIME=15,
            MAX_ROUTE_TIME=60,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )
        
        road_network = self.generate_time_matrix() 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_capacity_slack(self):
        """Test with clients that exactly match vehicle capacity"""
        problem = BasicDARPProblem(agent=1)
        
        problem.add_vehicle(BasicDARPVehicle(0, 0, 5, 0))
        
        # Clients whose total volume equals vehicle capacity
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=1,
            end_location=2,
            early_pickup=0,
            late_pickup=30,
            early_drop=10,
            late_drop=40,
            volume=6        # Large volume
        ))
                
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=60,
            MAX_WAITING_TIME=15,
            MAX_ROUTE_TIME=60,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )
        
        road_network = self.generate_time_matrix() 
        objective_value = problem.solve_problem(road_network, config)
        self.assertIsNone(objective_value)

    def test_single_vehicle_many_clients(self):
        """Test with many clients and only one vehicle"""
        problem = BasicDARPProblem(agent=1)
        
        # Single vehicle with large capacity
        problem.add_vehicle(BasicDARPVehicle(0, 0, 4, 0)) # Small vehicle capacity
        
        # Add many clients with small volumes but overlapping time windows
        for i in range(8):  # 8 clients
            problem.add_client(BasicDARPClient(
                client_id=i,
                start_location=i*2 + 1,
                end_location=i*2 + 2,
                early_pickup=i*5,      # Staggered pickup times
                late_pickup=i*5 + 20,
                early_drop=i*5 + 10,
                late_drop=i*5 + 30,
                volume=2             # Small volume
            ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=1440,   # Longer maximum time
            MAX_WAITING_TIME=1440,
            MAX_ROUTE_TIME=1440,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=20       # Longer search time for complex problem
        )
        
        road_network = self.generate_time_matrix() 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_low_travel_time(self):
        """Test with locations that have zero travel time between them"""
        problem = BasicDARPProblem(agent=1)
        problem.add_vehicle(BasicDARPVehicle(0, 0, 10, 0))
        
        # Clients with same locations (zero travel time)
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=1,
            end_location=1,  # Same as pickup location
            early_pickup=0,
            late_pickup=10,
            early_drop=0,    # Can be dropped off immediately
            late_drop=10,
            volume=3
        ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=30,
            MAX_WAITING_TIME=5,
            MAX_ROUTE_TIME=30,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )
        
        road_network = self.generate_time_matrix(min_time=1, max_time=2) 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_maximum_values(self):
        """
        Test with maximum possible values for time windows and capacity
        Note:In the model, MAX_ROUTE_TIME (Maximum time per vehicle route) is a constraint for the entire route. 
            When a result that does not exceed MAX_ROUTE_TIME is not found, it becomes infeasible. 
            To soften this, the vehicle end node must be droppable.
            There must be balance between time matrix's max value and MAX_ROUTE_TIME
        """
        problem = BasicDARPProblem(agent=1)
        
        # Vehicle with maximum reasonable capacity
        problem.add_vehicle(BasicDARPVehicle(0, 0, 999999, 0))
        
        # Client with maximum time windows
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=1,
            end_location=2,
            early_pickup=0,
            late_pickup=999999, # Very late pickup deadline
            early_drop=0,
            late_drop=999999,   # Very late delivery deadline
            volume=999999       # Maximum volume
        ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=999999,
            MAX_WAITING_TIME=999999,
            MAX_ROUTE_TIME=999999, # Infeasible if this value not bigger than time matrix max value.
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )
        
        size = (len(problem.vehicles) + len(problem.clients)) * 2 
        road_network = self.generate_time_matrix(size=size, min_time=90, max_time=99) 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_minimal_time_windows(self):
        """Test with minimal possible time windows"""
        problem = BasicDARPProblem(agent=1)
        problem.add_vehicle(BasicDARPVehicle(0, 0, 10, 0))
        
        # Clients with exact time windows (no flexibility)
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=1,
            end_location=2,
            early_pickup=10,
            late_pickup=10,    # Must pickup exactly at time 10
            early_drop=20,
            late_drop=20,      # Must deliver exactly at time 20
            volume=5
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=1,
            start_location=3,
            end_location=4,
            early_pickup=30,
            late_pickup=30,    # Must pickup exactly at time 30
            early_drop=40,
            late_drop=40,      # Must deliver exactly at time 40
            volume=5
        ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=60,
            MAX_WAITING_TIME=0,     # No waiting time allowed
            MAX_ROUTE_TIME=60,
            EARLY_ARRIVAL_PENALTY=1000,  # Heavy penalty for early arrival
            LATE_ARRIVAL_PENALTY=1000,    # Heavy penalty for late arrival
            SEARCH_TIME_LIMIT=10
        )
        
        road_network = self.generate_time_matrix(max_time=10) 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_multiple_depot_locations(self):
        """Test with vehicles starting and ending at different locations"""
        problem = BasicDARPProblem(agent=1)
        
        # Vehicles with different start and end locations
        problem.add_vehicle(BasicDARPVehicle(0, 0, 10, 5))    # Starts at 0, ends at 5
        problem.add_vehicle(BasicDARPVehicle(1, 10, 10, 15))  # Starts at 10, ends at 15
        
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=2,
            end_location=4,
            early_pickup=0,
            late_pickup=30,
            early_drop=10,
            late_drop=40,
            volume=5
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=1,
            start_location=12,
            end_location=14,
            early_pickup=0,
            late_pickup=30,
            early_drop=10,
            late_drop=40,
            volume=5
        ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=120,
            MAX_WAITING_TIME=30,
            MAX_ROUTE_TIME=120,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )
        
        road_network = self.generate_time_matrix(max_time=30) 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_sequential_strict_dependencies(self):
        """Test clients that must be served in strict sequence due to time windows"""
        problem = BasicDARPProblem(agent=1)
        problem.add_vehicle(BasicDARPVehicle(0, 0, 20, 0))
        
        # Each client must be completed before the next can start
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=1,
            end_location=2,
            early_pickup=0,
            late_pickup=5,
            early_drop=5,
            late_drop=10,
            volume=5
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=1,
            start_location=3,
            end_location=4,
            early_pickup=10,    # Can only start after first client is delivered
            late_pickup=15,
            early_drop=15,
            late_drop=20,
            volume=5
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=2,
            start_location=5,
            end_location=6,
            early_pickup=20,    # Can only start after second client is delivered
            late_pickup=25,
            early_drop=25,
            late_drop=30,
            volume=5
        ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=60,
            MAX_WAITING_TIME=0,
            MAX_ROUTE_TIME=60,
            EARLY_ARRIVAL_PENALTY=100,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )
        
        road_network = self.generate_time_matrix(max_time=10) 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_maximum_clients_minimum_time(self):
        """Test maximum number of clients with minimum possible time windows"""
        problem = BasicDARPProblem(agent=1)
        
        # Multiple vehicles with small capacity
        for i in range(5):
            problem.add_vehicle(BasicDARPVehicle(i, 0, 2, 0))
        
        # Many clients with tight time windows
        for i in range(20):  # Large number of clients
            problem.add_client(BasicDARPClient(
                client_id=i,
                start_location=i*2 + 1,
                end_location=i*2 + 2,
                early_pickup=i,
                late_pickup=i + 1,    # 1-minute window
                early_drop=i + 1,
                late_drop=i + 2,      # 1-minute window
                volume=1
            ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=30,
            MAX_WAITING_TIME=1,
            MAX_ROUTE_TIME=300,
            EARLY_ARRIVAL_PENALTY=1000,
            LATE_ARRIVAL_PENALTY=1000,
            SEARCH_TIME_LIMIT=10     
        )
        
        road_network = self.generate_time_matrix(size=50, max_time=10) 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_single_location_multiple_clients(self):
        """Test multiple clients at exactly the same location"""
        problem = BasicDARPProblem(agent=1)
        problem.add_vehicle(BasicDARPVehicle(0, 1, 20, 1))  # Start and end at location 1
        
        # Multiple clients at the same locations
        for i in range(5):
            problem.add_client(BasicDARPClient(
                client_id=i,
                start_location=1,     # All pickups at same location
                end_location=1,       # All deliveries at same location
                early_pickup=i*10,    # Staggered times
                late_pickup=i*10 + 5,
                early_drop=i*10 + 5,
                late_drop=i*10 + 10,
                volume=2
            ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=60,
            MAX_WAITING_TIME=5,
            MAX_ROUTE_TIME=60,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )
        
        road_network = self.generate_time_matrix(max_time=30) 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_fractional_volumes(self):
        """Test with very small fractional volumes"""
        problem = BasicDARPProblem(agent=1)
        problem.add_vehicle(BasicDARPVehicle(0, 0, 1, 0))
        
        # Clients with tiny fractional volumes
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=1,
            end_location=2,
            early_pickup=0,
            late_pickup=30,
            early_drop=10,
            late_drop=40,
            volume=0.1    # Very small volume
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=1,
            start_location=3,
            end_location=4,
            early_pickup=0,
            late_pickup=30,
            early_drop=10,
            late_drop=40,
            volume=0.1   # Tiny volume
        ))
        
        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=60,
            MAX_WAITING_TIME=15,
            MAX_ROUTE_TIME=60,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )
        
        road_network = self.generate_time_matrix() 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_large_number_of_vehicles(self):
        """Test with a large number of vehicles to assess scalability"""
        problem = BasicDARPProblem(agent=1)

        # Adding many vehicles
        for i in range(20):  # 20 vehicles
            problem.add_vehicle(BasicDARPVehicle(i, 0, 10, 0))

        # Adding a few clients
        for i in range(2):
            problem.add_client(BasicDARPClient(
                client_id=i,
                start_location=i * 2 + 1,
                end_location=i * 2 + 2,
                early_pickup=0,
                late_pickup=60,
                early_drop=10,
                late_drop=70,
                volume=2
            ))

        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=180,
            MAX_WAITING_TIME=20,
            MAX_ROUTE_TIME=180,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=20
        )

        road_network = self.generate_time_matrix() 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_varying_vehicle_capacities(self):
        """Test with vehicles having different capacities"""
        problem = BasicDARPProblem(agent=1)

        # Vehicles with different capacities
        problem.add_vehicle(BasicDARPVehicle(0, 0, 5, 0))
        problem.add_vehicle(BasicDARPVehicle(1, 0, 10, 0))
        problem.add_vehicle(BasicDARPVehicle(2, 0, 20, 0))

        # Clients with different volume needs
        for i in range(6):
            problem.add_client(BasicDARPClient(
                client_id=i,
                start_location=i * 2 + 1,
                end_location=i * 2 + 2,
                early_pickup=0,
                late_pickup=60,
                early_drop=10,
                late_drop=70,
                volume=(i + 1) * 2
            ))

        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=120,
            MAX_WAITING_TIME=20,
            MAX_ROUTE_TIME=120,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )

        road_network = self.generate_time_matrix() 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_multiple_vehicles_different_depots(self):
        """Test with multiple vehicles starting and ending at different locations"""
        problem = BasicDARPProblem(agent=1)

        # Vehicles at different depots
        problem.add_vehicle(BasicDARPVehicle(0, 0, 10, 5))
        problem.add_vehicle(BasicDARPVehicle(1, 10, 10, 15))
        problem.add_vehicle(BasicDARPVehicle(2, 20, 10, 25))

        # Clients spread across the map
        for i in range(20):
            problem.add_client(BasicDARPClient(
                client_id=i,
                start_location=i * 2 + 1,
                end_location=i * 2 + 2,
                early_pickup=0,
                late_pickup=10,
                early_drop=60,
                late_drop=70,
                volume=3
            ))

        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=120,
            MAX_WAITING_TIME=20,
            MAX_ROUTE_TIME=120,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )

        road_network = self.generate_time_matrix(size=50) 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_tight_capacity_constraints(self):
        """Test where vehicle capacities are barely enough"""
        problem = BasicDARPProblem(agent=1)

        # Vehicle capacity just fits all clients
        problem.add_vehicle(BasicDARPVehicle(0, 0, 10, 0))

        for i in range(5):
            problem.add_client(BasicDARPClient(
                client_id=i,
                start_location=i * 2 + 1,
                end_location=i * 2 + 2,
                early_pickup=0,
                late_pickup=60,
                early_drop=10,
                late_drop=70,
                volume=2
            ))

        config = SolverConfig(
            DEPOT_MAX_TIME_WIN=120,
            MAX_WAITING_TIME=20,
            MAX_ROUTE_TIME=120,
            EARLY_ARRIVAL_PENALTY=10,
            LATE_ARRIVAL_PENALTY=100,
            SEARCH_TIME_LIMIT=10
        )

        road_network = self.generate_time_matrix() 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_no_clients(self):
        """Test a scenario with vehicles but no clients."""
        problem = BasicDARPProblem(agent=1)
        problem.add_vehicle(BasicDARPVehicle(0, 0, 10, 0))
        
        config = SolverConfig()
        objective_value = problem.solve_problem({}, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
        self.assertGreaterEqual(objective_value, 0)

    def test_no_vehicles(self):
        """Test a scenario where clients exist but there are no vehicles."""
        problem = BasicDARPProblem(agent=1)
        problem.add_client(BasicDARPClient(0, 1, 2, 0, 30, 10, 40, 5))
        
        config = SolverConfig()
        objective_value = problem.solve_problem({}, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNone(objective_value)

    def test_vehicle_capacity_exceeded(self):
        """Test a scenario where the total demand exceeds vehicle capacity."""
        problem = BasicDARPProblem(agent=1)
        problem.add_vehicle(BasicDARPVehicle(0, 0, 5, 0))
        
        # Total demand is 6 but vehicle capacity is only 5
        problem.add_client(BasicDARPClient(0, 1, 2, 0, 30, 10, 40, 6))
        
        config = SolverConfig()
        road_network = self.generate_time_matrix() 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNone(objective_value)
    
    def test_infeasible_time_windows(self):
        """
        Test a scenario where a client's pickup and delivery cannot be satisfied within time constraints.
        But it must still find solution with high cost.
        """
        problem = BasicDARPProblem(agent=1)
        problem.add_vehicle(BasicDARPVehicle(0, 0, 10, 0))
        
        # Time windows are too strict
        problem.add_client(BasicDARPClient(0, 1, 2, 0, 5, 10, 15, 3))
        problem.add_client(BasicDARPClient(1, 3, 4, 16, 20, 21, 25, 3))
        
        config = SolverConfig()
        road_network = self.generate_time_matrix(min_time=10) 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNotNone(objective_value)
    
    def test_disconnected_graph(self):
        """
        Test a scenario where clients' locations are disconnected in the time matrix.
        When vehicle's max volume is bigger, model can finding alternative route,  
        it can picking up another client and then deliver from another route. 
        """
        problem = BasicDARPProblem(agent=1)
        problem.add_vehicle(BasicDARPVehicle(0, 0, 5, 0))  
        
        problem.add_client(BasicDARPClient(
            client_id=0,
            start_location=1,
            end_location=2,
            early_pickup=0,    
            late_pickup=30,    
            early_drop=10,     
            late_drop=40,      
            volume=5
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=1,
            start_location=3,
            end_location=4,
            early_pickup=10,
            late_pickup=40,
            early_drop=20,
            late_drop=50,
            volume=3
        ))
        
        problem.add_client(BasicDARPClient(
            client_id=2,
            start_location=5,
            end_location=6,
            early_pickup=15,
            late_pickup=45,
            early_drop=25,
            late_drop=55,
            volume=4
        ))

        # Creating a time matrix filled with 5s, except for disconnected nodes       
        road_network = self.generate_time_matrix() 
        # Disconnect specific nodes by setting them to a large value
        road_network[5][6] = 999999 # clien_id:2's start-end route
        
        config = SolverConfig()
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNone(objective_value)
    
    def test_zero_capacity_vehicle(self):
        """Test a vehicle with zero capacity, which should not be able to serve any client."""
        problem = BasicDARPProblem(agent=1)
        problem.add_vehicle(BasicDARPVehicle(0, 0, 0, 0))  # Zero capacity
        problem.add_client(BasicDARPClient(0, 1, 2, 0, 30, 10, 40, 3))
        
        config = SolverConfig()
        road_network = self.generate_time_matrix() 
        objective_value = problem.solve_problem(road_network, config)
        print(f"Objective Value: {objective_value}")
        self.assertIsNone(objective_value)
    
if __name__ == "__main__":
    unittest.main(verbosity=2)
    # same time window clients