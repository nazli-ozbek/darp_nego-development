from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from functools import partial

def create_data_model():
    data = {}

    # depot indexes:        0, 7, 8
    # client homes indexes: 1, 2 ,3
    # hospitals indexes:    4, 5(copy of 4), 6
    data["time_matrix"] = [
        [0, 6, 9, 8, 7, 7, 3, 3, 3], 
        [6, 0, 8, 3, 2, 2, 6, 6, 6], 
        [9, 8, 0, 11, 10, 10, 6, 6, 6], 
        [8, 3, 11, 0, 1, 1, 7, 7, 7], 
        [7, 2, 10, 1, 0, 0, 6, 6, 6], 
        [7, 2, 10, 1, 0, 0, 6, 6, 6], 
        [3, 6, 6, 7, 6, 6, 0, 0, 0], 
        [3, 6, 6, 7, 6, 6, 0, 0, 0], 
        [3, 6, 6, 7, 6, 6, 0, 0, 0], 
    ]
    
    # time windows for the locations, requested times for a visit.
    # vehicles must visit a location within its time window.
    # for client's (early pick-up time, late pick-up time)
    data["time_windows"] = [
        (0, 999999),  # depot 0
        (7, 12),      # client 1 home
        (10, 15),     # client 2 home
        (16, 18),     # client 3 home
        (15, 20),  # hospital 4 for client 1 and 2
        (22, 30),  # hospital 4 copy but index is 5
        (20, 30),  # hospital 5 for client 3 but index is 6
        (0, 999999),  # depot 7 
        (0, 999999),  # depot 8 
    ]
    
    # the first entry is index of the pickup location, and the second is the index of the delivery location.
    # clients' start and end locations.
    data["pickups_deliveries"] = [
        [1, 4], # client 1 is going hospital 4, which is index  4
        [2, 5], # client 2 is going hospital 4 copy, which is index 5 
        [3, 6], # client 3 is going hospital 5 which is index 6
    ]

    # client home nodes' capacity assigned to client's volumes for load.
    # hospital nodes' capacity weights are assigned to negative value of client's volume for unload.
    data["demands"] = [0, 2, 1, 2, -2, -1, -2, 0, 0]
    data["vehicle_capacities"] = [1, 4]

    data["num_vehicles"] = 2
    # data["depot"] = 0
    data["starts"] = [8, 0]
    data["ends"] = [0, 7]
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console with slack times and actual waiting time."""
    print(f"Objective: {solution.ObjectiveValue()}")
    time_dimension = routing.GetDimensionOrDie("Time")
    total_time = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            slack_var = time_dimension.SlackVar(index)

            # Get slack values
            slack_min = solution.Min(slack_var) if slack_var is not None else 0
            slack_max = solution.Max(slack_var) if slack_var is not None else 0

            node_id = manager.IndexToNode(index)

            # Calculate allowed waiting time (waiting is only needed if arriving early)
            waiting_status = ""
            if not routing.IsStart(index):
              earliest, latest = data["time_windows"][node_id]  # Extract time window
              arrival_time = solution.Min(time_var)
              min_waiting_time =  earliest - arrival_time
              max_waiting_time =  latest - arrival_time
              waiting_status = f"Waiting Allowed ({min_waiting_time},{max_waiting_time})" if max_waiting_time > 0 else "❌ Not Allowed"
            plan_output += (
                f"{node_id} "
                f"Time({solution.Min(time_var)},{solution.Max(time_var)}) "
                f"Slack({slack_min},{slack_max}) "
                f"{waiting_status} -> \n"
            )
            index = solution.Value(routing.NextVar(index))
        # Process last node
        time_var = time_dimension.CumulVar(index)
        node_id = manager.IndexToNode(index)
        plan_output += (
            f"{node_id} "
            f"Time({solution.Min(time_var)},{solution.Max(time_var)})\n"
        )
        plan_output += f"Time of the route: {solution.Min(time_var)} min\n"
        print(plan_output)
        total_time += solution.Min(time_var)
    print(f"Total time of all routes: {total_time} min")


def create_time_evaluator(data):
    """Returns the travel time between the two nodes."""
    # Pre-calculations if needed.
    _time_matrix = data["time_matrix"]

    def time_callback(manager, from_index, to_index):
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return _time_matrix[from_node][to_node]
    
    return time_callback


def add_time_window_constraints(routing, manager, data, time_evaluator):
    """Add Time windows constraint and define transportation requests"""
    time = "Time"
    routing.AddDimension(
        time_evaluator,
        20,  # allow waiting time
        999999,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time,
    )
    time_dimension = routing.GetDimensionOrDie(time)

    # Add time window constraints for each location except depot.
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for location_idx, time_window in enumerate(data["time_windows"]):
        # if location_idx == data["depot"]:
        if location_idx in list(set(data["starts"] + data["ends"])):
            continue
        # Adjust the penalty cost for late arrival (per minute cost)
        penalty_early = -5
        penalty_late = 10
        index = manager.NodeToIndex(location_idx)
        # time_dimension.CumulVar(index).SetMin(time_window[0])  # Allow early arrival
        time_dimension.SetCumulVarSoftUpperBound(index, time_window[0], penalty_early)
        time_dimension.SetCumulVarSoftUpperBound(index, time_window[1], penalty_late)
        # index = manager.NodeToIndex(location_idx)
        # time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
    # Add time window constraints for each vehicle start node
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
        # routing.AddToAssignment(time_dimension.SlackVar(routing.End(vehicle_id)))

    # Instantiate route start and end times to produce feasible times.
    for i in range(data["num_vehicles"]):
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
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


def create_capacity_evaluator(data):
    """Creates callback to get capacities at each location."""
    _demands = data['demands']

    def capacity_evaluator(manager, from_node):
        """Returns the demand of the current node"""
        return _demands[manager.IndexToNode(from_node)]

    return capacity_evaluator


def add_capacity_constraints(routing, manager, data, demand_evaluator_index):
    """Adds capacity constraint"""
    vehicle_capacities = data['vehicle_capacities']
    capacity = 'Capacity'
    routing.AddDimensionWithVehicleCapacity(
        demand_evaluator_index,
        0,  # null capacity slack
        # max(vehicle_capacities),
        vehicle_capacities,
        True,  # start cumul to zero
        capacity
    )

    # Add Slack for reseting to zero unload depot nodes.
    # e.g. vehicle with load 10/15 arrives at node 1 (depot unload)
    # so we have CumulVar = 10(current load) + -15(unload) + 5(slack) = 0.
    capacity_dimension = routing.GetDimensionOrDie(capacity)
    # Allow to drop reloading nodes with zero cost.
    # for node in [4,5,6,7]:
    #     node_index = manager.NodeToIndex(node)
    #     routing.AddDisjunction([node_index], 0)

    # # Allow to drop regular node with a cost.
    # for node in [1,2,3]:
    #     node_index = manager.NodeToIndex(node)
    #     capacity_dimension.SlackVar(node_index).SetValue(0)
    #     routing.AddDisjunction([node_index], 100_000)



def main():
    
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(
        len(data["time_matrix"]), data["num_vehicles"], data["starts"], data["ends"]#data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)


    # Define weight of each edge
    time_evaluator_index = routing.RegisterTransitCallback(
        partial(create_time_evaluator(data), manager))
    routing.SetArcCostEvaluatorOfAllVehicles(time_evaluator_index)

    # Add Time Window constraint
    add_time_window_constraints(routing, manager, data, time_evaluator_index)

    # Add Capacity constraint
    capacity_evaluator_index = routing.RegisterUnaryTransitCallback(
        partial(create_capacity_evaluator(data), manager))
    add_capacity_constraints(routing, manager, data, capacity_evaluator_index)



    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # )
    # search_parameters.time_limit.FromSeconds(3)


    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    # if solution:
    print_solution(data, manager, routing, solution)


if __name__ == "__main__":
    main()