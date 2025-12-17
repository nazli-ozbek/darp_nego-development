from darp_nego import GISRoadNetwork, BasicDARPVehicle, BasicDARPClient, BasicDARPProblem, BasicDARPNegotiationProblem
import json

network = GISRoadNetwork() # open_route_key="5b3ce3597851110001cf62483628fb83525f4c22bf3386c79cab5f2a"
#0 - base 1. Gran Turia
network.add_location(39.46221367592752, -0.4134617915839749)
#1- base 2. Bioparc
network.add_location(39.482620321347234, -0.4189549555363035)
#2- base 3. Torres de Serrano
network.add_location(39.48421019818298, -0.38204776023159587)
#3- hosp 1. Dr. Peset
network.add_location(39.45475845756498, -0.3963814846322243)
#4 - hosp 2. La Fe
network.add_location(39.44579457146099, -0.37998782339657916)
#5 - hosp 3. Hosp General
network.add_location(39.46869642976281, -0.4078030587219635)
#6 - cl1
network.add_location(39.46556562696568, -0.4123520851199856)
#7 - cl2
network.add_location(39.47045226809236, -0.38696765925655685)
#8 - cl3
network.add_location(39.4704688324176, -0.37480115900835415)
#9 - cl 4
network.add_location(39.474874796647065, -0.3613257409773533)
#10 - cl 5
network.add_location(39.47575264342876, -0.3475928307673085)
#Calculate distances
network.compute_travel_times()

# sources = sorted(set(key[0] for key in network._cached_distances))
# destinations = sorted(set(key[1] for key in network._cached_distances))
# print(f"{'':<5}", " ".join(f"{dest:<5}" for dest in destinations))

# for source in sources:
#     row = []
#     for destination in destinations:
#         time = network._cached_distances.get((source, destination), 'N/A')
#         row.append(f"{time:<5}")    
#     print(f"{source:<5}", " ".join(row))

vehicle1 = BasicDARPVehicle(0, start_location=0, end_location=0, max_volume=10)
vehicle2 = BasicDARPVehicle(1, start_location=0, end_location=1, max_volume=5)
vehicle3 = BasicDARPVehicle(2, start_location=1, end_location=1, max_volume=15)
vehicle4 = BasicDARPVehicle(3, start_location=2, end_location=2, max_volume=8)

client1 = BasicDARPClient(0, start_location=7, end_location=5, early_pickup=150, late_pickup=300, early_drop=200, late_drop=800, volume=2)
client2 = BasicDARPClient(1, start_location=10, end_location=4, early_pickup=10, late_pickup=400, early_drop=300, late_drop=760, volume=1)
client3 = BasicDARPClient(2, start_location=9, end_location=3, early_pickup=2, late_pickup=340, early_drop=230, late_drop=800, volume=4)
client4 = BasicDARPClient(3, start_location=8, end_location=4, early_pickup=20, late_pickup=200, early_drop=30, late_drop=600, volume=2)
client5 = BasicDARPClient(4, start_location=6, end_location=5, early_pickup=300, late_pickup=890, early_drop=670, late_drop=900, volume=1)

problem1 = BasicDARPProblem(0, vehicles=[vehicle1, vehicle2, vehicle3], clients=[client1, client2, client3])
problem2 = BasicDARPProblem(1, vehicles=[vehicle4], clients=[client4, client5])


nego_problem = BasicDARPNegotiationProblem(0, road_network=network, agents=[problem1, problem2])


json_obj = nego_problem.to_json()
print(type(json_obj))
#print(json_obj)
f = open("example_instance.json", "w")
json.dump(json_obj, f, indent=2)
f.close()



