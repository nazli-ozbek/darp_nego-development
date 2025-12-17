from typing import Iterable, Optional, TypeVar
from itertools import product
from openrouteservice import Client
from openrouteservice.distance_matrix import distance_matrix
from openrouteservice.directions import directions

from darp_nego.utils import GeoTools


class GISRoadNetwork:

    def __init__(self):
        """
        Basic class constructor for GISRoadNetwork
        TODO: I do not think we need our own ORS server yet. This may change :(
        """
        self._id_to_coord = dict()
        self._location_coord = set()
        self._cached_distances = dict()
        self._geo_tools = GeoTools()

    def get_number_locations(self) -> int:
        """
        Returns the number of locations in the road network
        :return: The number of locations
        """
        return len(self._location_coord)

    def add_location(self, latitude: float, longitude: float) -> bool:
        """
        Add a new location to the network based on latitude and longitude
        :param latitude: latitude of the point of interest
        :param longitude: longitude of the point of interest
        :return: True if the location was added False if it already existed
        """
        added = True
        if not (latitude, longitude) in self._location_coord:
            next_id = len(self._location_coord)
            self._id_to_coord[next_id] = (latitude, longitude)
            self._location_coord.add((latitude, longitude))
        else:
            added = False
        return added

    def _location_id_to_ors_coords(self, location: int) -> (float, float):
        """
        Internal method that internally translates between an internal location id and
        coordinates in ORS format (i.e., (lng, lat))
        :param location: The id of the location
        :return: A tuple with the first element being the longitude and the second being the latitude of the location
        """
        if location < 0 or location >= len(self._location_coord):
            raise Exception("Invalid location id {}".format(location))
        lat, lng = self._id_to_coord[location]
        return lng, lat

    def add_travel_time(self, source: int, destination: int, time: float) -> None:
        """
        Adds travel time of the shortest route between source and destination
        :param source: A valid location identifier acting as source
        :param destination: A valid location identifier acting as destination
        :param time: The travel time of the best path between source and destination
        :return:
        """
        if source < 0 or source >= len(self._location_coord):
            raise Exception("Invalid source location id {}".format(source))
        if destination < 0 or destination >= len(self._location_coord):
            raise Exception("Invalid destination location id {}".format(destination))
        self._cached_distances[source, destination] = time

    def compute_travel_times(self) -> None:
        """
        Computes the travel time between all of the pairs of locations in the current
        road network. The API key must be set in order to call ORS
        :return:
        """
        last_id = len(self._location_coord)
        durations = self._geo_tools.get_distance_matrix([self._location_id_to_ors_coords(i) for i in range(last_id)])

        for i in range(last_id):
            for j in range(last_id):
                travel_time_seconds = durations[i][j]
                self.add_travel_time(i, j, travel_time_seconds)

    def get_travel_time(self, source: int, destination: int) -> float:
        """
        Gets the travel time of the best route between a source and destination
        :param source: A valid location identifier acting as a source
        :param destination: A valid location identifier acting as a destination
        :return: The travel time in seconds of the best path between the source and destination
        """
        if source < 0 or source >= len(self._location_coord):
            raise Exception("Invalid source location id {}".format(source))
        if destination < 0 or destination >= len(self._location_coord):
            raise Exception("Invalid destination location id {}".format(destination))
        if not (source, destination) in self._cached_distances:
            raise Exception("There is no cached travel time between {} and {}. You may need to call "
                            "compute_travel_times".format(source, destination))
        return self._cached_distances[source, destination]

    def to_json(self) -> dict:
        """
        Transforms the current object into a JSON object that can be serialized into a string
        :return: A JSON object representing this GISRoadNetwork
        """
        n_locations = len(self._location_coord)
        return {
            "number_locations": n_locations,
            "locations": [
                {"id": l_id, "lat": self._id_to_coord[l_id][0], "lng": self._id_to_coord[l_id][1]} for l_id in
                sorted(list(self._id_to_coord.keys()))
            ],
            "travel_times": [
                {"source": i, "destination": j, "time": self._cached_distances[i, j]}
                for i, j in product(range(n_locations), range(n_locations))]
        }

    @classmethod
    def from_json(cls, obj: dict) -> TypeVar('GISRoadNetwork'):
        """
        Class method that creates a GISRoadNetwork object from a valid JSON object
        :param obj: A valid JSON object.
        :return: An equivalent GISRoadNetwork object
        """
        gis_nw = GISRoadNetwork()
        for location_dict in obj["locations"]:
            gis_nw.add_location(location_dict["lat"], location_dict["lng"])
        for travel_dict in obj["travel_times"]:
            gis_nw.add_travel_time(travel_dict["source"], travel_dict["destination"], travel_dict["time"])
        return gis_nw

