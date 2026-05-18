from openrouteservice import Client, directions, isochrones, distance_matrix
import pyproj
import shapely as shp
import numpy as np

class GeoTools:

    def __init__(self, base_url="http://localhost:8080/ors", ors_to_gmaps_factor=1.75):
        self._client = Client(key="5b3ce3597851110001cf62483628fb83525f4c22bf3386c79cab5f2a", base_url="https://api.openrouteservice.org")
        self._ors_to_gmaps_factor = ors_to_gmaps_factor
        self._transformer_4326_3857 = pyproj.Transformer.from_proj(pyproj.Proj('epsg:4326'), pyproj.Proj('epsg:3857'), always_xy=True)
        self._transformer_3857_4326 = pyproj.Transformer.from_proj(pyproj.Proj('epsg:3857'), pyproj.Proj('epsg:4326'), always_xy=True)

    def get_distance_matrix(self, locations):
        response = distance_matrix.distance_matrix(self._client, locations,
                                   profile="driving-car",
                                   metrics=["duration"])
        for i in range(len(response["durations"])):
            for j in range(len(response["durations"][i])):
                response["durations"][i][j] = response["durations"][i][j]*self._ors_to_gmaps_factor
        return response["durations"]

    def get_route(self, origin_lat, origin_lon, destination_lat, destination_lon):
        dir_response = directions(self._client, [(origin_lat, origin_lon), (destination_lat, destination_lon)], profile="driving-car", format="geojson", geometry_simplify=True, instructions=False)
        return dir_response["features"][0]["summary"]["duration"]*self._ors_to_gmaps_factor, dir_response["features"][0]["geometry"]["coordinates"], dir_response["features"][0]["geometry"]

    def get_isochrone_from_line(self, geom_line, minutes_isochrone=180, distance_sample=200):
        line_string = shp.geometry.shape(geom_line)
        converted_ls = shp.ops.transform(self._transformer_4326_3857.transform, line_string)

        distances = np.arange(0, converted_ls.length, distance_sample)
        projected_points = [shp.ops.transform(self._transformer_3857_4326.transform, converted_ls.interpolate(distance)) for distance in distances] + [converted_ls.boundary.geoms[1]]
        points = [ (p.x, p.y) for p in projected_points]

        iso_response = isochrones.isochrones(self._client, points, range=[minutes_isochrone/self._ors_to_gmaps_factor])

        convert_isochrones = [ shp.transform(self._transformer_4326_3857.transform,shp.geometry.shape(f["geometry"])) for f in iso_response["features"] ]

        return shp.ops.unary_union(convert_isochrones)


