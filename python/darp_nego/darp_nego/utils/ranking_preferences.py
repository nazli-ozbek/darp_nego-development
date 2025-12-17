from typing import Optional, Iterable

from networkx.algorithms import ancestors, descendants, dag_longest_path_length, dag_longest_path
from networkx import MultiDiGraph, DiGraph


class RankingPreferences:

    def __init__(self, list_clients: Optional[Iterable[int]] = None):
        self.preference_graph = DiGraph()
        if list_clients:
            for client_id in list_clients:
                self.preference_graph.add_node(client_id)

    def add_client(self, client_id: int):
        self.preference_graph.add_node(client_id)

    def update_prefers(self, a, b):
        if not self.preference_graph.has_node(a):
            self.preference_graph.add_node(a)
        if not self.preference_graph.has_node(b):
            self.preference_graph.add_node(b)
        self.preference_graph.add_edge(a, b)

    def get_better_assignment(self, from_node: int) -> set:
        return descendants(self.preference_graph, from_node)


    def get_worse_assignments(self, from_node: int) -> set:
        return ancestors(self.preference_graph, from_node)

    def _get_may_improve(self, from_node: int) -> DiGraph:
        worse_agents = self.get_worse_assignments(from_node)
        may_improve = set(self.preference_graph.nodes).difference(worse_agents)
        return self.preference_graph.subgraph(may_improve)

    def _get_similarity_full_ranking(self, graph):
        current_length = dag_longest_path_length(graph)
        return current_length/(len(graph)-1)

    def similarity_full_ranking(self, from_node: int) -> float:
        subgraph = self._get_may_improve(from_node)
        return self._get_similarity_full_ranking(subgraph)

    def simulate_similarity_full_ranking(self, from_node: int, new_node: int, weights: Iterable[float] = [0.5, 0.5]) -> float:
        subgraph = self._get_may_improve(from_node)

        similarity1 = similarity2 = self._get_similarity_full_ranking(subgraph)

        if new_node in subgraph.nodes and (not new_node in self.get_better_assignment(from_node)):
            if not subgraph.has_edge(new_node, from_node):
                subgraph2 = subgraph.copy()
                subgraph2.add_edge(from_node, new_node)
                similarity2 = self._get_similarity_full_ranking(subgraph2)
            if not subgraph.has_edge(from_node, new_node):
                subgraph1 = subgraph.copy()
                subgraph1.add_edge(new_node, from_node)
                similarity1 = self._get_similarity_full_ranking(subgraph1)

        return weights[0]*similarity1 + weights[1]*similarity2





