from typing import Optional, Iterable

from .classic_single_mediated_text import ClassicSingleMediatedTextMechanism
from ..negotiator import BasicDARPNegotiator
from ..outcome.darp_outcome import DARPNegotiationDomain, BasicDARPOutcome
from networkx.algorithms import weakly_connected_components
from networkx import MultiDiGraph
from ..utils import RankingPreferences


class SingleMediatedTextMechanism(ClassicSingleMediatedTextMechanism):

    def __init__(self, agents: Optional[Iterable[BasicDARPNegotiator]] = None, max_rounds: int = 300, **kwargs):
        super().__init__(agents, max_rounds=max_rounds, **kwargs)
        self.improvements = None
        self.preferences = None

    def calculate_similarities_client_requests(self):
        self.similarities = {}
        for client_id in self.domain.known_clients:
            request = self.domain.get_client(client_id)


    def randomized_egalitarian_outcome(self, s=0.5):
        n = len(self.improvements)
        agent_list = [agent_id for agent_id in self.improvements]
        agent_list = sorted(agent_list, key=lambda o: self.improvements[o], reverse=False)
        agent_list = [(i, agent_list[0][1]) for i in range(len(agent_list))]
        prob_list = [((2-0.5)/n + (2*i*(s-1))/(n*(n-1)), agent_id) for (i, agent_id) in agent_list]


    def generate_outcome(self, basic=True):
        if basic:
            return self.domain.generate_random_outcome()
        else:
            pass

    def prenegotiation(self):
        super().prenegotiation()

        self.improvements = dict()
        for agent in self.participants.values():
            #For each agent, initialize improvements dict
            self.improvements[agent.agent_id] = 0

        self.preferences = dict()
        for agent in self.participants.values():
            self.preferences[agent.agent_id] = RankingPreferences(self.domain.known_clients.keys())

    def build_graph(self, outcome: BasicDARPOutcome, positive_responses: set):

        g = MultiDiGraph()

        for client_id in outcome:
            new_owner = outcome[client_id]
            old_owner = self.domain.get_current_owner(client_id)
            if new_owner != old_owner and (new_owner in positive_responses) and (old_owner in positive_responses):
                if not g.has_node(new_owner):
                    g.add_node(new_owner)
                if not g.has_node(old_owner):
                    g.add_node(old_owner)
                g.add_edge(new_owner, old_owner, client=client_id)

        return g

    def get_acceptable_part(self, outcome: BasicDARPOutcome, graph: MultiDiGraph) -> BasicDARPOutcome:

        components = weakly_connected_components(graph)

        changes = set()
        for component in components:
            clients_involved_graph = set()
            clients_involved_outcome = set()
            subgraph = graph.subgraph(component)
            for node in subgraph.nodes:
                gets = self.domain.gets_clients(outcome, node)
                gives = self.domain.gives_clients(outcome, node)
                clients_involved_outcome.update(gets + gives)
            for (u, v, d) in subgraph.edges(data="client"):
                clients_involved_graph.add(d)
            if clients_involved_outcome == clients_involved_graph:
                changes.update(clients_involved_graph)
        new_outcome = {client_id: outcome[client_id] for client_id in changes}
        return new_outcome

    def round(self):
        outcome = self.generate_outcome()
        involved_agents = self.domain.get_involved_agents(outcome)
        responses = set()
        for agent_id in involved_agents:
            new_clients = self.domain.gets_clients(outcome, agent_id)
            old_clients = self.domain.gives_clients(outcome, agent_id)
            if self.participants[agent_id].is_acceptable(outcome):
                responses.add(agent_id)
                if len(new_clients) == 1 and len(old_clients) == 1:
                    self.preferences[agent_id].update_prefers(old_clients[0], new_clients[0])
            else:
                if len(new_clients) == 1 and len(old_clients) == 1:
                    self.preferences[agent_id].update_prefers(new_clients[0], old_clients[0])
        g = self.build_graph(outcome, responses)
        acceptable_part = self.get_acceptable_part(outcome, g)

        for client_id in acceptable_part:
            new_owner = acceptable_part[client_id]
            self.improvements[new_owner] = self.improvements[new_owner] + 1
            self.domain.update_client(client_id, new_owner)

        if acceptable_part:
            for agent in self.participants:
                agent.update_agreement(acceptable_part)
