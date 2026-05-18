from typing import Optional, Iterable
from collections import defaultdict

from networkx import MultiDiGraph
from networkx.algorithms import weakly_connected_components

from .classic_single_mediated_text_impl import ClassicSingleMediatedTextMechanism
from ...core.negotiator import BasicDARPNegotiator
from ...core.outcome.darp_outcome import BasicDARPOutcome


class ClassicPartialSingleMediatedTextMechanism(ClassicSingleMediatedTextMechanism):
    """
    Heuristic outcome generation + partial acceptance execution.
    Keeps heuristic proposal logic, but applies acceptable sub-swaps even when
    not all participants accept (same acceptance style as learning mechanism).
    """

    def __init__(self, agents: Optional[Iterable[BasicDARPNegotiator]] = None,
                 max_rounds: int = 300, **kwargs):
        super().__init__(agents=agents, max_rounds=max_rounds, **kwargs)
        self.improvements = {}

    def prenegotiation(self):
        super().prenegotiation()
        self.improvements = {agent.agent_id: 0 for agent in self.participants.values()}

    def build_graph(self, outcome: BasicDARPOutcome, positive_responses: set):
        g = MultiDiGraph()
        for client_id in outcome:
            new_owner = outcome[client_id]
            old_owner = self.domain.get_current_owner(client_id)
            if new_owner != old_owner and new_owner in positive_responses and old_owner in positive_responses:
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
            for (_, _, client_id) in subgraph.edges(data="client"):
                clients_involved_graph.add(client_id)
            if clients_involved_outcome == clients_involved_graph:
                changes.update(clients_involved_graph)
        return {client_id: outcome[client_id] for client_id in changes}

    def _apply_partial_swaps_to_agents(self, swap_decisions):
        include_map = defaultdict(list)
        exclude_map = defaultdict(list)
        for client_id, old_owner, new_owner in swap_decisions:
            include_map[new_owner].append(client_id)
            exclude_map[old_owner].append(client_id)

        affected_agents = set(include_map.keys()) | set(exclude_map.keys())
        for agent_id in affected_agents:
            agent = self.participants[agent_id]
            including = include_map.get(agent_id, [])
            excluding = exclude_map.get(agent_id, [])
            if not including and not excluding:
                continue
            previous_cost = agent.current_utility
            new_cost = agent.get_utility(including=including, excluding=excluding)
            agent.utility_change = previous_cost - new_cost
            agent.current_utility = new_cost
            if agent._last_utility_problem is not None:
                agent.darp_problem = agent._last_utility_problem
                agent._last_utility_problem = None

    def round(self, outcome):
        self.logger.log_outcome(outcome, self.domain)

        involved_agents = self.domain.get_involved_agents(outcome)
        responses = set()
        agent_responses = {}

        for agent_id in self.participants.keys():
            if agent_id in involved_agents:
                accepted, temp_utility = self.participants[agent_id].is_acceptable(outcome)
                current_utility = self.participants[agent_id].current_utility
                agent_responses[agent_id] = {
                    "accepted": accepted,
                    "utility_change": current_utility - temp_utility,
                    "current_utility": current_utility,
                    "proposed_utility": temp_utility,
                }
                if accepted:
                    responses.add(agent_id)
            else:
                current_utility = self.participants[agent_id].current_utility
                agent_responses[agent_id] = {
                    "accepted": False,
                    "utility_change": 0.0,
                    "current_utility": current_utility,
                    "proposed_utility": current_utility,
                }

        self.logger.log_responses(responses, self.participants)

        graph = self.build_graph(outcome, responses)
        acceptable_part = self.get_acceptable_part(outcome, graph)

        pending_updates = []
        for client_id, new_owner in acceptable_part.items():
            old_owner = self.domain.get_current_owner(client_id)
            pending_updates.append((client_id, old_owner, new_owner))
            self.improvements[new_owner] += 1

        num_accepts = len(responses)
        num_swaps = len(acceptable_part)
        is_full_acceptance = (num_accepts == len(self.participants))

        if pending_updates:
            print(f"Applying {len(pending_updates)} partial/full swaps.")
            self._apply_partial_swaps_to_agents(pending_updates)
            for client_id, _, new_owner in pending_updates:
                self.domain.update_client(client_id, new_owner)

        round_summary = {
            "agent_responses": agent_responses,
            "num_accepts": num_accepts,
            "num_swaps": num_swaps,
            "full_acceptance": is_full_acceptance,
            "partial_acceptance": (num_swaps > 0 and not is_full_acceptance),
            "applied_swaps": [
                {"client_id": client_id, "from": old_owner, "to": new_owner}
                for client_id, old_owner, new_owner in pending_updates
            ],
        }

        if is_full_acceptance:
            print("🎉 All agents accepted! Full agreement reached.")
            return True, round_summary

        print(f"📊 Result: {num_accepts}/{len(self.participants)} accepts, {num_swaps} swaps applied.")
        return False, round_summary
