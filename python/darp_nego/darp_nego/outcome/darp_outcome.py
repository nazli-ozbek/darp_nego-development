from typing import Optional, Iterable, Dict, List
import random
import numpy as np
from darp_nego import BasicDARPClient

import itertools
import math
from typing import Tuple
from collections import deque

BasicDARPOutcome = dict


class DARPNegotiationDomain:
    """
    Container for all clients that will be part of the negotiation.
    It tracks which agent currently owns each client.
    It allows the mediator to track the current state and generate valid proposals.
    """
    def __init__(self, clients: Optional[Iterable[BasicDARPClient]] = None, owners: Optional[Iterable[str]] = None):
        if clients and owners:
            print(clients, owners)
            if len(clients) == len(owners):
                self.known_clients = {c.client_id: c for c in clients}
                self.client_owners = {cid: o for (o, cid) in zip(owners, [c.client_id for c in clients])}
            else:
                raise Exception(
                    "Both clients and owners list should have the same length: {} {}".format(len(clients), len(owners)))
        else:
            self.known_clients = dict()
            self.client_owners = dict()

    def generate_random_outcome(self):
        clients = [client_id for client_id in self.client_owners]
        agent_ids = [agent_id for agent_id in self.client_owners.values()]
        random.shuffle(clients)
        random.shuffle(agent_ids)
        return {client: owner for (client, owner) in zip(clients, agent_ids)}
    
    
    def get_involved_agents(self, outcome: BasicDARPOutcome) -> Iterable[str]:
        involved = set()
        for client_id in outcome:
            new_owner = outcome[client_id]
            old_owner = self.client_owners[client_id]
            if new_owner != old_owner:
                involved.add(new_owner)
                involved.add(old_owner)
        return involved

    def get_current_owner(self, client_id: int) -> str:
        return self.client_owners[client_id]

    def gives_clients(self, outcome: BasicDARPOutcome, agent: str):
        gives = []
        for client_id in outcome:
            new_owner = outcome[client_id]
            if self.client_owners[client_id] == agent and new_owner != agent:
                gives.append(client_id)
        return gives

    def gets_clients(self, outcome: BasicDARPOutcome, agent: str):
        gets = []
        for client_id in outcome:
            new_owner = outcome[client_id]
            if new_owner == agent and self.client_owners[client_id] != agent:
                gets.append(client_id)
        return gets

    def add_client(self, client: BasicDARPClient, owner: str):
        self.known_clients[client.client_id] = client
        self.client_owners[client.client_id] = owner

    def get_client(self, client_id: int) -> BasicDARPClient:
        return self.known_clients[client_id]

    def remove_client(self, client: BasicDARPClient):
        del self.known_clients[client.client_id]
        del self.client_owners[client.client_id]

    def update_client(self, client_id: int, owner: str):
        self.client_owners[client_id] = owner

    def _calculate_feature_vectors_and_distances(self):
        """
        Calculate normalized feature vectors and distance matrix for all clients.
        This is a shared utility method to avoid code duplication.
        
        Returns:
            tuple: (client_vectors, distance_matrix, client_ids)
                - client_vectors: dict mapping client_id to normalized feature vector
                - distance_matrix: dict mapping client_id to dict of distances to other clients
                - client_ids: list of client IDs in order
        """
        if len(self.known_clients) < 2:
            return None, None, None
        
        # Create cost-based feature vectors
        all_features = []
        client_ids = list(self.known_clients.keys())
        
        for client_id in client_ids:
            client = self.known_clients[client_id]
            sx, sy = client.start_coordinates
            ex, ey = client.end_coordinates
            
            # Calculate derived features that better represent routing cost
            pickup_time_window = client.late_pickup - client.early_pickup
            drop_time_window = client.late_drop - client.early_drop
            total_service_time = (client.late_drop - client.early_pickup)
            travel_distance = np.sqrt((ex - sx)**2 + (ey - sy)**2)
            
            # Calculate distance from origin (depot assumed at 0,0)
            pickup_distance_from_depot = np.sqrt(sx**2 + sy**2)
            drop_distance_from_depot = np.sqrt(ex**2 + ey**2)
            
            features = [
                pickup_time_window,      # Tighter windows = more constrained = higher cost
                drop_time_window,        # Tighter windows = more constrained = higher cost  
                total_service_time,      # Longer total time = higher cost
                travel_distance,         # Longer trips = higher cost
                client.volume,           # Higher volume = higher cost
                pickup_distance_from_depot,  # Farther from depot = higher cost
                drop_distance_from_depot,    # Farther from depot = higher cost
                client.early_pickup,     # Very early/late times can be costly
                client.late_drop,        # Very late drops can be costly
            ]
            all_features.append(features)
        
        # Convert to numpy array and normalize
        all_features = np.array(all_features)
        feature_means = np.mean(all_features, axis=0)
        feature_stds = np.std(all_features, axis=0)
        feature_stds[feature_stds == 0] = 1  # Avoid division by zero
        
        # Create normalized vectors
        client_vectors = {}
        for i, client_id in enumerate(client_ids):
            normalized_features = (all_features[i] - feature_means) / feature_stds
            client_vectors[client_id] = normalized_features
        
        # Calculate distance matrix
        distance_matrix = {}
        for client_a in client_ids:
            distance_matrix[client_a] = {}
            for client_b in client_ids:
                if client_a != client_b:
                    vector_a = client_vectors[client_a]
                    vector_b = client_vectors[client_b]
                    distance = np.linalg.norm(vector_a - vector_b)
                    distance_matrix[client_a][client_b] = distance
                else:
                    distance_matrix[client_a][client_b] = 0.0
        
        return client_vectors, distance_matrix, client_ids

    def analyze_client_costs(self):
        """
        Analyze and display the cost ranking of clients based on their feature distances.
        Should be called once during prenegotiation after all clients are revealed.
        Returns the analysis data for logging.
        """
        print(f"\n=== CLIENT COST ANALYSIS ===")
        
        client_vectors, distance_matrix, client_ids = self._calculate_feature_vectors_and_distances()
        
        if client_ids is None:
            print("Not enough clients for cost analysis.")
            return None
        
        # Calculate and display cost ranking
        avg_distances = {}
        for client_id in client_ids:
            distances_from_client = [distance_matrix[client_id][other] for other in client_ids if other != client_id]
            avg_distance = np.mean(distances_from_client)
            avg_distances[client_id] = avg_distance
        
        sorted_by_avg_distance = sorted(avg_distances.items(), key=lambda x: x[1], reverse=True)
        cost_ranking = []
        for i, (client_id, avg_dist) in enumerate(sorted_by_avg_distance):
            if i == 0:
                emoji, cost_level = "🔴", "most costly"
            elif i == len(sorted_by_avg_distance) - 1:
                emoji, cost_level = "🟢", "least costly"
            else:
                emoji, cost_level = "🟡", "moderate" # middle
            print(f"{emoji} Client {client_id}: {avg_dist:.2f} ({cost_level})")
            cost_ranking.append({
                'client_id': client_id,
                'avg_distance': avg_dist,
                'cost_level': cost_level,
                'emoji': emoji
            })
        
        print("=== END CLIENT ANALYSIS ===\n")
        
        # Return the analysis data for logging
        return {
            'avg_distances': avg_distances,
            'cost_ranking': cost_ranking,
            'sorted_client_ids': [item[0] for item in sorted_by_avg_distance]
        }

    def generate_swaped_outcome(self, max_swaps_per_pair=1):
        """
        Generates an outcome where clients are swapped between agents.
        Can swap multiple clients between each agent pair.
        
        Parameters:
            max_swaps_per_pair: Maximum number of clients to swap between each agent pair
        
        Returns:
            A dictionary mapping client_ids to new owner agent_ids
        """
        # Start with current client ownership
        outcome = {client_id: owner for client_id, owner in self.client_owners.items()}
        
        # Get unique agent IDs
        agent_ids = list(set(self.client_owners.values()))
        
        # If fewer than 2 agents, no swaps possible
        if len(agent_ids) < 2:
            return outcome
        
        # Get clients by owner
        clients_by_owner = {}
        for client_id, owner in self.client_owners.items():
            if owner not in clients_by_owner:
                clients_by_owner[owner] = []
            clients_by_owner[owner].append(client_id)
        
        # Create list of agents with clients to swap
        agents_with_clients = [agent for agent in agent_ids if agent in clients_by_owner]
        
        # If fewer than 2 agents have clients, no swaps possible
        if len(agents_with_clients) < 2:
            return outcome
        
        # Shuffle the agents to create random pairings
        random.shuffle(agents_with_clients)
        
        # Create pairs of agents (if odd number, last one stays unpaired)
        agent_pairs = []
        for i in range(0, len(agents_with_clients) - 1, 2):
            agent_pairs.append((agents_with_clients[i], agents_with_clients[i+1]))
        
        # For each pair, swap their clients
        for agent1, agent2 in agent_pairs:
            # Get clients from each agent
            agent1_clients = clients_by_owner[agent1]
            agent2_clients = clients_by_owner[agent2]
            
            # Determine how many clients to swap
            num_swaps = min(
                len(agent1_clients),  # Can't swap more than agent1 has
                len(agent2_clients),  # Can't swap more than agent2 has
                max_swaps_per_pair    # User-defined maximum
            )
            
            # Swap the determined number of clients
            for i in range(num_swaps):
                client1 = agent1_clients[i]
                client2 = agent2_clients[i]
                
                # Perform the swap in the outcome
                outcome[client1] = agent2
                outcome[client2] = agent1
        
        return outcome



    def generate_outcome_farthest_distance(
        self,
        revealed_clients: Dict[str, List[int]] | None = None,
        round_history: List[dict] | None = None,
    ) -> Dict[int, str] | None:

        # -------- defaults ----------
        if revealed_clients is None:
            revealed_clients = {}
            for cid, owner in self.client_owners.items():
                revealed_clients.setdefault(owner, []).append(cid)
        if round_history is None:
            round_history = []
        if len(revealed_clients) < 2:
            return None

        agents = list(revealed_clients)
        n_agents = len(agents)

        # -------- history ----------
        seen = {self._outcome_key(rh.get("proposed_outcome", {})) for rh in round_history}
        usage, reject = {}, {}
        for rh in round_history:
            out = rh.get("proposed_outcome", {})
            for cid in out:
                usage[cid] = usage.get(cid, 0) + 1
            if any(not r.get("accepted", False)
                for r in rh.get("agent_responses", {}).values()):
                for cid in out:
                    reject[cid] = reject.get(cid, 0) + 1

        # -------- distance matrix ----------
        _, dist, _ = self._calculate_feature_vectors_and_distances()
        if dist is None:
            return None

        def set_distance(cset: Tuple[int, ...]) -> float:
            """Total pair-wise distance inside a client set."""
            total = 0.0
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    total += dist[cset[i]][cset[j]]
            return total

        def assign_distance(order: Tuple[int, ...]) -> float:
            """Distance yielded by a receiver permutation."""
            return sum(dist[order[i]][order[p]]
                    for i, p in enumerate(order))

        # ------------------------------------------------------------------
        # 1) rank **client sets** by distance (desc), tie-break by history
        # ------------------------------------------------------------------
        pools = [revealed_clients[a] for a in agents]
        set_ranking = []
        for combo in itertools.product(*pools):
            d = set_distance(combo)
            hist_pen = sum(usage.get(c, 0) + reject.get(c, 0) for c in combo)
            set_ranking.append((d, -hist_pen, combo))  # −penalty → smaller is better
        set_ranking.sort(reverse=True)                 # biggest distance first

        # ------------------------------------------------------------------
        # 2) for every set, iterate permutations ordered by assignment dist
        # ---------------------------------------------------------------
        def assign_distance(cset: Tuple[int, ...], perm: Tuple[int, ...]) -> float:
            """Sum of distances after assigning each cset[i] to receiver perm[i]."""
            return sum(dist[cset[i]][cset[perm[i]]] for i in range(n_agents))

        for _, _, combo in set_ranking:
            # order permutations by assignment distance (desc)
            permuted = sorted(
                itertools.permutations(range(n_agents)),
                key=lambda perm: assign_distance(combo, perm),
                reverse=True,
            )
            for perm in permuted:
                proposal = {cid: owner for cid, owner in self.client_owners.items()}
                for giver_idx, recv_idx in enumerate(perm):
                    giver_ag = agents[giver_idx]
                    recv_ag  = agents[recv_idx]
                    cid      = combo[giver_idx]
                    proposal[cid] = recv_ag
                if self._outcome_key(proposal) not in seen:
                    return proposal

        # explored everything
        return None


    def _outcome_key(self, mapping: Dict[int, str]) -> Tuple[Tuple[int, str], ...]:
        """Stable key so we can store outcomes in a set."""
        return tuple(sorted(mapping.items()))
