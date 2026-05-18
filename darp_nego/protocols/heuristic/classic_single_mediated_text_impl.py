from typing import Optional, Iterable, Dict, List

from ..common.basic_darp_protocol import BasicDARPMechanism
from ...core.negotiator import BasicDARPNegotiator
from ...core.outcome.darp_outcome import DARPNegotiationDomain
from ...logging.negotiation_logger import DARPNegotiationLogger
from ...logging.darp_routing_logger import DARPRoutingLogger
from ...core.darp.road_network import GISRoadNetwork

import numpy as np

class ClassicSingleMediatedTextMechanism(BasicDARPMechanism):

    def __init__(self, agents: Optional[Iterable[BasicDARPNegotiator]] = None, max_rounds: int = 300, 
                 log_dir: str = "logs", session_id: Optional[str] = None, road_network: Optional[GISRoadNetwork] = None, **kwargs):
        super().__init__(agents, **kwargs)
        self.max_rounds = max_rounds
        self.domain = None

        self.road_network = road_network
        self.clients_to_be_shared = []
        
        # Use provided session ID or generate a new one
        if session_id:
            self.session_id = session_id
        else:
            from datetime import datetime
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize loggers with the session ID and appropriate subdirectories
        self.logger = DARPNegotiationLogger(log_dir=log_dir, session_id=self.session_id, log_subdir="negotiation")
        self.routing_logger = DARPRoutingLogger(log_dir=log_dir, session_id=self.session_id, log_subdir="routing")


    def generate_outcome(self, method: str = "swap", round_history: List[dict] = []):
        """
        Generate an outcome using the specified method.
        
        Parameters:
            method: "swap", "farthest"
            round_history: For method
        """
        if method == "farthest":
            # Create revealed_clients mapping from self.clients_to_be_shared
            # Handle multiple revealed clients per agent
            revealed_clients = {}
            for agent_id, client in self.clients_to_be_shared:
                if agent_id not in revealed_clients:
                    revealed_clients[agent_id] = []
                revealed_clients[agent_id].append(client.client_id)
            return self.domain.generate_outcome_farthest_distance(revealed_clients=revealed_clients, round_history=round_history)
        else:  # method == "swap" or default
            return self.domain.generate_swaped_outcome()


    def prenegotiation(self):
        self.domain = DARPNegotiationDomain()
        clients_to_be_shared = []
        client_revelations = []  # Track the order of client revelations
        
        # Log initial routes for each agent
        for agent in self.participants.values():
            self.routing_logger.log_darp_solution(
                agent.agent_id, 
                "initial", 
                agent.darp_problem
            )        
        # For each agent, ask what clients they are willing to give in
        for agent in self.participants.values():
            # Get list of clients to reveal
            clients_to_reveal = agent.reveal_client()
            if clients_to_reveal:
                for client in clients_to_reveal:
                    clients_to_be_shared.append((agent.agent_id, client))
                    client_revelations.append((agent.agent_id, client.client_id))  # Track revelation order
                    self.domain.add_client(client, agent.agent_id)
                    print(f"Agent {agent.agent_id} revealed client id: {client.client_id}")
        # For each agent, inform about the clients to negotiate
        # they know both their initial clients and revealed clients now
        for (agent_id, client) in clients_to_be_shared:
            for agent in self.participants.values():
                agent.add_client_information(client, agent_id)    
        self.clients_to_be_shared = clients_to_be_shared   
        
        # Print distances between clients
        distance_analysis = self.print_client_distances()
        
        # Analyze client costs after all clients have been revealed
        cost_analysis = self.domain.analyze_client_costs()
        
        self.logger.log_prenegotiation(self.participants, self.domain, distance_analysis, cost_analysis, client_revelations)

    def print_client_distances(self):
        """
        Print the feature vector distances between all pairs of clients.
        Returns the analysis data for logging.
        """
        print(f"\n=== CLIENT FEATURE VECTOR DISTANCE MATRIX ===")
        
        client_vectors, distance_matrix, client_ids = self.domain._calculate_feature_vectors_and_distances()
        
        if client_ids is None:
            print("Not enough clients for distance analysis.")
            return None
        
        # Print header
        print(f"{'Client':>8}", end="")
        for client_id in client_ids:
            print(f"{client_id:>8}", end="")
        print()
        
        # Print feature vector distance matrix
        for client_a_id in client_ids:
            print(f"{client_a_id:>8}", end="")
            for client_b_id in client_ids:
                distance = distance_matrix[client_a_id][client_b_id]
                print(f"{distance:>8.2f}", end="")
            print()
        
        print("=== END FEATURE VECTOR DISTANCE MATRIX ===\n")
        
        # Convert NumPy arrays to lists for JSON serialization
        client_vectors_serializable = {}
        for client_id, vector in client_vectors.items():
            client_vectors_serializable[client_id] = vector.tolist()
        
        # Return the analysis data for logging
        return {
            'client_ids': client_ids,
            'distance_matrix': distance_matrix,
            'client_vectors': client_vectors_serializable
        }


    def round(self, outcome):

        self.logger.log_outcome(outcome, self.domain)

        agent_utilities = {}
        agent_responses = {}

        # Step 1: Collect responses and utilities
        for agent in self.participants.values():
            acceptable_condition, temp_utility_change = agent.is_acceptable(outcome)
            agent_utilities[agent.agent_id] = (acceptable_condition, temp_utility_change)
            agent_responses[agent.agent_id] = {
                'accepted': acceptable_condition,
                'utility_change': agent.current_utility - temp_utility_change,
                'current_utility': agent.current_utility,
                'proposed_utility': temp_utility_change
            }

        responses = {agent_id for agent_id, (accepted, _) in agent_utilities.items() if accepted}
        self.logger.log_responses(responses, self.participants)

        # Store detailed round history
        round_history = {
            'proposed_outcome': outcome,
            'agent_responses': agent_responses,
            'full_acceptance': len(responses) == len(self.participants),
            'applied_swaps': []
        }

        # Step 2: Full acceptance -> full swap
        if len(responses) == len(self.participants): 
            # Store the applied swaps BEFORE updating the domain
            round_history['applied_swaps'] = [
                {'client_id': client_id, 'from': self.domain.get_current_owner(client_id), 'to': new_owner}
                for client_id, new_owner in outcome.items()
            ]
            
            # Update agents and domain
            for agent in self.participants.values():
                agent.update_agreement(outcome)
            for client_id in outcome:
                new_owner = outcome[client_id]
                self.domain.update_client(client_id, new_owner)
            
            print("🎉 All agents accepted! Agreement reached.")
            return True, round_history

        # No partial swaps allowed - only full acceptance
        print(f"📊 Result: {len(responses)}/{len(self.participants)} agents accepted. Continuing negotiation.")
        return False, round_history



    def negotiation(self):
        """
        Conduct the negotiation process until full agreement or max rounds reached.
        No partial swaps allowed - only proceeds when ALL agents accept.
        Returns:
            tuple: (success, log_paths)
        """
        previous_outcomes = []
        seen_outcomes = set()
        round_history = []

        for agent in self.participants.values():
            agent.prepare_negotiation()


        round_idx = 0
        while round_idx < self.max_rounds:
            round_number = round_idx + 1

            print(f"\n🔄 ROUND {round_number}")

            # Update all agents with current client ownership state before each round
            for agent in self.participants.values():
                for client_id, current_owner in self.domain.client_owners.items():
                    if client_id in agent._known_clients:
                        agent._client_owners[client_id] = current_owner

            # Generate outcome using farthest distance method
            outcome = self.generate_outcome(method="farthest", round_history=round_history)
            
            # Check if all combinations have been exhausted
            if outcome is None:
                print(f"\n🚨 NEGOTIATION TERMINATED - All combinations explored without agreement")
                self.logger.log_final_state(False, round_number, self.participants, self.domain)
                
                for agent in self.participants.values():
                    self.routing_logger.log_darp_solution(
                        agent.agent_id,
                        "final",
                        agent.darp_problem
                    )
                
                routing_log_path = self.routing_logger.save_logs()
                log_paths = self.logger.save_logs()
                return log_paths
            
            print(f"💡 Proposed outcome: {outcome}")
            outcome_key = tuple(sorted(outcome.items()))


            # Mark this outcome as seen (always store)
            seen_outcomes.add(outcome_key)
            # Run the round
            is_accepted, round_summary = self.round(outcome)
            round_history.append(round_summary)
            self.logger.log_round(round_number, is_accepted, self.participants, round_summary)
            # Check termination conditions
            if is_accepted:
                print(f"\n🎉 Agreement reached after {round_number} rounds!")
                print("💰 Final utility changes:")
                for agent in self.participants.values():
                    print(f"   Agent {agent.agent_id}: {agent.current_utility - agent.utility_change} → {agent.current_utility}")

                self.logger.log_final_state(True, round_number, self.participants, self.domain)

                for agent in self.participants.values():
                    self.routing_logger.log_darp_solution(
                        agent.agent_id,
                        "final",
                        agent.darp_problem
                    )

                routing_log_path = self.routing_logger.save_logs()
                log_paths = self.logger.save_logs()
                return log_paths
            
            round_idx += 1

        print(f"\nMaximum rounds ({self.max_rounds}) reached without agreement")
        self.logger.log_final_state(False, self.max_rounds, self.participants, self.domain)

        for agent in self.participants.values():
            self.routing_logger.log_darp_solution(
                agent.agent_id,
                "final",
                agent.darp_problem
            )

        routing_log_path = self.routing_logger.save_logs()
        log_paths = self.logger.save_logs()
        return log_paths
