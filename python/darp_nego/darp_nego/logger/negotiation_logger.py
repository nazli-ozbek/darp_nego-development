import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

class DARPNegotiationLogger:
    """
    Logger for DARP negotiation process that tracks state round by round
    and generates both JSON and human-readable TXT logs.
    Supports both full and partial acceptance depending on protocol behavior.
    """
    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None, log_subdir: str = ""):
        self.start_time = datetime.now()
        
        # Use the provided session_id or generate a new one
        self.session_id = session_id or self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Create session-specific directory path
        self.log_dir = os.path.join(log_dir, self.session_id, log_subdir)
        
        self.start_timestamp = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_data = {
            "session_id": self.session_id,
            "timestamp": self.start_timestamp,
            "prenegotiation": {},
            "rounds": [],
            "final_state": {}
        }
        self.execution_start_time = time.time()
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def log_prenegotiation(self, participants: Dict, domain: Any, distance_analysis: Dict = None, cost_analysis: Dict = None, client_revelations: List[tuple] = None):
        """Log the prenegotiation state"""
        revealed_clients = {}
        
        # We'll collect only the revealed clients, which we can get from the domain
        for agent_id in participants.keys():
            # Get revealed clients (those added to the domain with this agent as owner)
            revealed = [
                client_id for client_id in domain.client_owners 
                if domain.get_current_owner(client_id) == agent_id
            ]
            revealed_clients[agent_id] = revealed
        
        initial_utilities = {}
        for agent_id, agent in participants.items():
            initial_utilities[agent_id] = agent.current_utility
        
        self.log_data["prenegotiation"] = {
            "participants": list(participants.keys()),
            "revealed_clients": revealed_clients,
            "initial_utilities": initial_utilities,
            "distance_analysis": distance_analysis,
            "cost_analysis": cost_analysis,
            "client_revelations": client_revelations or []
        }
    
    def log_outcome(self, proposed_outcome: Dict, domain: Any):
        """Log the outcome of a negotiation round"""
        proposed_transfers = []
        for client_id, agent_id in proposed_outcome.items():
            current_owner = domain.get_current_owner(client_id)
            proposed_transfers.append({
                "client_id": client_id,
                "from": current_owner,
                "to": agent_id
            })
        self.current_proposed_transfers = proposed_transfers
    
    def log_responses(self, responses: set, participants: Dict):
        """Log agent responses to a proposed outcome"""
        agent_responses = {}        
        for agent_id in participants.keys():
            accepted = agent_id in responses
            agent_responses[agent_id] = accepted
        self.current_agent_responses = agent_responses
    
    def log_round(self, round_number: int, is_accepted: bool, 
                  participants: Dict, round_summary: Dict = None):
        """Log the state of a negotiation round"""
        # Use detailed round summary if provided, otherwise fall back to basic logging
        if round_summary and 'agent_responses' in round_summary:
            # Use the detailed agent responses from the round summary
            detailed_responses = round_summary['agent_responses']
            
            # Extract utility changes and responses from detailed responses
            utility_changes = {}
            agent_responses = {}
            proposed_utilities = {}
            current_utilities = {}
            for agent_id, response_data in detailed_responses.items():
                agent_responses[agent_id] = response_data['accepted']
                proposed_utilities[agent_id] = response_data['proposed_utility']
                current_utilities[agent_id] = response_data['current_utility']
                # Utility change is current - proposed (positive means improvement in cost)
                utility_changes[agent_id] = current_utilities[agent_id] - proposed_utilities[agent_id]
        else:
            # Fallback to basic logging (for backward compatibility)
            utility_changes = {}
            agent_responses = self.current_agent_responses
            proposed_utilities = {}
            current_utilities = {}
            for agent_id, agent in participants.items():
                current_utilities[agent_id] = agent.current_utility
                if is_accepted and agent_responses.get(agent_id, False):
                    proposed_utilities[agent_id] = agent.current_utility + agent.utility_change
                    utility_changes[agent_id] = agent.utility_change
                else:
                    proposed_utilities[agent_id] = agent.current_utility
                    utility_changes[agent_id] = 0
        
        # Normalize acceptance/swap info (backward compatible with old logs)
        full_acceptance = bool(
            (round_summary or {}).get("full_acceptance", is_accepted)
        )
        num_swaps = int((round_summary or {}).get("num_swaps", 0) or 0)
        partial_acceptance = bool(
            (round_summary or {}).get("partial_acceptance", (num_swaps > 0 and not full_acceptance))
        )
        num_accepts = int(
            (round_summary or {}).get("num_accepts", sum(1 for accepted in agent_responses.values() if accepted))
        )

        # Create round data
        round_data = {
            "round_number": round_number,
            "proposed_transfers": getattr(self, 'current_proposed_transfers', []),
            "is_accepted": is_accepted,
            "agent_responses": agent_responses,
            "utility_changes": utility_changes,
            "current_utilities": current_utilities,
            "proposed_utilities": proposed_utilities,
            "full_acceptance": full_acceptance,
            "partial_acceptance": partial_acceptance,
            "num_accepts": num_accepts,
            "num_swaps": num_swaps,
            "applied_swaps": (round_summary or {}).get('applied_swaps', [])
        }
        
        self.log_data["rounds"].append(round_data)
        
        # Clear temporary data
        if hasattr(self, 'current_proposed_transfers'):
            del self.current_proposed_transfers
        if hasattr(self, 'current_agent_responses'):
            del self.current_agent_responses
    
    def log_final_state(self, agreement_reached: bool, round_count: int, 
                      participants: Dict, domain: Any):
        """Log the final state after negotiation"""
        final_utilities = {}
        final_client_assignments = {}
        
        # Get the final utilities from the last accepted round's proposed utilities
        # or from current agent utilities if no rounds were accepted
        if self.log_data["rounds"] and any(round_data["is_accepted"] for round_data in self.log_data["rounds"]):
            # Find the last accepted round
            last_accepted_round = None
            for round_data in reversed(self.log_data["rounds"]):
                if round_data["is_accepted"]:
                    last_accepted_round = round_data
                    break
            
            if last_accepted_round:
                final_utilities = last_accepted_round["proposed_utilities"]
            else:
                # Fallback to current agent utilities
                for agent_id, agent in participants.items():
                    final_utilities[agent_id] = agent.current_utility
        else:
            # No accepted rounds, use current agent utilities
            for agent_id, agent in participants.items():
                final_utilities[agent_id] = agent.current_utility
        
        for client_id, owner in domain.client_owners.items():
            final_client_assignments[client_id] = owner
        
        # Calculate utility changes from initial to final state
        initial_utilities = self.log_data["prenegotiation"]["initial_utilities"]
        utility_changes = {agent_id: final_utilities[agent_id] - initial_utilities[agent_id] 
                          for agent_id in final_utilities}
        
        avg_utility_change = sum(utility_changes.values()) / len(utility_changes) if utility_changes else 0
        
        # Count full/partial/rejections
        full_acceptances = sum(1 for round_data in self.log_data["rounds"] if round_data.get("full_acceptance", False))
        partial_acceptances = sum(
            1 for round_data in self.log_data["rounds"]
            if round_data.get("partial_acceptance", False)
        )
        full_rejections = len(self.log_data["rounds"]) - full_acceptances - partial_acceptances
        total_swaps_applied = sum(int(round_data.get("num_swaps", 0) or 0) for round_data in self.log_data["rounds"])
        
        # Calculate total utility changes from all accepted rounds
        total_utility_changes = {}
        for agent_id in final_utilities:
            total_utility_changes[agent_id] = 0
            for round_data in self.log_data["rounds"]:
                if round_data["is_accepted"]:
                    total_utility_changes[agent_id] += round_data["utility_changes"].get(agent_id, 0)
        
        self.log_data["final_state"] = {
            "agreement_reached": agreement_reached,
            "total_rounds": round_count,
            "full_acceptances": full_acceptances,
            "partial_acceptances": partial_acceptances,
            "full_rejections": full_rejections,
            "total_swaps_applied": total_swaps_applied,
            "final_utilities": final_utilities,
            "initial_utilities": initial_utilities,
            "utility_changes": utility_changes,
            "total_utility_changes": total_utility_changes,
            "avg_utility_change": avg_utility_change,
            "final_client_assignments": final_client_assignments,
            "execution_time": time.time() - self.execution_start_time
        }
    
    def save_logs(self):
        """Save logs to files and return the file paths"""
        # Add execution time to the log data
        self.log_data["execution_time"] = time.time() - self.execution_start_time
        
        # Use session_id in filenames for consistent identification
        json_filename = os.path.join(self.log_dir, f"negotiation_{self.session_id}.json")
        txt_filename = os.path.join(self.log_dir, f"negotiation_{self.session_id}.txt")
        
        # Save JSON log
        with open(json_filename, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        
        # Save human-readable TXT log
        with open(txt_filename, 'w', encoding='utf-8') as f:
            self._write_txt_log(f, json_filename)
        
        # Print saved file paths to terminal
        print(f"\nNegotiation logs saved:")
        print(f"- JSON: {os.path.abspath(json_filename)}")
        print(f"- TXT: {os.path.abspath(txt_filename)}")
        
        return json_filename, txt_filename
        
    def _write_txt_log(self, file, json_filename: str = None):
        """Write human-readable text log"""
        file.write(f"=== DARP Negotiation Log ===\n")
        file.write(f"Session ID: {self.session_id}\n")
        file.write(f"Started: {self.start_timestamp}\n")
        
        # Write client distance analysis at the top
        distance_analysis = self.log_data["prenegotiation"].get("distance_analysis")
        cost_analysis = self.log_data["prenegotiation"].get("cost_analysis")
        
        if distance_analysis:
            file.write("\n=== CLIENT FEATURE VECTOR DISTANCE MATRIX ===\n")
            client_ids = distance_analysis.get('client_ids', [])
            distance_matrix = distance_analysis.get('distance_matrix', {})
            
            if client_ids:
                # Write header
                file.write(f"{'Client':>8}")
                for client_id in client_ids:
                    file.write(f"{client_id:>8}")
                file.write("\n")
                
                # Write distance matrix
                for client_a_id in client_ids:
                    file.write(f"{client_a_id:>8}")
                    for client_b_id in client_ids:
                        distance = distance_matrix.get(client_a_id, {}).get(client_b_id, 0.0)
                        file.write(f"{distance:>8.2f}")
                    file.write("\n")
                file.write("=== END FEATURE VECTOR DISTANCE MATRIX ===\n")
        
        if cost_analysis:
            file.write("\n=== CLIENT COST ANALYSIS ===\n")
            cost_ranking = cost_analysis.get('cost_ranking', [])
            for client_info in cost_ranking:
                emoji = client_info.get('emoji', '')
                client_id = client_info.get('client_id', '')
                avg_dist = client_info.get('avg_distance', 0.0)
                cost_level = client_info.get('cost_level', '')
                file.write(f"Client {client_id}: {avg_dist:.2f} ({cost_level})\n")
            file.write("=== END CLIENT COST ANALYSIS ===\n")
        
        # Write client revelation information
        client_revelations = self.log_data["prenegotiation"].get("client_revelations", [])
        if client_revelations:
            file.write("\n=== CLIENT REVELATION SEQUENCE ===\n")
            for agent_id, client_id in client_revelations:
                file.write(f"Agent {agent_id} revealed client id: {client_id}\n")
            file.write("=== END CLIENT REVELATION SEQUENCE ===\n")
        
        # Write prenegotiation info
        file.write("\n=== Pre-negotiation Phase ===\n")
        participants = self.log_data["prenegotiation"]["participants"]
        file.write(f"Participants: {', '.join(participants)}\n\n")
        
        for agent_id in participants:
            revealed = self.log_data["prenegotiation"]["revealed_clients"].get(agent_id, [])
            initial_utility = self.log_data["prenegotiation"]["initial_utilities"].get(agent_id, 0)
            
            file.write(f"Agent {agent_id} enters the negotiation:\n")
            file.write(f"  - Initial utility: {initial_utility}\n")
            file.write(f"  - Clients revealed for negotiation: {revealed}\n\n")
        
        # Negotiation Rounds
        for round_data in self.log_data["rounds"]:
            round_num = round_data["round_number"]
            file.write(f"ROUND {round_num}\n")
            file.write("-" * 40 + "\n")
            
            file.write("Mediator proposes:\n")
            for propose in round_data["proposed_transfers"]:
                file.write(f"  - Transfer client {propose['client_id']} from Agent {propose['from']} to Agent {propose['to']}\n")
            
            if not round_data["proposed_transfers"]:
                file.write("  - No transfers proposed.\n")
            
            file.write("\nResponses:\n")
            for agent_id, accepted in round_data["agent_responses"].items():
                utility_change = round_data["utility_changes"].get(agent_id, 0)
                current_utility = round_data["current_utilities"].get(agent_id, 0)
                proposed_utility = round_data["proposed_utilities"].get(agent_id, current_utility)
                response = "ACCEPTS" if accepted else "REJECTS"
                
                if response == "ACCEPTS":
                    file.write(f"  - Agent {agent_id}: {response} (cost: {current_utility} -> {proposed_utility}, change: {utility_change})\n")
                else:
                    file.write(f"  - Agent {agent_id}: {response} (cost: {current_utility})\n")
            
            if round_data["is_accepted"]:
                file.write("\n[ACCEPTED] FULL ACCEPTANCE - All agents accepted the proposal\n\n")
                file.write("Applied swaps:\n")
                for swap in round_data.get("applied_swaps", []):
                    file.write(f"  - Client {swap['client_id']}: {swap['from']} -> {swap['to']}\n")
                
                file.write("\nUpdated costs:\n")
                for agent_id, utility in round_data["current_utilities"].items():
                    file.write(f"  - Agent {agent_id}: {utility}\n")
            else:
                file.write("\n[REJECTED] FULL REJECTION - Continuing negotiation\n")
            
            file.write("\n")
        
        # Closing section
        file.write("CLOSING\n")
        file.write("-" * 40 + "\n")
        
        final = self.log_data["final_state"]
        if final["agreement_reached"]:
            file.write("[SUCCESS] Agreement reached!\n\n")
        else:
            file.write("[FAILED] No agreement reached.\n\n")
        
        file.write(f"Negotiation completed in {final['total_rounds']} rounds ({final['execution_time']:.2f} seconds)\n")
        file.write(f"Full acceptances: {final['full_acceptances']}\n")
        file.write(f"Full rejections: {final['full_rejections']}\n\n")
        
        # Final client allocation
        file.write("Final allocation of clients:\n")
        client_by_agent = {}
        for client_id, owner in final["final_client_assignments"].items():
            if owner not in client_by_agent:
                client_by_agent[owner] = []
            client_by_agent[owner].append(client_id)
            
        for agent_id, clients in client_by_agent.items():
            file.write(f"  - Agent {agent_id}: Clients {clients}\n")
        
        file.write("\nCost changes:\n")
        for agent_id, change in final["utility_changes"].items():
            initial = final["initial_utilities"][agent_id]
            final_utility = final["final_utilities"][agent_id]
            total_change = final["total_utility_changes"].get(agent_id, 0)
            file.write(f"  - Agent {agent_id}: {initial} -> {final_utility} (net change: {change}, total from accepted rounds: {total_change})\n")
        
        file.write(f"\nAverage utility change: {final['avg_utility_change']:.1f}\n\n")
        
        # Footer
        file.write("========================================\n")
        file.write(f"Negotiation concluded at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # If we have the JSON filename, include it at the end
        if json_filename:
            file.write(f"\n=== Log Files ===\n")
            file.write(f"- JSON: {os.path.abspath(json_filename)}\n") 
