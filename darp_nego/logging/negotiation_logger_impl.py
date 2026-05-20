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
            "schema_version": "2.0",
            "session_id": self.session_id,
            "timestamp": self.start_timestamp,
            "deprecated_fields": {
                "rounds[].num_swaps": "legacy alias of rounds[].applied_swap_count",
                "rounds[].utility_changes": "legacy alias of rounds[].cost_delta",
                "final_state.utility_changes": "legacy alias of final_state.final_cost_delta_by_agent",
                "final_state.total_utility_changes": "legacy alias of final_state.total_cost_delta_by_agent"
            },
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
    
    @staticmethod
    def _is_changed_transfer(transfer: Dict) -> bool:
        return transfer.get("from") != transfer.get("to")

    @classmethod
    def _filter_changed_transfers(cls, transfers: List[Dict]) -> List[Dict]:
        return [transfer for transfer in transfers if cls._is_changed_transfer(transfer)]

    def _transfers_from_outcome(self, outcome: Dict, domain: Any) -> List[Dict]:
        transfers = []
        for client_id, agent_id in outcome.items():
            current_owner = domain.get_current_owner(client_id)
            transfer = {
                "client_id": client_id,
                "from": current_owner,
                "to": agent_id
            }
            if self._is_changed_transfer(transfer):
                transfers.append(transfer)
        return transfers

    @staticmethod
    def _cost_delta(current_costs: Dict, new_costs: Dict) -> Dict:
        return {
            agent_id: new_costs.get(agent_id, current_cost) - current_cost
            for agent_id, current_cost in current_costs.items()
        }

    @staticmethod
    def _cost_saving(cost_delta: Dict) -> Dict:
        return {agent_id: -delta for agent_id, delta in cost_delta.items()}

    def _collect_strategy_metadata(self, round_summary: Dict) -> Dict:
        if not round_summary:
            return {}
        metadata = dict(round_summary.get("strategy_metadata", {}) or {})
        for key in (
            "replay_buffer_size",
            "proposal_source",
            "pair_attempts",
            "training_loss",
            "model_active",
        ):
            if key in round_summary:
                metadata[key] = round_summary[key]
        return metadata

    def _warn_round_invariants(self, round_data: Dict) -> List[str]:
        warnings = []
        if round_data["applied_swap_count"] != len(round_data.get("applied_swaps", [])):
            warnings.append("applied_swap_count does not match len(applied_swaps)")
        if any(not self._is_changed_transfer(swap) for swap in round_data.get("applied_swaps", [])):
            warnings.append("applied_swaps contains a no-op transfer")
        if any(not self._is_changed_transfer(swap) for swap in round_data.get("proposed_transfers", [])):
            warnings.append("proposed_transfers contains a no-op transfer")
        classes = [
            bool(round_data.get("full_acceptance", False)),
            bool(round_data.get("partial_acceptance", False)),
            bool(round_data.get("rejection", False)),
        ]
        if sum(1 for value in classes if value) != 1:
            warnings.append("round classification is not mutually exclusive/exhaustive")
        return warnings

    def log_outcome(self, proposed_outcome: Dict, domain: Any):
        """Log the outcome of a negotiation round"""
        self.current_proposed_transfers = self._transfers_from_outcome(proposed_outcome, domain)
    
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
            
            # Extract cost deltas and responses from detailed responses.
            agent_responses = {}
            proposed_utilities = {}
            current_utilities = {}
            for agent_id, response_data in detailed_responses.items():
                agent_responses[agent_id] = response_data['accepted']
                proposed_utilities[agent_id] = response_data['proposed_utility']
                current_utilities[agent_id] = response_data['current_utility']
            cost_delta = self._cost_delta(current_utilities, proposed_utilities)
            cost_saving = self._cost_saving(cost_delta)
        else:
            # Fallback to basic logging (for backward compatibility)
            agent_responses = self.current_agent_responses
            proposed_utilities = {}
            current_utilities = {}
            for agent_id, agent in participants.items():
                current_utilities[agent_id] = agent.current_utility
                if is_accepted and agent_responses.get(agent_id, False):
                    proposed_utilities[agent_id] = agent.current_utility + agent.utility_change
                else:
                    proposed_utilities[agent_id] = agent.current_utility
            cost_delta = self._cost_delta(current_utilities, proposed_utilities)
            cost_saving = self._cost_saving(cost_delta)
        
        # Normalize acceptance/swap info (backward compatible with old logs)
        proposed_transfers = self._filter_changed_transfers(
            getattr(self, 'current_proposed_transfers', [])
        )
        applied_swaps = self._filter_changed_transfers((round_summary or {}).get('applied_swaps', []))
        applied_swap_count = len(applied_swaps)
        involved_agents = sorted({
            str(agent_id)
            for transfer in proposed_transfers
            for agent_id in (transfer.get("from"), transfer.get("to"))
        })
        involved_agents = (round_summary or {}).get("involved_agents", involved_agents)
        all_involved_accepted = bool(
            (round_summary or {}).get(
                "all_involved_accepted",
                all(agent_responses.get(agent_id, False) for agent_id in involved_agents) if involved_agents else False
            )
        )
        all_participants_accepted = bool(
            (round_summary or {}).get(
                "all_participants_accepted",
                all(agent_responses.get(agent_id, False) for agent_id in participants.keys()) if participants else False
            )
        )
        full_acceptance_scope = (round_summary or {}).get("full_acceptance_scope", "all_participants")
        fallback_full_acceptance = all_participants_accepted if full_acceptance_scope == "all_participants" else all_involved_accepted
        full_acceptance = bool((round_summary or {}).get("full_acceptance", is_accepted or fallback_full_acceptance))
        partial_acceptance = bool(
            (round_summary or {}).get("partial_acceptance", (applied_swap_count > 0 and not full_acceptance))
        )
        rejection = bool(not full_acceptance and not partial_acceptance)
        applied_cost_delta = cost_delta if (full_acceptance or partial_acceptance) else {
            agent_id: 0 for agent_id in current_utilities
        }
        round_cost_after = proposed_utilities if (full_acceptance or partial_acceptance) else current_utilities
        num_accepts = int(
            (round_summary or {}).get("num_accepts", sum(1 for accepted in agent_responses.values() if accepted))
        )

        # Create round data
        round_data = {
            "round_number": round_number,
            "proposed_outcome": (round_summary or {}).get("proposed_outcome"),
            "proposed_transfers": proposed_transfers,
            "proposed_swap_count": len(proposed_transfers),
            "is_accepted": is_accepted,
            "agent_responses": agent_responses,
            "cost_delta": cost_delta,
            "cost_saving": cost_saving,
            "utility_changes": cost_delta,
            "current_utilities": current_utilities,
            "proposed_utilities": proposed_utilities,
            "round_cost_before_by_agent": current_utilities,
            "round_cost_after_by_agent": round_cost_after,
            "proposed_cost_delta_by_agent": cost_delta,
            "applied_cost_delta_by_agent": applied_cost_delta,
            "full_acceptance": full_acceptance,
            "full_acceptance_scope": full_acceptance_scope,
            "all_involved_accepted": all_involved_accepted,
            "all_participants_accepted": all_participants_accepted,
            "partial_acceptance": partial_acceptance,
            "rejection": rejection,
            "num_accepts": num_accepts,
            "applied_swap_count": applied_swap_count,
            "num_swaps": applied_swap_count,
            "applied_swaps": applied_swaps,
            "strategy_metadata": self._collect_strategy_metadata(round_summary or {})
        }
        invariant_warnings = list((round_summary or {}).get("validation_warnings", []) or [])
        invariant_warnings.extend(self._warn_round_invariants(round_data))
        if invariant_warnings:
            round_data["validation_warnings"] = invariant_warnings
        
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
        
        # Agent state is the source of truth after both full and partial swaps.
        for agent_id, agent in participants.items():
            final_utilities[agent_id] = agent.current_utility
        
        for client_id, owner in domain.client_owners.items():
            final_client_assignments[client_id] = owner
        
        # Calculate cost deltas from initial to final state.
        initial_utilities = self.log_data["prenegotiation"]["initial_utilities"]
        final_cost_delta = {
            agent_id: final_utilities[agent_id] - initial_utilities[agent_id]
            for agent_id in final_utilities
        }
        final_cost_saving = self._cost_saving(final_cost_delta)
        
        avg_cost_delta = sum(final_cost_delta.values()) / len(final_cost_delta) if final_cost_delta else 0
        avg_cost_saving = sum(final_cost_saving.values()) / len(final_cost_saving) if final_cost_saving else 0
        
        # Count full/partial/rejections
        full_acceptances = sum(1 for round_data in self.log_data["rounds"] if round_data.get("full_acceptance", False))
        partial_acceptances = sum(
            1 for round_data in self.log_data["rounds"]
            if round_data.get("partial_acceptance", False)
        )
        full_rejections = len(self.log_data["rounds"]) - full_acceptances - partial_acceptances
        total_swaps_applied = sum(len(round_data.get("applied_swaps", [])) for round_data in self.log_data["rounds"])
        
        # Calculate total applied cost deltas from rounds with applied swaps.
        total_cost_delta = {}
        for agent_id in final_utilities:
            total_cost_delta[agent_id] = 0
            for round_data in self.log_data["rounds"]:
                if round_data.get("full_acceptance", False) or round_data.get("partial_acceptance", False):
                    total_cost_delta[agent_id] += round_data.get("applied_cost_delta_by_agent", {}).get(agent_id, 0)
        total_cost_saving = self._cost_saving(total_cost_delta)
        
        self.log_data["final_state"] = {
            "agreement_reached": agreement_reached,
            "total_rounds": round_count,
            "full_acceptances": full_acceptances,
            "partial_acceptances": partial_acceptances,
            "full_rejections": full_rejections,
            "total_swaps_applied": total_swaps_applied,
            "final_utilities": final_utilities,
            "initial_utilities": initial_utilities,
            "final_cost_delta_by_agent": final_cost_delta,
            "final_cost_saving_by_agent": final_cost_saving,
            "total_cost_delta_by_agent": total_cost_delta,
            "total_cost_saving_by_agent": total_cost_saving,
            "utility_changes": final_cost_delta,
            "total_utility_changes": total_cost_delta,
            "avg_cost_delta": avg_cost_delta,
            "avg_cost_saving": avg_cost_saving,
            "avg_utility_change": avg_cost_delta,
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
                cost_delta = round_data.get("cost_delta", round_data.get("utility_changes", {})).get(agent_id, 0)
                cost_saving = round_data.get("cost_saving", {}).get(agent_id, -cost_delta)
                current_utility = round_data["current_utilities"].get(agent_id, 0)
                proposed_utility = round_data["proposed_utilities"].get(agent_id, current_utility)
                response = "ACCEPTS" if accepted else "REJECTS"
                
                if response == "ACCEPTS":
                    file.write(f"  - Agent {agent_id}: {response} (cost: {current_utility} -> {proposed_utility}, cost_delta: {cost_delta}, cost_saving: {cost_saving})\n")
                else:
                    file.write(f"  - Agent {agent_id}: {response} (cost: {current_utility})\n")
            
            if round_data.get("full_acceptance", round_data["is_accepted"]):
                file.write("\n[ACCEPTED] FULL ACCEPTANCE - All agents accepted the proposal\n\n")
                file.write("Applied swaps:\n")
                for swap in round_data.get("applied_swaps", []):
                    file.write(f"  - Client {swap['client_id']}: {swap['from']} -> {swap['to']}\n")
            elif round_data.get("partial_acceptance", False):
                file.write("\n[PARTIAL] PARTIAL ACCEPTANCE - Accepted sub-swaps were applied\n\n")
                file.write("Applied swaps:\n")
                for swap in round_data.get("applied_swaps", []):
                    file.write(f"  - Client {swap['client_id']}: {swap['from']} -> {swap['to']}\n")
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
        file.write(f"Partial acceptances: {final['partial_acceptances']}\n")
        file.write(f"Total swaps applied: {final['total_swaps_applied']}\n")
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
        for agent_id, change in final.get("final_cost_delta_by_agent", final["utility_changes"]).items():
            initial = final["initial_utilities"][agent_id]
            final_utility = final["final_utilities"][agent_id]
            saving = final.get("final_cost_saving_by_agent", {}).get(agent_id, -change)
            total_change = final.get("total_cost_delta_by_agent", final["total_utility_changes"]).get(agent_id, 0)
            file.write(f"  - Agent {agent_id}: {initial} -> {final_utility} (cost_delta: {change}, cost_saving: {saving}, total applied cost_delta: {total_change})\n")
        
        file.write(f"\nAverage cost delta: {final.get('avg_cost_delta', final['avg_utility_change']):.1f}\n")
        file.write(f"Average cost saving: {final.get('avg_cost_saving', 0):.1f}\n\n")
        
        # Footer
        file.write("========================================\n")
        file.write(f"Negotiation concluded at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # If we have the JSON filename, include it at the end
        if json_filename:
            file.write(f"\n=== Log Files ===\n")
            file.write(f"- JSON: {os.path.abspath(json_filename)}\n") 
