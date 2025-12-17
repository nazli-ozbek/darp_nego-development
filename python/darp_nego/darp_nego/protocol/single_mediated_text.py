from typing import Optional, Iterable
import random
import math
from collections import defaultdict, deque
import torch
from ..learning.logistic_swap_model import LogisticSwapModel
from .classic_single_mediated_text import ClassicSingleMediatedTextMechanism
from ..negotiator import BasicDARPNegotiator
from ..outcome.darp_outcome import DARPNegotiationDomain, BasicDARPOutcome
from networkx.algorithms import weakly_connected_components
from networkx import MultiDiGraph
from ..utils import RankingPreferences


class SingleMediatedTextMechanism(ClassicSingleMediatedTextMechanism):

    def __init__(self, agents: Optional[Iterable[BasicDARPNegotiator]] = None,
                 max_rounds: int = 300,
                 **kwargs):
        super().__init__(agents, max_rounds=max_rounds, **kwargs)

        # === Core State ===
        self.improvements = {}
        self.preferences = {}

        # === Learning State ===
        self.client_feature_dim = 5
        self.agent_behavior_feature_dim = 4
        self.agent_behavior_window = kwargs.get("agent_behavior_window", 20)
        segment_bucket_param = kwargs.get("segment_bucket_size")
        self._segment_bucket_dynamic = segment_bucket_param is None
        self.segment_bucket_min = kwargs.get("segment_bucket_min", 1.0)
        self.segment_bucket_max = kwargs.get("segment_bucket_max", 10000.0)
        self.segment_bucket_target_divisions = kwargs.get("segment_bucket_target_divisions", 4)
        self.segment_bucket_size = segment_bucket_param if segment_bucket_param is not None else 0.0
        self.swap_feature_dim = self.client_feature_dim * 2 + self.agent_behavior_feature_dim
        self.training_buffer_maxlen = kwargs.get("swap_buffer_maxlen", 500)
        self.agent_swap_models = {}
        self.swap_training_buffers = defaultdict(lambda: deque(maxlen=self.training_buffer_maxlen))
        default_min_samples = max(5, int(self.max_rounds * 0.2))
        self.min_samples_for_model = kwargs.get("min_samples_for_model", default_min_samples)
        self.max_model_swaps = kwargs.get("max_model_swaps", 3)
        self.swap_probability_floor = kwargs.get("swap_probability_floor", 0.5)
        self.agent_deterministic_accepts = set()
        self.frozen_pair_keys = set()
        self.current_pair_attempts = []
        self.agent_recent_outcomes = defaultdict(lambda: deque(maxlen=self.agent_behavior_window))
        self.agent_segment_stats = defaultdict(lambda: defaultdict(lambda: {"attempts": 0, "success": 0}))
        self.agent_total_stats = defaultdict(lambda: {"attempts": 0, "success": 0})
        self._clear_case_statistics()

        print("[INIT] Mediator initialized with logistic swap model (case-specific)")
        print(f"[INIT] LogisticSwapModel input_dim = {self.swap_feature_dim}")

    # --------------------------------------------------
    # Learning Helpers
    # --------------------------------------------------
    def _clear_case_statistics(self):
        """Reset per-case statistics for swap generation."""
        self.frozen_pair_keys.clear()

    def _reset_learning_state(self):
        """Reset the logistic regression model and all accumulated data."""
        self.agent_swap_models.clear()
        self.swap_training_buffers.clear()
        self._clear_case_statistics()
        self.agent_deterministic_accepts.clear()
        self.current_pair_attempts = []
        self.agent_recent_outcomes.clear()
        self.agent_segment_stats.clear()
        self.agent_total_stats.clear()
        if self._segment_bucket_dynamic:
            self.segment_bucket_size = 0.0
        print("[RESET] Logistic swap model and buffers have been reset.")

    @staticmethod
    def _agent_sort_key(agent_id: str):
        return int(agent_id) if str(agent_id).isdigit() else agent_id
    def _pair_key(self, agent_a: str, client_a: int, agent_b: str, client_b: int):
        agents = sorted([agent_a, agent_b], key=self._agent_sort_key)
        clients = tuple(sorted((client_a, client_b)))
        return (agents[0], agents[1], clients[0], clients[1])

    def _configure_segment_bucket_size(self):
        """Derive a reasonable grid size from the revealed client coordinates."""
        if not self._segment_bucket_dynamic:
            return
        if not self.domain or not getattr(self.domain, "known_clients", None):
            return
        coords = []
        for client in self.domain.known_clients.values():
            if getattr(client, "start_coordinates", None):
                coords.append(client.start_coordinates)
        if not coords:
            self.segment_bucket_size = max(self.segment_bucket_min, 1.0)
            print(f"[SEGMENT] No coordinates found; fallback bucket={self.segment_bucket_size:.2f}")
            return
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        dominant_span = max(span_x, span_y)
        if dominant_span <= 0:
            bucket = self.segment_bucket_min
        else:
            divisions = max(1.0, float(self.segment_bucket_target_divisions))
            bucket = dominant_span / divisions
        bucket = max(self.segment_bucket_min, min(bucket, self.segment_bucket_max))
        self.segment_bucket_size = bucket
        print(f"[SEGMENT] Auto bucket size={bucket:.2f} (span_x={span_x:.2f}, span_y={span_y:.2f})")

    def _get_client_feature_vector(self, client) -> list:
        pickup_window = (client.late_pickup - client.early_pickup) / 3600.0
        drop_window = (client.late_drop - client.early_drop) / 3600.0
        service_span = (client.late_drop - client.early_pickup) / 3600.0
        travel_distance = math.hypot(
            client.end_coordinates[0] - client.start_coordinates[0],
            client.end_coordinates[1] - client.start_coordinates[1],
        ) / 100.0
        client_volume = (client.volume - 1.0) / 99.0 if client.volume is not None else 0.0
        return [
            float(pickup_window),
            float(drop_window),
            float(service_span),
            float(travel_distance),
            float(client_volume),
        ]

    def _get_agent_model(self, agent_id: str) -> LogisticSwapModel:
        if agent_id not in self.agent_swap_models:
            self.agent_swap_models[agent_id] = LogisticSwapModel(input_dim=self.swap_feature_dim)
        return self.agent_swap_models[agent_id]

    def _segment_key(self, client) -> Optional[tuple]:
        if client is None or self.segment_bucket_size <= 0:
            return None
        bucket = self.segment_bucket_size
        sx, sy = client.start_coordinates
        return (int(sx // bucket), int(sy // bucket))

    def _get_agent_behavior_features(self, agent_id: str, receive_client_id: int) -> list:
        recent_history = self.agent_recent_outcomes.get(agent_id, [])
        if recent_history:
            recent_rate = sum(recent_history) / len(recent_history)
        else:
            recent_rate = 0.5

        totals = self.agent_total_stats.get(agent_id, {"attempts": 0, "success": 0})
        if totals["attempts"] > 0:
            overall_rate = totals["success"] / totals["attempts"]
        else:
            overall_rate = 0.5

        receive_client = self.domain.get_client(receive_client_id)
        segment_key = self._segment_key(receive_client)
        if segment_key is not None:
            segment_stats = self.agent_segment_stats[agent_id].get(segment_key, {"attempts": 0, "success": 0})
            attempts = segment_stats["attempts"]
            segment_rate = (segment_stats["success"] / attempts) if attempts > 0 else overall_rate
        else:
            attempts = 0
            segment_rate = overall_rate
        segment_confidence = min(attempts / max(1.0, float(self.agent_behavior_window)), 1.0)
        return [
            float(recent_rate),
            float(overall_rate),
            float(segment_rate),
            float(segment_confidence),
        ]

    def _update_agent_behavior_stats(self, agent_id: str, receive_client_id: int, accepted: bool):
        outcome_flag = 1.0 if accepted else 0.0
        self.agent_recent_outcomes[agent_id].append(outcome_flag)
        totals = self.agent_total_stats[agent_id]
        totals["attempts"] += 1
        totals["success"] += 1 if accepted else 0

        receive_client = self.domain.get_client(receive_client_id)
        segment_key = self._segment_key(receive_client)
        if segment_key is not None:
            segment_stats = self.agent_segment_stats[agent_id][segment_key]
            segment_stats["attempts"] += 1
            segment_stats["success"] += 1 if accepted else 0

    def _build_agent_swap_features(self, agent_id: str, give_client_id: int, receive_client_id: int) -> torch.Tensor:
        """Feature vector for an agent giving give_client and receiving receive_client."""
        give_client = self.domain.get_client(give_client_id)
        receive_client = self.domain.get_client(receive_client_id)
        behavior_features = self._get_agent_behavior_features(agent_id, receive_client_id)
        features = [
            *self._get_client_feature_vector(give_client),
            *self._get_client_feature_vector(receive_client),
            *behavior_features,
        ]
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _total_buffer_size(self) -> int:
        return sum(len(buffer) for buffer in self.swap_training_buffers.values())

    def _is_deterministic_swap(self, agent_id: str, give_client_id: int, receive_client_id: int) -> bool:
        return (agent_id, give_client_id, receive_client_id) in self.agent_deterministic_accepts

    def _mark_deterministic_swap(self, agent_id: str, give_client_id: int, receive_client_id: int):
        self.agent_deterministic_accepts.add((agent_id, give_client_id, receive_client_id))

    def _agent_swap_probability(self, agent_id: str, give_client_id: int, receive_client_id: int) -> float:
        if self._is_deterministic_swap(agent_id, give_client_id, receive_client_id):
            return 1.0
        features = self._build_agent_swap_features(agent_id, give_client_id, receive_client_id)
        model = self._get_agent_model(agent_id)
        return float(model.predict_proba(features))

    # --------------------------------------------------
    # Outcome Generation (random or model-guided)
    # --------------------------------------------------

    def generate_outcome(self, method: str = "random", round_history=None, basic=True):
        if round_history is None:
            round_history = []
        self.current_pair_attempts = []

        dataset_size = self._total_buffer_size()
        if dataset_size < self.min_samples_for_model:
            if method == "farthest":
                print(f"[GEN] Cold start (dataset={dataset_size}, min={self.min_samples_for_model}) -> farthest-distance proposal")
                base_outcome = self.domain.generate_outcome_farthest_distance(
                    revealed_clients=None, round_history=round_history
                )
            else:
                print(f"[GEN] Cold start (dataset={dataset_size}, min={self.min_samples_for_model}) -> random swap proposal")
                base_outcome = self.domain.generate_random_outcome()
            if base_outcome:
                self._capture_pairs_from_outcome(base_outcome)
            return base_outcome

        outcome = self._generate_model_guided_outcome()
        if outcome is not None:
            return outcome

        print("[GEN] Model proposal exhausted, reverting to farthest-distance heuristic.")
        if method == "farthest":
            return self.domain.generate_outcome_farthest_distance(
                revealed_clients=None, round_history=round_history
            )
        return self.domain.generate_random_outcome()

    def _generate_model_guided_outcome(self) -> Optional[BasicDARPOutcome]:
        if not self.domain or not self.domain.client_owners:
            return None

        round_num = getattr(self, "current_round_number", 0)
        base_outcome = {cid: owner for cid, owner in self.domain.client_owners.items()}

        candidates = []
        participant_ids = sorted(self.participants.keys(), key=self._agent_sort_key)
        clients_by_agent = defaultdict(list)
        for client_id, owner in self.domain.client_owners.items():
            clients_by_agent[owner].append(client_id)

        for i in range(len(participant_ids)):
            agent_a = participant_ids[i]
            for j in range(i + 1, len(participant_ids)):
                agent_b = participant_ids[j]
                for client_a in clients_by_agent.get(agent_a, []):
                    for client_b in clients_by_agent.get(agent_b, []):
                        if self._is_pair_frozen(agent_a, client_a, agent_b, client_b):
                            continue
                        prob_a = self._agent_swap_probability(agent_a, client_a, client_b)
                        prob_b = self._agent_swap_probability(agent_b, client_b, client_a)
                        prob = prob_a * prob_b
                        candidates.append((prob, agent_a, client_a, agent_b, client_b))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen_pairs = []
        used_clients = set()

        for prob, agent_a, client_a, agent_b, client_b in candidates:
            if client_a in used_clients or client_b in used_clients:
                continue
            if prob < self.swap_probability_floor and chosen_pairs:
                break
            base_outcome[client_a] = agent_b
            base_outcome[client_b] = agent_a
            chosen_pairs.append({
                "client_a": client_a,
                "agent_a": agent_a,
                "client_b": client_b,
                "agent_b": agent_b,
                "prob": prob
            })
            used_clients.update({client_a, client_b})
            if self.max_model_swaps and len(chosen_pairs) >= self.max_model_swaps:
                break

        if not chosen_pairs:
            prob, agent_a, client_a, agent_b, client_b = candidates[0]
            base_outcome[client_a] = agent_b
            base_outcome[client_b] = agent_a
            chosen_pairs = [{
                "client_a": client_a,
                "agent_a": agent_a,
                "client_b": client_b,
                "agent_b": agent_b,
                "prob": prob
            }]
            used_clients.update({client_a, client_b})

        self.current_pair_attempts = chosen_pairs

        top_logs = ", ".join(
            f"c{pair['client_a']}@{pair['agent_a']}<->c{pair['client_b']}@{pair['agent_b']} (p={pair['prob']:.2f})"
            for pair in chosen_pairs
        )
        print(f"[GEN] Logistic model selected swaps: {top_logs}")
        return base_outcome

    def _record_pair_training_examples(
        self,
        pair_attempts: Iterable[dict],
        acceptable_part: BasicDARPOutcome,
        round_num: int,
        agent_responses: dict,
    ):
        """Persist features and labels for each attempted client pair swap."""
        if not pair_attempts:
            return {}
        recent_samples = defaultdict(list)
        for pair in pair_attempts:
            client_a = pair["client_a"]
            agent_a = pair["agent_a"]
            client_b = pair["client_b"]
            agent_b = pair["agent_b"]
            expected_a_owner = agent_b
            expected_b_owner = agent_a
            is_accepted = (
                acceptable_part.get(client_a) == expected_a_owner
                and acceptable_part.get(client_b) == expected_b_owner
            )
            label_value = 1.0 if is_accepted else 0.0
            features_a = self._build_agent_swap_features(agent_a, client_a, client_b)
            features_b = self._build_agent_swap_features(agent_b, client_b, client_a)
            label_tensor = torch.tensor([[label_value]], dtype=torch.float32)
            self.swap_training_buffers[agent_a].append((features_a, label_tensor))
            self.swap_training_buffers[agent_b].append((features_b, label_tensor.clone()))
            recent_samples[agent_a].append((features_a, label_tensor))
            recent_samples[agent_b].append((features_b, label_tensor.clone()))
            self._update_agent_behavior_stats(agent_a, client_b, is_accepted)
            self._update_agent_behavior_stats(agent_b, client_a, is_accepted)
            if label_value >= 0.5:
                self._freeze_pair(agent_a, client_a, agent_b, client_b)
                self._mark_deterministic_swap(agent_a, client_a, client_b)
                self._mark_deterministic_swap(agent_b, client_b, client_a)
        return dict(recent_samples)

    def _train_swap_model(self, batch_size: int = 16, recent_samples: Optional[dict] = None) -> Optional[float]:
        """
        Train the logistic regression models using the latest swaps (online update)
        and optionally reinforce with a random batch from each agent's replay buffer.
        """
        losses = []

        if recent_samples:
            for agent_id, samples in recent_samples.items():
                if not samples:
                    continue
                x_recent = torch.cat([sample[0] for sample in samples], dim=0)
                y_recent = torch.cat([sample[1] for sample in samples], dim=0)
                model = self._get_agent_model(agent_id)
                losses.append(model.train_batch(x_recent, y_recent))

        for agent_id, buffer in self.swap_training_buffers.items():
            if len(buffer) >= batch_size:
                buffer_list = list(buffer)
                recent_count = max(1, batch_size // 2)
                recent_samples = buffer_list[-recent_count:]

                remaining = batch_size - len(recent_samples)
                historical_pool = buffer_list[:-recent_count]
                if remaining > 0 and historical_pool:
                    if len(historical_pool) <= remaining:
                        historical_samples = historical_pool
                    else:
                        historical_samples = random.sample(historical_pool, remaining)
                else:
                    historical_samples = []

                batch = recent_samples + historical_samples
                x_batch = torch.cat([item[0] for item in batch], dim=0)
                y_batch = torch.cat([item[1] for item in batch], dim=0)
                model = self._get_agent_model(agent_id)
                losses.append(model.train_batch(x_batch, y_batch))

        if not losses:
            return None
        return sum(losses) / len(losses)

    def _capture_pairs_from_outcome(self, outcome: BasicDARPOutcome):
        """Extract simple client pairs from a heuristic outcome so the model can train during cold start."""
        participant_ids = sorted(self.participants.keys(), key=self._agent_sort_key)
        clients_by_agent = defaultdict(list)
        for client_id, owner in outcome.items():
            clients_by_agent[owner].append(client_id)

        pair_attempts = []
        for i in range(len(participant_ids)):
            agent_a = participant_ids[i]
            for j in range(i + 1, len(participant_ids)):
                agent_b = participant_ids[j]
                for client_a in clients_by_agent.get(agent_a, []):
                    for client_b in clients_by_agent.get(agent_b, []):
                        if self._is_pair_frozen(agent_a, client_a, agent_b, client_b):
                            continue
                        pair_attempts.append({
                            "client_a": client_a,
                            "agent_a": agent_a,
                            "client_b": client_b,
                            "agent_b": agent_b,
                        })

        if pair_attempts:
            self.current_pair_attempts = pair_attempts


    def _is_pair_frozen(self, agent_a: str, client_a: int, agent_b: str, client_b: int) -> bool:
        return self._pair_key(agent_a, client_a, agent_b, client_b) in self.frozen_pair_keys

    def _freeze_pair(self, agent_a: str, client_a: int, agent_b: str, client_b: int):
        key = self._pair_key(agent_a, client_a, agent_b, client_b)
        self.frozen_pair_keys.add(key)

    def _apply_partial_swaps_to_agents(self, swap_decisions):
        """
        After a subset of swaps is accepted, update each affected agent's
        internal DARP problem and utility so the logger can track real cost changes.
        """
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
            # Skip if nothing effectively changes for this agent
            if not including and not excluding:
                continue
            previous_cost = agent.current_utility
            new_cost = agent.get_utility(including=including, excluding=excluding)
            agent.utility_change = previous_cost - new_cost
            agent.current_utility = new_cost
            if agent._last_utility_problem is not None:
                agent.darp_problem = agent._last_utility_problem
                agent._last_utility_problem = None

    # --------------------------------------------------
    # Prenegotiation
    # --------------------------------------------------
    def prenegotiation(self):
        self._reset_learning_state()

        super().prenegotiation()
        self._configure_segment_bucket_size()

        self.improvements = {agent.agent_id: 0 for agent in self.participants.values()}
        self.preferences = {
            agent.agent_id: RankingPreferences(self.domain.known_clients.keys())
            for agent in self.participants.values()
        }

    # --------------------------------------------------
    # Graph Construction
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Negotiation
    # --------------------------------------------------
    def negotiation(self):
        """
        Conduct the negotiation process until full agreement or max rounds reached.
        Integrates learning logic from SingleMediatedTextMechanism.
        """
        previous_outcomes = []
        seen_outcomes = set()
        round_history = []

        for agent in self.participants.values():
            agent.prepare_negotiation()

        round_idx = 0
        while round_idx < self.max_rounds:
            round_number = round_idx + 1
            self.current_round_number = round_number
            print(f"\n🔄 ROUND {round_number}")

            # Sync client ownerships before each round
            for agent in self.participants.values():
                for client_id, current_owner in self.domain.client_owners.items():
                    if client_id in agent._known_clients:
                        agent._client_owners[client_id] = current_owner

            # Generate outcome (model-guided or farthest)
            outcome = self.generate_outcome(method="farthest", round_history=round_history)
            if outcome is None:
                print(f"\n🚨 NEGOTIATION TERMINATED - All combinations explored without agreement")
                self.logger.log_final_state(False, round_number, self.participants, self.domain)
                for agent in self.participants.values():
                    self.routing_logger.log_darp_solution(agent.agent_id, "final", agent.darp_problem)
                self.routing_logger.save_logs()
                return self.logger.save_logs()

            print(f"💡 Proposed outcome: {outcome}")
            outcome_key = tuple(sorted(outcome.items()))
            seen_outcomes.add(outcome_key)

            is_accepted, round_summary = self.round(outcome, round_num=round_number)
            round_history.append(round_summary)
            # Safety fallback: ensure logger attributes exist
            if not hasattr(self.logger, "current_agent_responses"):
                self.logger.current_agent_responses = {agent_id: False for agent_id in self.participants}
            if not hasattr(self.logger, "current_proposed_transfers"):
                self.logger.current_proposed_transfers = []

            self.logger.log_round(round_number, is_accepted, self.participants, round_summary)

            # Termination condition
            if is_accepted:
                print(f"\n🎉 Agreement reached after {round_number} rounds!")
                self.logger.log_final_state(True, round_number, self.participants, self.domain)
                for agent in self.participants.values():
                    self.routing_logger.log_darp_solution(agent.agent_id, "final", agent.darp_problem)
                self.routing_logger.save_logs()
                return self.logger.save_logs()

            round_idx += 1

        print(f"\n⚠️ Maximum rounds ({self.max_rounds}) reached without agreement")
        self.logger.log_final_state(False, self.max_rounds, self.participants, self.domain)
        for agent in self.participants.values():
            self.routing_logger.log_darp_solution(agent.agent_id, "final", agent.darp_problem)
        self.routing_logger.save_logs()
        return self.logger.save_logs()

    # --------------------------------------------------
    # Acceptable Sub-Outcome Extraction
    # --------------------------------------------------
    
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

    # --------------------------------------------------
    # Negotiation Round (with Learning)
    # --------------------------------------------------
    def round(self, outcome, round_num=0):
        if outcome is None:
            outcome = self.generate_outcome()

        involved_agents = self.domain.get_involved_agents(outcome)
        if not involved_agents:
            # Utility logging: collect utility information for all agents when no agents are involved
            agent_responses_for_logging = {}
            for agent_id in self.participants.keys():
                current_utility = self.participants[agent_id].current_utility
                agent_responses_for_logging[agent_id] = {
                    "accepted": False,
                    "proposed_utility": current_utility,
                    "current_utility": current_utility
                }
            replay_buffer_size = self._total_buffer_size()
            return False, {
                "round": round_num, 
                "num_accepts": 0, 
                "num_swaps": 0,
                "agent_responses": agent_responses_for_logging,
                "replay_buffer_size": replay_buffer_size,
            }

        responses = set()
        # Utility logging: collect utility information for all agents
        agent_responses_for_logging = {}
        for agent_id in involved_agents:
            new_clients = self.domain.gets_clients(outcome, agent_id)
            old_clients = self.domain.gives_clients(outcome, agent_id)
            acceptable_condition, temp_utility = self.participants[agent_id].is_acceptable(outcome)
            # Store utility information for logging
            current_utility = self.participants[agent_id].current_utility
            agent_responses_for_logging[agent_id] = {
                "accepted": acceptable_condition,
                "proposed_utility": temp_utility,
                "current_utility": current_utility
            }
            if acceptable_condition:
                responses.add(agent_id)
                if len(new_clients) == 1 and len(old_clients) == 1:
                    self.preferences[agent_id].update_prefers(old_clients[0], new_clients[0])
            else:
                if len(new_clients) == 1 and len(old_clients) == 1:
                    self.preferences[agent_id].update_prefers(new_clients[0], old_clients[0])
        
        # Utility logging: also collect utility information for agents not involved in this outcome
        for agent_id in self.participants.keys():
            if agent_id not in involved_agents:
                current_utility = self.participants[agent_id].current_utility
                agent_responses_for_logging[agent_id] = {
                    "accepted": False,
                    "proposed_utility": current_utility,
                    "current_utility": current_utility
                }

        # Logging
        if hasattr(self, "logger"):
            self.logger.log_outcome(outcome, self.domain)
            self.logger.log_responses(responses, self.participants)

        g = self.build_graph(outcome, responses)
        acceptable_part = self.get_acceptable_part(outcome, g)
        num_accepts = len(responses)
        num_swaps = len(acceptable_part)

        pending_updates = []
        for client_id in acceptable_part:
            new_owner = acceptable_part[client_id]
            self.improvements[new_owner] += 1
            old_owner = self.domain.get_current_owner(client_id)
            pending_updates.append((client_id, old_owner, new_owner))

        pair_attempts = getattr(self, "current_pair_attempts", [])
        recent_samples = self._record_pair_training_examples(
            pair_attempts, acceptable_part, round_num, agent_responses_for_logging
        )
        training_loss = self._train_swap_model(recent_samples=recent_samples)
        self.current_pair_attempts = []

        print(f"[LOG] Round {round_num}: Accepts={num_accepts}, Swaps={num_swaps}")
        if training_loss is not None:
            print(f"[LR-TRAIN] Logistic regression loss={training_loss:.4f}")

        replay_buffer_size = self._total_buffer_size()

        # === APPLY PARTIAL AGREEMENTS ===
        if acceptable_part:
            print(f"Applying {len(acceptable_part)} partial swaps (partial acceptance).")
            self._apply_partial_swaps_to_agents(pending_updates)
            for client_id, _, new_owner in pending_updates:
                self.domain.update_client(client_id, new_owner)

        # === AGREEMENT CHECK ===
        is_full_acceptance = (num_accepts == len(self.participants))
        if is_full_acceptance:
            print("🎉 All agents accepted! Full agreement reached.")
            return True, {
                "round": round_num,
                "num_accepts": num_accepts,
                "num_swaps": num_swaps,
                "agent_responses": agent_responses_for_logging,
                "replay_buffer_size": replay_buffer_size,
            }
        print(f"[BUFFER] Swap dataset size: {self._total_buffer_size()}")

        return False, {
            "round": round_num,
            "num_accepts": num_accepts,
            "num_swaps": num_swaps,
            "agent_responses": agent_responses_for_logging,
            "replay_buffer_size": replay_buffer_size,
        }
