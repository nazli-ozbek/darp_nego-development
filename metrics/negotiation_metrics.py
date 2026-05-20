"""
Metrics calculation system for DARP negotiation analysis.
Provides standardized KPIs and analysis tools for negotiation outcomes.s
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


def standard_gini(values: List[float]) -> float:
    """Standard Gini coefficient for non-negative values."""
    if not values:
        return 0.0
    values = [float(value) for value in values]
    mean_value = sum(values) / len(values)
    if mean_value == 0:
        return 0.0
    absolute_diffs = sum(abs(x_i - x_j) for x_i in values for x_j in values)
    return absolute_diffs / (2 * len(values) ** 2 * mean_value)


def pareto_improvement_from_cost_deltas(cost_deltas: Dict[str, float]) -> bool:
    """Cost-minimization Pareto improvement: no one worse, at least one better."""
    if not cost_deltas:
        return False
    return all(delta <= 0 for delta in cost_deltas.values()) and any(
        delta < 0 for delta in cost_deltas.values()
    )


def _final_cost_delta(final_state: Dict[str, Any], prenegotiation: Dict[str, Any]) -> Dict[str, float]:
    if "final_cost_delta_by_agent" in final_state:
        return dict(final_state["final_cost_delta_by_agent"])
    initial = final_state.get("initial_utilities") or prenegotiation.get("initial_utilities", {})
    final = final_state.get("final_utilities", {})
    return {agent_id: final[agent_id] - initial[agent_id] for agent_id in final if agent_id in initial}


def _final_cost_saving(final_state: Dict[str, Any], prenegotiation: Dict[str, Any]) -> Dict[str, float]:
    if "final_cost_saving_by_agent" in final_state:
        return dict(final_state["final_cost_saving_by_agent"])
    return {agent_id: -delta for agent_id, delta in _final_cost_delta(final_state, prenegotiation).items()}


def _applied_swap_count(round_data: Dict[str, Any]) -> int:
    applied_swaps = round_data.get("applied_swaps", [])
    if applied_swaps:
        return len([swap for swap in applied_swaps if swap.get("from") != swap.get("to")])
    return int(round_data.get("applied_swap_count", round_data.get("num_swaps", 0)) or 0)


def calculate_canonical_metrics_from_log(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return canonical negotiation metrics from a v1 or v2 negotiation log."""
    final_state = log_data.get("final_state", {})
    prenegotiation = log_data.get("prenegotiation", {})
    rounds = log_data.get("rounds", [])
    final_utilities = final_state.get("final_utilities", {})
    initial_utilities = final_state.get("initial_utilities") or prenegotiation.get("initial_utilities", {})
    cost_delta = _final_cost_delta(final_state, prenegotiation)
    cost_saving = _final_cost_saving(final_state, prenegotiation)
    total_cost = sum(final_utilities.values())
    total_initial_cost = sum(initial_utilities.values())
    total_cost_delta = sum(cost_delta.values())
    total_cost_saving = sum(cost_saving.values())
    full_acceptances = sum(
        1 for round_data in rounds
        if bool(round_data.get("full_acceptance", round_data.get("is_accepted", False)))
    )
    partial_acceptances = sum(1 for round_data in rounds if bool(round_data.get("partial_acceptance", False)))
    total_rounds = len(rounds)
    partial_swaps_count = sum(
        _applied_swap_count(round_data)
        for round_data in rounds
        if bool(round_data.get("partial_acceptance", False))
    )
    total_swaps_applied = sum(_applied_swap_count(round_data) for round_data in rounds)
    return {
        "total_cost": total_cost,
        "total_initial_cost": total_initial_cost,
        "total_cost_delta": total_cost_delta,
        "total_cost_saving": total_cost_saving,
        "avg_cost_delta": (total_cost_delta / len(cost_delta)) if cost_delta else 0,
        "avg_cost_saving": (total_cost_saving / len(cost_saving)) if cost_saving else 0,
        "final_cost_delta_by_agent": cost_delta,
        "final_cost_saving_by_agent": cost_saving,
        "pareto_improvement": pareto_improvement_from_cost_deltas(cost_delta),
        "gini_coefficient": standard_gini(list(final_utilities.values())),
        "min_max_ratio": (
            min(final_utilities.values()) / max(final_utilities.values())
            if final_utilities and max(final_utilities.values()) > 0 else 1
        ),
        "rounds_to_agreement": final_state.get("total_rounds", total_rounds),
        "agreement_reached": bool(final_state.get("agreement_reached", False)),
        "full_acceptance_rate": full_acceptances / total_rounds if total_rounds else 0,
        "partial_acceptance_rate": partial_acceptances / total_rounds if total_rounds else 0,
        "total_acceptance_rate": (full_acceptances + partial_acceptances) / total_rounds if total_rounds else 0,
        "partial_swaps_count": partial_swaps_count,
        "total_swaps_applied": total_swaps_applied,
        "execution_time": final_state.get("execution_time", log_data.get("execution_time", 0)),
    }


@dataclass
class NegotiationMetricsSummary:
    """Summary of calculated metrics for a negotiation session"""
    # Efficiency metrics
    total_cost: float  # Sum of final route costs; lower is better
    total_cost_saving: float  # Sum(initial_costs) - sum(final_costs); higher is better
    avg_cost_delta: float  # Average final - initial cost; negative is better
    avg_cost_saving: float  # Average initial - final cost; positive is better
    pareto_improvement: bool  # No agent worse and at least one agent better
    
    # Fairness metrics
    utility_distribution: Dict[str, float]  # Final utility for each agent
    gini_coefficient: float  # Measure of equality (0=perfect equality, 1=perfect inequality)
    min_max_ratio: float  # Ratio of min utility to max utility
    
    # Process metrics
    rounds_to_agreement: int  # Number of rounds until agreement (or max if no agreement)
    agreement_reached: bool  # Whether agreement was reached
    full_acceptance_rate: float  # Percentage of proposals fully accepted
    partial_acceptance_rate: float  # Percentage of proposals partially accepted
    total_acceptance_rate: float  # Percentage of proposals with full or partial acceptance
    partial_swaps_count: int  # Total number of swaps applied through partial acceptance
    participating_agents: Dict[str, int]  # Participation count in applied partial swaps by agent
    
    # Time metrics
    execution_time: float  # Total execution time in seconds

    @property
    def social_welfare(self) -> float:
        """Deprecated alias kept for old consumers; this is actually total cost."""
        return self.total_cost

    @property
    def avg_utility_change(self) -> float:
        """Deprecated alias for avg_cost_delta."""
        return self.avg_cost_delta

    @property
    def pareto_optimal(self) -> bool:
        """Deprecated alias for pareto_improvement."""
        return self.pareto_improvement
    
    def __repr__(self) -> str:
        """Print-friendly representation of metrics summary"""
        return (
            f"=== Negotiation Metrics Summary ===\n"
            f"Agreement reached: {self.agreement_reached} in {self.rounds_to_agreement} rounds\n"
            f"Total cost: {self.total_cost:.2f}\n"
            f"Total cost saving: {self.total_cost_saving:.2f}\n"
            f"Avg cost delta: {self.avg_cost_delta:.2f} (negative is better)\n"
            f"Avg cost saving: {self.avg_cost_saving:.2f} (positive is better)\n"
            f"Pareto improvement: {self.pareto_improvement}\n"
            f"Gini coefficient: {self.gini_coefficient:.4f} (0=equal, 1=unequal)\n"
            f"Min/Max utility ratio: {self.min_max_ratio:.4f}\n"
            f"Full acceptance rate: {self.full_acceptance_rate:.2%}\n"
            f"Partial acceptance rate: {self.partial_acceptance_rate:.2%}\n"
            f"Total acceptance rate: {self.total_acceptance_rate:.2%}\n"
            f"Partial swaps count: {self.partial_swaps_count}\n"
            f"Execution time: {self.execution_time:.2f}s"
        )


class DARPNegotiationMetrics:
    """
    Calculates and analyzes metrics from DARP negotiation logs.
    
    This class provides methods to:
    1. Load negotiation logs
    2. Calculate various performance metrics
    3. Visualize negotiation outcomes
    4. Compare multiple negotiation strategies
    """
    
    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize metrics calculator, optionally with a log file.
        
        Args:
            log_path: Path to negotiation log JSON file (optional)
        """
        self.log_data = None
        if log_path:
            self.load_log(log_path)
    
    def load_log(self, log_path: str) -> None:
        """
        Load negotiation log data from a JSON file.
        
        Args:
            log_path: Path to negotiation log JSON file
        """
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        with open(log_path, 'r') as f:
            self.log_data = json.load(f)
    
    def calculate_metrics(self) -> NegotiationMetricsSummary:
        """
        Calculate comprehensive metrics from loaded negotiation log.
        
        Returns:
            NegotiationMetricsSummary containing all calculated metrics
        
        Raises:
            ValueError: If no log data has been loaded
        """
        if not self.log_data:
            raise ValueError("No log data loaded. Call load_log() first.")
        
        final_state = self.log_data["final_state"]
        canonical = calculate_canonical_metrics_from_log(self.log_data)
        
        # Calculate metrics by category
        fairness_metrics = self._calculate_fairness_metrics()
        process_metrics = self._calculate_process_metrics()
        
        # Create and return summary
        return NegotiationMetricsSummary(
            # Efficiency metrics
            total_cost=canonical["total_cost"],
            total_cost_saving=canonical["total_cost_saving"],
            avg_cost_delta=canonical["avg_cost_delta"],
            avg_cost_saving=canonical["avg_cost_saving"],
            pareto_improvement=canonical["pareto_improvement"],
            
            # Fairness metrics
            utility_distribution=final_state["final_utilities"],
            gini_coefficient=fairness_metrics["gini_coefficient"],
            min_max_ratio=fairness_metrics["min_max_ratio"],
            
            # Process metrics
            rounds_to_agreement=canonical["rounds_to_agreement"],
            agreement_reached=canonical["agreement_reached"],
            full_acceptance_rate=process_metrics["full_acceptance_rate"],
            partial_acceptance_rate=process_metrics["partial_acceptance_rate"],
            total_acceptance_rate=process_metrics["total_acceptance_rate"],
            partial_swaps_count=process_metrics["partial_swaps_count"],
            participating_agents=process_metrics["participating_agents"],
            
            # Time metrics
            execution_time=canonical["execution_time"]
        )
    
    def _calculate_efficiency_metrics(self) -> Dict:
        """Calculate efficiency-related metrics"""
        final_state = self.log_data["final_state"]
        
        canonical = calculate_canonical_metrics_from_log(self.log_data)
        return {"pareto_improvement": canonical["pareto_improvement"]}
    
    def _calculate_fairness_metrics(self) -> Dict:
        """Calculate fairness-related metrics"""
        final_state = self.log_data["final_state"]
        utilities = list(final_state["final_utilities"].values())
        
        gini = standard_gini(utilities)
        
        # Calculate min/max ratio (1.0 = perfectly equal)
        max_utility = max(utilities) if utilities else 1
        min_utility = min(utilities) if utilities else 0
        min_max_ratio = min_utility / max_utility if max_utility > 0 else 1
        
        return {
            "gini_coefficient": gini,
            "min_max_ratio": min_max_ratio
        }
    
    def _calculate_process_metrics(self) -> Dict:
        """Calculate process-related metrics"""
        if not self.log_data:
            raise ValueError("No log data loaded. Call load_log() first.")
        
        rounds = self.log_data["rounds"]
        total_rounds = len(rounds)
        
        canonical = calculate_canonical_metrics_from_log(self.log_data)

        # Agent participation counts in applied partial swaps
        participant_counts = {agent_id: 0 for agent_id in self.log_data["prenegotiation"]["participants"]}
        for round_data in rounds:
            if not bool(round_data.get("partial_acceptance", False)):
                continue
            for applied in round_data.get("applied_swaps", []):
                from_agent = str(applied.get("from"))
                to_agent = str(applied.get("to"))
                if from_agent in participant_counts:
                    participant_counts[from_agent] += 1
                if to_agent in participant_counts:
                    participant_counts[to_agent] += 1
        
        return {
            "full_acceptance_rate": canonical["full_acceptance_rate"],
            "partial_acceptance_rate": canonical["partial_acceptance_rate"],
            "total_acceptance_rate": canonical["total_acceptance_rate"],
            "partial_swaps_count": canonical["partial_swaps_count"],
            "participating_agents": participant_counts
        }
    
    def plot_utility_changes(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Plot the utility changes over negotiation rounds.
        
        Args:
            save_path: Optional file path to save the plot
            show: Whether to show the plot
        """
        if not self.log_data:
            raise ValueError("No log data loaded. Call load_log() first.")
        
        rounds = self.log_data["rounds"]
        participants = self.log_data["prenegotiation"]["participants"]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Initialize utility tracking with initial values
        utilities = {agent: [self.log_data["prenegotiation"]["initial_utilities"][agent]] for agent in participants}
        
        # Track utilities over rounds using the actual current_utilities data from logs
        for round_data in rounds:
            for agent in participants:
                if round_data.get("full_acceptance", False) or round_data.get("partial_acceptance", False):
                    current_util = round_data["proposed_utilities"].get(agent, round_data["current_utilities"][agent])
                else:
                    current_util = round_data["current_utilities"][agent]
                utilities[agent].append(current_util)
        
        # Plot each agent's utility
        round_numbers = list(range(len(rounds) + 1))  # Include round 0
        for agent, utils in utilities.items():
            plt.plot(round_numbers, utils, marker='o', linestyle='-', label=f"Agent {agent}")
        
        # Add plot details
        plt.title("Agent Utilities Over Negotiation Rounds")
        plt.xlabel("Round")
        plt.ylabel("Utility")
        plt.xticks(round_numbers)  # Show all round numbers
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
    
    def plot_acceptance_heatmap(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Plot a heatmap showing which agents accepted/rejected proposals.
        
        Args:
            save_path: Optional file path to save the plot
            show: Whether to show the plot
        """
        if not self.log_data:
            raise ValueError("No log data loaded. Call load_log() first.")
        
        rounds = self.log_data["rounds"]
        participants = self.log_data["prenegotiation"]["participants"]
        
        # Create acceptance matrix
        acceptance_matrix = []
        for round_data in rounds:
            row = []
            for agent in participants:
                accepted = 1 if round_data["agent_responses"].get(agent, False) else 0
                row.append(accepted)
            acceptance_matrix.append(row)
        
        # Convert to numpy array for heatmap
        acceptance_matrix = np.array(acceptance_matrix)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(acceptance_matrix, cmap='RdYlGn', aspect='auto')
        plt.colorbar(ticks=[0, 1], label='Acceptance')
        
        # Add labels
        plt.title('Agent Acceptance by Round')
        plt.xlabel('Agents')
        plt.ylabel('Round')
        plt.yticks(range(len(rounds)), range(1, len(rounds) + 1))
        plt.xticks(range(len(participants)), [f"Agent {agent}" for agent in participants])
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
    
    def plot_partial_acceptance_analysis(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """
        Plot analysis of acceptance patterns (full, partial, rejection).
        
        Args:
            save_path: Optional file path to save the plot
            show: Whether to show the plot
        """
        if not self.log_data:
            raise ValueError("No log data loaded. Call load_log() first.")
        
        rounds = self.log_data["rounds"]
        participants = self.log_data["prenegotiation"]["participants"]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        process_metrics = self._calculate_process_metrics()
        participant_counts = process_metrics["participating_agents"]
        
        agents = list(participant_counts.keys())
        counts = list(participant_counts.values())
        
        ax1.bar(agents, counts, color='lightgray')
        ax1.set_title('Participation in Partial Swaps by Agent')
        ax1.set_xlabel('Agent ID')
        ax1.set_ylabel('Number of Partial Swaps')
        for i, v in enumerate(counts):
            ax1.text(i, v + 0.1, str(v), ha='center')
        
        # 2. Pie chart of full/partial/rejections
        full_acceptances = sum(
            1 for round_data in rounds
            if bool(round_data.get("full_acceptance", round_data.get("is_accepted", False)))
        )
        partial_acceptances = sum(1 for round_data in rounds if bool(round_data.get("partial_acceptance", False)))
        rejections = len(rounds) - full_acceptances - partial_acceptances

        acceptance_labels = ['Full Acceptances', 'Partial Acceptances', 'Rejections']
        acceptance_data = [full_acceptances, partial_acceptances, rejections]
        colors = ['#66b3ff', '#ffd166', '#ff9999']
        
        ax2.pie(acceptance_data, labels=acceptance_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Proposal Outcomes')
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def compare_logs(log_paths: List[str], labels: Optional[List[str]] = None) -> Dict[str, List[NegotiationMetricsSummary]]:
        """
        Compare metrics across multiple negotiation logs.
        
        Args:
            log_paths: List of paths to log files
            labels: Optional list of labels for each log
        
        Returns:
            Dictionary mapping labels to metric summaries
        """
        if not labels:
            labels = [f"Negotiation {i+1}" for i in range(len(log_paths))]
        
        if len(labels) != len(log_paths):
            raise ValueError("Number of labels must match number of log paths")
        
        results = {}
        for label, path in zip(labels, log_paths):
            calculator = DARPNegotiationMetrics(path)
            results[label] = calculator.calculate_metrics()
        
        return results
    
    def save_metrics(
        self,
        output_dir: str = "metrics_output",
        run_id: Optional[str] = None,
        save_plots: bool = True,
        save_text: bool = True,
    ) -> Dict[str, str]:
        """
        Save metrics outputs to a dedicated folder with unique names.
        
        Args:
            output_dir: Base directory for metrics output
            run_id: Optional unique identifier for this run (timestamp used if not provided)
            save_plots: Whether to generate per-session PNG plots
            save_text: Whether to write the human-readable text summary
        
        Returns:
            Dictionary mapping output types to their file paths
        """
        from datetime import datetime
        
        # Create unique run ID if not provided
        if not run_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            agents_count = len(self.log_data["prenegotiation"]["participants"])
            rounds_count = len(self.log_data["rounds"])
            agreement = "success" if self.log_data["final_state"]["agreement_reached"] else "failed"
            run_id = f"{timestamp}_{agents_count}agents_{rounds_count}rounds_{agreement}"
        
        # Create output directory structure - use run_id as the session folder
        run_dir = os.path.join(output_dir, run_id, "negotiation")
        os.makedirs(run_dir, exist_ok=True)
        
        # Calculate metrics
        metrics_summary = self.calculate_metrics()
        
        # Save summary to JSON
        summary_path = os.path.join(run_dir, "metrics_summary.json")
        with open(summary_path, 'w') as f:
            # Convert summary to dictionary for JSON serialization
            summary_dict = {
                "total_cost": metrics_summary.total_cost,
                "total_cost_saving": metrics_summary.total_cost_saving,
                "avg_cost_delta": metrics_summary.avg_cost_delta,
                "avg_cost_saving": metrics_summary.avg_cost_saving,
                "pareto_improvement": metrics_summary.pareto_improvement,
                "social_welfare": metrics_summary.social_welfare,
                "avg_utility_change": metrics_summary.avg_utility_change,
                "pareto_optimal": metrics_summary.pareto_optimal,
                "utility_distribution": metrics_summary.utility_distribution,
                "gini_coefficient": metrics_summary.gini_coefficient,
                "min_max_ratio": metrics_summary.min_max_ratio,
                "rounds_to_agreement": metrics_summary.rounds_to_agreement,
                "agreement_reached": metrics_summary.agreement_reached,
                "full_acceptance_rate": metrics_summary.full_acceptance_rate,
                "partial_acceptance_rate": metrics_summary.partial_acceptance_rate,
                "total_acceptance_rate": metrics_summary.total_acceptance_rate,
                "partial_swaps_count": metrics_summary.partial_swaps_count,
                "execution_time": metrics_summary.execution_time
            }
            json.dump(summary_dict, f, indent=2)
        
        outputs = {
            "directory": run_dir,
            "summary_json": summary_path,
        }

        if save_text:
            summary_txt_path = os.path.join(run_dir, "metrics_summary.txt")
            with open(summary_txt_path, 'w') as f:
                f.write(str(metrics_summary))
            outputs["summary_txt"] = summary_txt_path
        
        if save_plots:
            utility_plot_path = os.path.join(run_dir, "utility_changes.png")
            self.plot_utility_changes(save_path=utility_plot_path, show=False)
            
            heatmap_path = os.path.join(run_dir, "acceptance_heatmap.png")
            self.plot_acceptance_heatmap(save_path=heatmap_path, show=False)
            
            partial_acceptance_path = os.path.join(run_dir, "partial_acceptance_analysis.png")
            self.plot_partial_acceptance_analysis(save_path=partial_acceptance_path, show=False)

            outputs.update({
                "utility_plot": utility_plot_path,
                "acceptance_heatmap": heatmap_path,
                "partial_acceptance_plot": partial_acceptance_path,
            })
        
        print(f"\nNegotiation metrics saved to: {os.path.abspath(run_dir)}")
        
        return outputs
