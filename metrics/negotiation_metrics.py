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


@dataclass
class NegotiationMetricsSummary:
    """Summary of calculated metrics for a negotiation session"""
    # Efficiency metrics
    social_welfare: float  # Sum of all agent utilities
    avg_utility_change: float  # Average utility improvement across agents
    pareto_optimal: bool  # Whether the solution is Pareto optimal
    
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
    
    def __repr__(self) -> str:
        """Print-friendly representation of metrics summary"""
        return (
            f"=== Negotiation Metrics Summary ===\n"
            f"Agreement reached: {self.agreement_reached} in {self.rounds_to_agreement} rounds\n"
            f"Social welfare: {self.social_welfare:.2f}\n"
            f"Avg utility change: {self.avg_utility_change:.2f}\n"
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
        
        # Extract key data
        final_state = self.log_data["final_state"]
        rounds = self.log_data["rounds"]
        
        # Calculate metrics by category
        efficiency_metrics = self._calculate_efficiency_metrics()
        fairness_metrics = self._calculate_fairness_metrics()
        process_metrics = self._calculate_process_metrics()
        
        # Create and return summary
        return NegotiationMetricsSummary(
            # Efficiency metrics
            social_welfare=sum(final_state["final_utilities"].values()),
            avg_utility_change=final_state["avg_utility_change"],
            pareto_optimal=efficiency_metrics["pareto_optimal"],
            
            # Fairness metrics
            utility_distribution=final_state["final_utilities"],
            gini_coefficient=fairness_metrics["gini_coefficient"],
            min_max_ratio=fairness_metrics["min_max_ratio"],
            
            # Process metrics
            rounds_to_agreement=final_state["total_rounds"],
            agreement_reached=final_state["agreement_reached"],
            full_acceptance_rate=process_metrics["full_acceptance_rate"],
            partial_acceptance_rate=process_metrics["partial_acceptance_rate"],
            total_acceptance_rate=process_metrics["total_acceptance_rate"],
            partial_swaps_count=process_metrics["partial_swaps_count"],
            participating_agents=process_metrics["participating_agents"],
            
            # Time metrics
            execution_time=final_state["execution_time"]
        )
    
    def _calculate_efficiency_metrics(self) -> Dict:
        """Calculate efficiency-related metrics"""
        final_state = self.log_data["final_state"]
        
        # Simple check for Pareto optimality (not comprehensive)
        # A real implementation would need to check all possible alternatives
        pareto_optimal = True
        
        return {
            "pareto_optimal": pareto_optimal
        }
    
    def _calculate_fairness_metrics(self) -> Dict:
        """Calculate fairness-related metrics"""
        final_state = self.log_data["final_state"]
        utilities = list(final_state["final_utilities"].values())
        
        # Calculate Gini coefficient (measure of inequality)
        sorted_utilities = sorted(utilities)
        height = sum(sorted_utilities)
        area = 0
        for i, utility in enumerate(sorted_utilities):
            area += utility * (len(utilities) - i)
        fair_area = height * len(utilities) / 2
        gini = 1 - area / fair_area if fair_area > 0 else 0
        
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
        
        # Calculate acceptance rates.
        # Backward compatibility:
        # - old logs may only have "is_accepted" (full acceptance)
        # - new logs carry "full_acceptance"/"partial_acceptance"/"num_swaps"
        full_acceptances = sum(
            1 for round_data in rounds
            if bool(round_data.get("full_acceptance", round_data.get("is_accepted", False)))
        )
        partial_acceptances = sum(
            1 for round_data in rounds
            if bool(round_data.get("partial_acceptance", False))
        )
        full_acceptance_rate = full_acceptances / total_rounds if total_rounds > 0 else 0
        partial_acceptance_rate = partial_acceptances / total_rounds if total_rounds > 0 else 0
        total_acceptance_rate = (full_acceptances + partial_acceptances) / total_rounds if total_rounds > 0 else 0

        # Count swaps applied in partial-acceptance rounds
        partial_swaps_count = sum(
            int(round_data.get("num_swaps", 0) or 0)
            for round_data in rounds
            if bool(round_data.get("partial_acceptance", False))
        )

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
            "full_acceptance_rate": full_acceptance_rate,
            "partial_acceptance_rate": partial_acceptance_rate,
            "total_acceptance_rate": total_acceptance_rate,
            "partial_swaps_count": partial_swaps_count,
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
