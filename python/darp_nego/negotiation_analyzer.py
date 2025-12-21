import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

class NegotiationAnalyzer:
    """
    Analyzer for negotiation logs with meta-learning metrics.
    Processes multiple case logs to extract performance metrics.
    """
    
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            # logs directory is in same directory as this script
            script_dir = Path(__file__).parent
            self.log_dir = script_dir / "logs"
        else:
            self.log_dir = Path(log_dir)
        
        self.cases = []
        self.df = None
        print(f"Log directory set to: {self.log_dir}")
        
    def load_logs(self, pattern: str = "*/negotiation/*.json") -> None:
        """Load all matching log files from the log directory."""
        print(f"📂 Searching in: {self.log_dir}")
        print(f"🔍 Pattern: {pattern}")
        
        # GLOB ile doğrudan ara
        full_pattern = str(self.log_dir / pattern)
        print(f"🔍 Full pattern: {full_pattern}")
        
        log_files = glob.glob(full_pattern)
        
        print(f"📊 Found {len(log_files)} files\n")
        
        # Debug: Eğer hiç dosya bulunamazsa
        if len(log_files) == 0:
            print("⚠️  No files found! Checking directory structure...")
            
            # logs/ altındaki klasörleri göster
            if self.log_dir.exists():
                case_dirs = [d for d in self.log_dir.iterdir() if d.is_dir()]
                print(f"\n📁 Found {len(case_dirs)} case directories:")
                for case_dir in sorted(case_dirs)[:5]:
                    print(f"  • {case_dir.name}/")
                    # negotiation/ klasörü var mı?
                    nego_dir = case_dir / "negotiation"
                    if nego_dir.exists():
                        json_files = list(nego_dir.glob("*.json"))
                        print(f"    ├─ negotiation/ ({len(json_files)} JSON files)")
                        for jf in json_files[:2]:
                            print(f"    │  └─ {jf.name}")
                    else:
                        print(f"    └─ No negotiation/ subdirectory")
                
                if len(case_dirs) > 5:
                    print(f"  ... and {len(case_dirs) - 5} more directories")
            
            return
        
        # Dosyaları yükle
        for log_file in sorted(log_files):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['log_file'] = log_file
                    data['log_filename'] = os.path.basename(log_file)
                    self.cases.append(data)
                    # Relative path göster
                    rel_path = Path(log_file).relative_to(self.log_dir)
                    print(f"✓ {rel_path}")
            except Exception as e:
                print(f"✗ Error loading {os.path.basename(log_file)}: {e}")
        
        print(f"\n{'='*60}")
        print(f"✓ Successfully loaded {len(self.cases)} cases")
        print(f"{'='*60}\n")
    
    def extract_case_metrics(self, case_data: dict) -> dict:
        """Extract key metrics from a single case."""
        try:
            final_state = case_data.get('final_state', {})
            prenegotiation = case_data.get('prenegotiation', {})
            rounds = case_data.get('rounds', [])
            
            # Get number of companies
            participants = prenegotiation.get('participants', [])
            if not participants:
                initial_utilities = prenegotiation.get('initial_utilities', {})
                participants = list(initial_utilities.keys())
            
            num_companies = len(participants)
            
            if num_companies == 0:
                return None
            
            # Basic negotiation metrics
            metrics = {
                'session_id': case_data.get('session_id', case_data.get('log_filename', 'unknown')),
                'num_companies': num_companies,
                'total_rounds': final_state.get('total_rounds', len(rounds)),
                'agreement_reached': final_state.get('agreement_reached', False),
                'execution_time': final_state.get('execution_time', 0),
                'avg_utility_change': final_state.get('avg_utility_change', 0),
            }
            
            # Initial and final utilities
            initial_utilities = prenegotiation.get('initial_utilities', {})
            final_utilities = final_state.get('final_utilities', initial_utilities)
            
            if initial_utilities:
                total_initial = sum(initial_utilities.values())
                total_final = sum(final_utilities.values())
                metrics['total_utility_change'] = total_final - total_initial
                metrics['total_initial_utility'] = total_initial
                metrics['total_final_utility'] = total_final
            else:
                metrics['total_utility_change'] = 0
                metrics['total_initial_utility'] = 0
                metrics['total_final_utility'] = 0
            
            # Per-company utility changes
            utility_changes = final_state.get('utility_changes', {})
            if utility_changes:
                metrics['max_utility_gain'] = max(utility_changes.values())
                metrics['min_utility_gain'] = min(utility_changes.values())
            else:
                metrics['max_utility_gain'] = 0
                metrics['min_utility_gain'] = 0
            
            # Round-by-round analysis
            if rounds:
                acceptance_counts = []
                swap_counts = []
                
                for round_data in rounds:
                    agent_responses = round_data.get('agent_responses', {})
                    applied_swaps = round_data.get('applied_swaps', [])
                    
                    num_accepts = sum(1 for v in agent_responses.values() if v)
                    acceptance_counts.append(num_accepts)
                    swap_counts.append(len(applied_swaps))
                
                metrics['avg_acceptances_per_round'] = np.mean(acceptance_counts) if acceptance_counts else 0
                metrics['max_acceptances_in_round'] = max(acceptance_counts) if acceptance_counts else 0
                metrics['total_swaps'] = sum(swap_counts)
                metrics['rounds_with_swaps'] = sum(1 for s in swap_counts if s > 0)
                
                # First agreement round (if any)
                full_acceptance_rounds = [
                    i+1 for i, r in enumerate(rounds) 
                    if sum(1 for v in r.get('agent_responses', {}).values() if v) == num_companies
                ]
                metrics['first_full_acceptance_round'] = full_acceptance_rounds[0] if full_acceptance_rounds else None
            else:
                metrics['avg_acceptances_per_round'] = 0
                metrics['max_acceptances_in_round'] = 0
                metrics['total_swaps'] = 0
                metrics['rounds_with_swaps'] = 0
                metrics['first_full_acceptance_round'] = None
            
            return metrics
        
        except Exception as e:
            print(f"Error extracting metrics: {e}")
            return None
    
    def create_dataframe(self) -> pd.DataFrame:
        """Convert all cases to a pandas DataFrame."""
        all_metrics = []
        for i, case in enumerate(self.cases):
            print(f"Processing case {i+1}/{len(self.cases)}...", end='\r')
            metrics = self.extract_case_metrics(case)
            if metrics is not None:
                all_metrics.append(metrics)
        
        print(f"\n✓ Extracted metrics from {len(all_metrics)} cases")
        
        if not all_metrics:
            print("ERROR: No valid metrics extracted!")
            return pd.DataFrame()
        
        self.df = pd.DataFrame(all_metrics)
        print(f"✓ Created DataFrame with shape: {self.df.shape}")
        return self.df
    
    def aggregate_by_companies(self) -> pd.DataFrame:
        """Aggregate results by number of companies."""
        if self.df is None or self.df.empty:
            self.create_dataframe()
        
        if self.df.empty:
            return pd.DataFrame()
        
        agg_df = self.df.groupby('num_companies').agg({
            'session_id': 'count',
            'total_rounds': 'mean',
            'agreement_reached': lambda x: (x.sum() / len(x)) * 100,
            'avg_utility_change': 'mean',
            'execution_time': 'mean',
            'total_swaps': 'mean',
            'first_full_acceptance_round': 'mean'
        }).round(2)
        
        agg_df.columns = [
            'Total Cases',
            'Avg. Rounds',
            'Agreement Rate (%)',
            'Avg. Utility Change',
            'Avg. Execution Time (s)',
            'Avg. Total Swaps',
            'Avg. First Agreement Round'
        ]
        
        return agg_df
    
    def analyze_learning_progression(self) -> pd.DataFrame:
        """Analyze how performance changes over sequential cases."""
        if self.df is None or self.df.empty:
            self.create_dataframe()
        
        if self.df.empty:
            return pd.DataFrame()
        
        df_sorted = self.df.sort_values('session_id').copy()
        df_sorted['case_number'] = range(1, len(df_sorted) + 1)
        
        max_case = len(df_sorted)
        if max_case <= 5:
            bins = [0, max_case]
            labels = [f'Cases 1-{max_case}']
        elif max_case <= 10:
            bins = [0, 5, max_case]
            labels = ['Cases 1-5', f'Cases 6-{max_case}']
        elif max_case <= 15:
            bins = [0, 5, 10, max_case]
            labels = ['Cases 1-5', 'Cases 6-10', f'Cases 11-{max_case}']
        elif max_case <= 20:
            bins = [0, 5, 10, 15, max_case]
            labels = ['Cases 1-5', 'Cases 6-10', 'Cases 11-15', f'Cases 16-{max_case}']
        else:
            bins = [0, 5, 10, 15, 20, max_case]
            labels = ['Cases 1-5', 'Cases 6-10', 'Cases 11-15', 'Cases 16-20', f'Cases 20+']
        
        df_sorted['case_range'] = pd.cut(df_sorted['case_number'], bins=bins, labels=labels)
        
        learning_df = df_sorted.groupby('case_range').agg({
            'total_rounds': ['mean', 'std'],
            'agreement_reached': lambda x: (x.sum() / len(x)) * 100,
            'avg_utility_change': ['mean', 'std'],
            'first_full_acceptance_round': 'mean'
        }).round(2)
        
        return learning_df
    
    def extract_round_details(self) -> pd.DataFrame:
        """Extract detailed round-by-round information."""
        round_records = []
        
        for case in self.cases:
            session_id = case.get('session_id', case.get('log_filename', 'unknown'))
            prenegotiation = case.get('prenegotiation', {})
            participants = prenegotiation.get('participants', [])
            if not participants:
                initial_utilities = prenegotiation.get('initial_utilities', {})
                participants = list(initial_utilities.keys())
            
            num_companies = len(participants)
            rounds = case.get('rounds', [])
            
            for round_data in rounds:
                round_num = round_data.get('round_number', 0)
                agent_responses = round_data.get('agent_responses', {})
                applied_swaps = round_data.get('applied_swaps', [])
                proposed_transfers = round_data.get('proposed_transfers', [])
                
                num_accepts = sum(1 for v in agent_responses.values() if v)
                num_swaps = len(applied_swaps)
                num_proposals = len(proposed_transfers)
                
                round_records.append({
                    'session_id': session_id,
                    'num_companies': num_companies,
                    'round_number': round_num,
                    'num_accepts': num_accepts,
                    'num_swaps': num_swaps,
                    'num_proposals': num_proposals,
                    'acceptance_ratio': num_accepts / max(1, num_companies),
                    'full_acceptance': num_accepts == num_companies
                })
        
        return pd.DataFrame(round_records)
    
    def plot_learning_curve(self, output_path: str = "learning_curve.png"):
        """Plot learning progression."""
        if self.df is None or self.df.empty:
            return
        
        df_sorted = self.df.sort_values('session_id').copy()
        df_sorted['case_number'] = range(1, len(df_sorted) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(df_sorted['case_number'], df_sorted['total_rounds'], marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Case Number')
        axes[0, 0].set_ylabel('Total Rounds')
        axes[0, 0].set_title('Negotiation Rounds Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        window = min(5, len(df_sorted))
        df_sorted['agreement_rolling'] = df_sorted['agreement_reached'].rolling(window=window, min_periods=1).mean()
        axes[0, 1].plot(df_sorted['case_number'], df_sorted['agreement_rolling'], marker='o', linewidth=2, markersize=6, color='green')
        axes[0, 1].set_xlabel('Case Number')
        axes[0, 1].set_ylabel(f'Agreement Rate (Rolling {window})')
        axes[0, 1].set_title('Agreement Rate Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(df_sorted['case_number'], df_sorted['avg_utility_change'], marker='o', linewidth=2, markersize=6, color='purple')
        axes[1, 0].set_xlabel('Case Number')
        axes[1, 0].set_ylabel('Avg. Utility Change')
        axes[1, 0].set_title('Utility Changes Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df_sorted['case_number'], df_sorted['total_swaps'], marker='o', linewidth=2, markersize=6, color='orange')
        axes[1, 1].set_xlabel('Case Number')
        axes[1, 1].set_ylabel('Total Swaps')
        axes[1, 1].set_title('Client Swaps Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Learning curve saved to {output_path}")
        plt.close()
    
    def plot_acceptance_heatmap(self, output_path: str = "acceptance_heatmap.png"):
        """Plot heatmap of acceptance rates."""
        round_df = self.extract_round_details()
        if round_df.empty:
            return
        
        max_cases = 20
        unique_sessions = round_df['session_id'].unique()[:max_cases]
        round_df_filtered = round_df[round_df['session_id'].isin(unique_sessions)]
        
        pivot_data = round_df_filtered.pivot_table(values='acceptance_ratio', index='round_number', columns='session_id', aggfunc='mean')
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot_data, cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': 'Acceptance Ratio'}, xticklabels=True, yticklabels=True)
        plt.xlabel('Case (Session ID)')
        plt.ylabel('Round Number')
        plt.title('Acceptance Ratio Heatmap')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap saved to {output_path}")
        plt.close()

    def save_results(self, output_dir: str = "analysis_results"):
        """Save all results."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.df is None or self.df.empty:
            return
        
        agg_df = self.aggregate_by_companies()
        if not agg_df.empty:
            agg_df.to_csv(os.path.join(output_dir, "aggregated_by_companies.csv"))
            print(f"✓ Saved: aggregated_by_companies.csv")
        
        learning_df = self.analyze_learning_progression()
        if not learning_df.empty:
            learning_df.to_csv(os.path.join(output_dir, "learning_progression.csv"))
            print(f"✓ Saved: learning_progression.csv")
        
        round_df = self.extract_round_details()
        if not round_df.empty:
            round_df.to_csv(os.path.join(output_dir, "round_details.csv"), index=False)
            print(f"✓ Saved: round_details.csv")
        
        self.df.to_csv(os.path.join(output_dir, "all_cases.csv"), index=False)
        print(f"✓ Saved: all_cases.csv")
        
        self.plot_learning_curve(os.path.join(output_dir, "learning_curve.png"))
        self.plot_acceptance_heatmap(os.path.join(output_dir, "acceptance_heatmap.png"))
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_dir}/")
        print(f"{'='*60}\n")
        if not agg_df.empty:
            print(agg_df)
        print(f"{'='*60}\n")
def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("NEGOTIATION LOG ANALYZER")
    print("=" * 60)

    # script dizini -> python\darp_nego
    script_dir = Path(__file__).parent

    # Bir üst dizine çık (D:\Documents\MSc\darp_nego\darp_nego)
    project_root = script_dir.parent.parent

    # Oradaki logs klasörünü hedef al
    log_dir = project_root / "logs"

    print(f"📁 Using log directory: {log_dir}")

    analyzer = NegotiationAnalyzer(log_dir=str(log_dir))

    # senin klasör yapına göre pattern
    analyzer.load_logs(pattern="llm_generated_case_*/negotiation/*.json")

    if len(analyzer.cases) == 0:
        print("\n❌ No log files found!")
        return

    analyzer.create_dataframe()
    analyzer.save_results(output_dir="analysis_results")
    print("\n✓ Analysis complete!")


def summarize_agreements(log_dir):
    """Count how many negotiation cases reached an agreement."""
    log_dir = Path(log_dir)
    json_files = list(log_dir.rglob("*.json"))

    total = len(json_files)
    if total == 0:
        print(f"⚠️ No JSON log files found in {log_dir}")
        return

    agreements = 0
    no_agreements = 0

    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("final_state", {}).get("agreement_reached") is True:
                agreements += 1
            else:
                no_agreements += 1
        except Exception as e:
            print(f"⚠️ Error reading {file.name}: {e}")

    print("=" * 60)
    print(f"📊 Agreement Summary from {total} logs")
    print(f"🤝 Agreements reached: {agreements}")
    print(f"🚫 No agreement:       {no_agreements}")
    print(f"📈 Success rate:       {agreements / total * 100:.1f}%")
    print("=" * 60)

    
if __name__ == "__main__":
    main()
    
    summarize_agreements(r"D:\Documents\MSc\darp_nego\darp_nego\logs")