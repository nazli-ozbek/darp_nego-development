#!/usr/bin/env python3
"""
Script to find cases with full_acceptance in negotiation JSON files.
Scans logs directory for folders matching llm_generated* pattern and checks
negotiation JSON files for full_acceptance cases.
"""

import json
import os
import glob
from pathlib import Path


def find_full_acceptance_cases(logs_dir="darp_nego/logs"):
    """
    Find all cases with full_acceptance in negotiation JSON files.
    
    Args:
        logs_dir: Path to the logs directory
        
    Returns:
        List of tuples (case_folder, session_id, full_acceptance_info)
    """
    cases_with_acceptance = []
    
    # Find all directories matching llm_generated* pattern
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"Error: Logs directory '{logs_dir}' not found!")
        return cases_with_acceptance
    
    # Get all directories matching the pattern
    pattern_dirs = sorted(logs_path.glob("llm_generated*"))
    
    print(f"Found {len(pattern_dirs)} directories matching pattern 'llm_generated*'")
    print("-" * 80)
    
    for case_dir in pattern_dirs:
        # Check if negotiation subdirectory exists
        negotiation_dir = case_dir / "negotiation"
        if not negotiation_dir.exists():
            continue
        
        # Find JSON files in negotiation directory
        json_files = list(negotiation_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                session_id = data.get("session_id", "unknown")
                
                # Get number of agents (companies)
                prenegotiation = data.get("prenegotiation", {})
                participants = prenegotiation.get("participants", [])
                num_agents = len(participants)
                
                full_acceptance_info = {
                    "has_full_acceptance": False,
                    "rounds_with_acceptance": [],
                    "total_full_acceptances": 0,
                    "agreement_reached": False,
                    "avg_utility_change": 0.0,
                    "num_agents": num_agents
                }
                
                # Check rounds for full_acceptance
                rounds = data.get("rounds", [])
                for round_data in rounds:
                    if round_data.get("full_acceptance", False):
                        full_acceptance_info["has_full_acceptance"] = True
                        round_num = round_data.get("round_number", 0)
                        full_acceptance_info["rounds_with_acceptance"].append(round_num)
                
                # Check final_state
                final_state = data.get("final_state", {})
                full_acceptance_info["total_full_acceptances"] = final_state.get("full_acceptances", 0)
                full_acceptance_info["agreement_reached"] = final_state.get("agreement_reached", False)
                full_acceptance_info["avg_utility_change"] = final_state.get("avg_utility_change", 0.0)
                
                # If there are any full acceptances, add to list
                # Exclude cases where avg_utility_change is 0 (no actual cost change)
                if (full_acceptance_info["has_full_acceptance"] or 
                    full_acceptance_info["total_full_acceptances"] > 0):
                    # Filter out cases with avg_utility_change == 0
                    if full_acceptance_info["avg_utility_change"] != 0.0:
                        cases_with_acceptance.append((
                            case_dir.name,
                            session_id,
                            full_acceptance_info
                        ))
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file {json_file}: {e}")
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    return cases_with_acceptance


def main():
    """Main function to run the script"""
    cases = find_full_acceptance_cases()
    
    print(f"\n{'='*80}")
    print(f"FULL ACCEPTANCE CASES FOUND: {len(cases)}")
    print(f"{'='*80}\n")
    
    if cases:
        for case_folder, session_id, info in cases:
            print(f"Case Folder: {case_folder}")
            print(f"  Session ID: {session_id}")
            print(f"  Number of Agents (Companies): {info['num_agents']}")
            print(f"  Agreement Reached: {info['agreement_reached']}")
            print(f"  Total Full Acceptances: {info['total_full_acceptances']}")
            print(f"  Avg Utility Change: {info['avg_utility_change']}")
            if info['rounds_with_acceptance']:
                print(f"  Rounds with Full Acceptance: {info['rounds_with_acceptance']}")
            print()
    else:
        print("No cases with full_acceptance found.")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(cases)} case(s) with full_acceptance found")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

