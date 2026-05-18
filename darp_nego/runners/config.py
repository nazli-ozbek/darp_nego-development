DEBUG_SINGLE_SCENARIO = False

# Select one by commenting/uncommenting:
RUN_STRATEGY = "heuristic"
# RUN_STRATEGY = "learning"
# RUN_STRATEGY = "heuristic_partial"

# Fair comparison mode (paired multi-seed run):
# FAIR_COMPARISON_MODE = True
FAIR_COMPARISON_MODE = False

# Keep the same seed list on all PCs for paired comparison.
FAIR_COMPARISON_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

# OR-Tools determinism controls for baseline reproducibility.
ORTOOLS_NUM_SEARCH_WORKERS = 1

# Scenario source selection:
# USE_LATEST_SCENARIO = True
USE_LATEST_SCENARIO = False

# If USE_LATEST_SCENARIO is False, set one scenario JSON path below (absolute path recommended).
SCENARIO_JSON_PATH = "/Users/nazliozbek/Desktop/darp-nego/darp_nego-development_new/scenario_generation/data/llm_generated/company_cases_demo_2x3.json"
# SCENARIO_JSON_PATH = "/Users/nazliozbek/Desktop/darp-nego/darp_nego-development_new/scenario_generation/data/llm_generated/company_cases_merged_4_6_8.json"
