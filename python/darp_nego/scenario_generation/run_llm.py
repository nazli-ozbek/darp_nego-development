import argparse
import os
import sys
import random
from datetime import datetime

SCENARIO_DIR = os.path.dirname(__file__)
PROJECT_PY_DIR = os.path.dirname(SCENARIO_DIR)
if PROJECT_PY_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PY_DIR)

from llm_scenario import LLMScenarioConfig, generate_case_with_llm, postprocess_case, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DARP scenarios using Gemini LLM.")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API key (or set GEMINI_API_KEY).")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name.")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of cases to generate.")
    parser.add_argument("--num-companies", type=int, default=3, help="Companies per case.")
    parser.add_argument("--clients-min", type=int, default=20, help="Min clients per company.")
    parser.add_argument("--clients-max", type=int, default=30, help="Max clients per company.")
    parser.add_argument("--vehicles-min", type=int, default=4, help="Min vehicles per company.")
    parser.add_argument("--vehicles-max", type=int, default=5, help="Max vehicles per company.")
    parser.add_argument("--num-hospitals", type=int, default=3, help="Number of hospital nodes.")
    parser.add_argument("--output-dir", default="data", help="Output base directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for regime selection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set GEMINI_API_KEY.")

    config = LLMScenarioConfig(
        num_companies=args.num_companies,
        clients_per_company=(args.clients_min, args.clients_max),
        vehicles_per_company=(args.vehicles_min, args.vehicles_max),
        num_hospitals=args.num_hospitals,
    )

    regimes = ["A", "B", "C", "D", "E", "F"]
    regime_rng = random.Random(args.seed)

    cases_data = {}
    for case_index in range(1, args.num_cases + 1):
        regime = regime_rng.choice(regimes)
        config.diversity_regime = regime
        raw_case = generate_case_with_llm(
            api_key=args.api_key,
            config=config,
            model_name=args.model,
        )
        case_data = postprocess_case(raw_case, config)
        cases_data[f"case_{case_index}"] = case_data

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_name = os.path.join(args.output_dir, timestamp)
    file_path_cases = save_json(folder_name, "company_cases.json", cases_data)
    print(f"Saved: {file_path_cases}")


if __name__ == "__main__":
    main()
