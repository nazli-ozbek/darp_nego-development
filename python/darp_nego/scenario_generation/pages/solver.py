import contextlib
import io
import os
import json
from datetime import datetime
import streamlit as st
from darp_nego.darp.basic_darp import BasicDARPVehicle, BasicDARPClient, BasicDARPProblem


def get_latest_generated_data_folder(data_directory: str) -> str:
    """Find the latest timestamped folder in the 'data' directory."""
    timestamped_folders = [
        folder for folder in os.listdir(data_directory) 
        if os.path.isdir(os.path.join(data_directory, folder))
    ]
    
    if not timestamped_folders:
        raise FileNotFoundError("No timestamped folders found in the 'data' directory.")

    latest_folder = max(timestamped_folders, key=lambda x: datetime.strptime(x, '%Y_%m_%d_%H_%M'))

    return os.path.join(data_directory, latest_folder)

def load_generated_data(data_directory: str):
    """Load the company_cases.json file from the latest timestamped folder."""
    latest_folder = get_latest_generated_data_folder(data_directory)
    json_file_path = os.path.join(latest_folder, "company_cases.json")

    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"No company_cases.json found in {latest_folder}")

    with open(json_file_path, "r") as f:
        data = json.load(f)

    return data, latest_folder

def run_solver(data, latest_folder):
    """Run the solver and stream output."""
    output_box = st.empty()
    status_box = st.empty()
    log_lines = []
    result_summary = {} 


    for case_key, case_data in data.items():
        companies = case_data.get("companies", {})
        road_network = case_data.get("time_matrix", [])
        result_summary[case_key] = {}

        for company_key, company_data in companies.items():
            clients_data = company_data.get("clients", {})
            vehicles_data = company_data.get("vehicles", {})
            result_summary[case_key][company_key] = {}

            vehicles = [BasicDARPVehicle.from_json(vehicle) for vehicle in vehicles_data]
            clients = [BasicDARPClient.from_json(client) for client in clients_data]
            problem = BasicDARPProblem(len(vehicles), vehicles=vehicles, clients=clients)

            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                status_box.info(f"⏳  Scenario {case_key} - {company_key} is solving...")
                solution_cost = problem.solve_problem(road_network=road_network, config=None)

                if solution_cost is None:
                    log_lines.append(f"⚠️  Scenario {case_key} - {company_key} is infeasible!")
                else:
                    log_lines.append(f"✅ Scenario {case_key} - {company_key}: Solution Cost = {solution_cost}")

                printed_output = buf.getvalue()                
                log_lines.append("```text\n" + printed_output + "\n```")
                

            output_box.markdown("### Output Log\n" + "\n\n".join(log_lines))
            status_box.empty()

            result_summary[case_key][company_key] = {
                "solution_cost": solution_cost,
                "log": log_lines
            }
            
    solver_output_path = os.path.join(latest_folder, "solver_output.json")
    with open(solver_output_path, "w") as f:
        json.dump(result_summary, f, indent=4)
    st.success(f"Solver results saved to `{solver_output_path}`")


# --- Streamlit UI ---
st.title("🧠 DARP Problem Solver")

if st.button("Run Solver"):
    data_dir = "data/"
    try:
        data, latest_folder = load_generated_data(data_dir)
        run_solver(data, latest_folder)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
