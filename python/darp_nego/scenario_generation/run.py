import sys
sys.path.append('.')
from pages.outlier_generator import generate_scenarios_with_outliers, save_json
import json
from datetime import datetime

# Generate scenarios
scenarios = generate_scenarios_with_outliers()

# Create timestamp folder
timestamp = '2025_08_08_14_00'
folder_name = f'data/{timestamp}'

# Save each scenario
cases_data = {}
for i, scenario in enumerate(scenarios, 1):
    cases_data[f'case_{i}'] = scenario

file_path_cases = save_json(folder_name, 'company_cases.json', cases_data)
print(f'Generated {len(scenarios)} scenarios with outliers!')
print(f'Cases saved in: {file_path_cases}')

# Show summary
print('\\nScenario Summary:')
for i, (case_name, case_data) in enumerate(cases_data.items()):
    num_companies = len(case_data['companies'])
    total_clients = sum(len(company['clients']) for company in case_data['companies'].values())
    print(f'- {case_name}: {num_companies} companies, {total_clients} total clients')
