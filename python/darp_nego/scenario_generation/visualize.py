import json
import matplotlib.pyplot as plt
import os

def plot_case_locations(case_name, case_data):
    coords = case_data["coordinates"]
    companies = case_data["companies"]

    labels = {}
    colors = {}
    shapes = {}

    for cname, cdata in companies.items():
        company_idx = int(cname.split("_")[1])
        for cl in cdata["clients"]:
            labels[str(cl["start_location"])] = f"C{cl['client_id']}_PU"
            colors[str(cl["start_location"])] = f"C{company_idx}"
            shapes[str(cl["start_location"])] = "o"

    fig, ax = plt.subplots(figsize=(10, 8))
    for lid, (x, y) in coords.items():
        label = labels.get(str(lid), f"Loc {lid}")
        color = colors.get(str(lid), "gray")
        marker = shapes.get(str(lid), "o")
        ax.scatter(x, y, color=color, marker=marker, s=100)
        ax.text(x + 0.3, y + 0.3, label, fontsize=8)

    ax.set_title(f"2D Location Map for {case_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    plt.tight_layout()
    return fig

def main():
    file_path = "python/darp_nego/scenario_generation/data/2025_11_06_21_52/company_cases.json"  # <-- change this if needed
    output_folder = "case_visuals/2025_11_06_21_52"
    os.makedirs(output_folder, exist_ok=True)

    with open(file_path, "r") as f:
        all_cases = json.load(f)

    for case_name, case_data in all_cases.items():
        fig = plot_case_locations(case_name, case_data)
        fig.savefig(os.path.join(output_folder, f"{case_name}.png"))
        plt.close(fig)

    print(f"Saved {len(all_cases)} visualizations in '{output_folder}'")

if __name__ == "__main__":
    main()
