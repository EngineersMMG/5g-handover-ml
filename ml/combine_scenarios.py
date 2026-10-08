import pandas as pd
from pathlib import Path
import re

input_dir = Path("data/raw/scenarios")
output_file = Path("data/processed/combined-scenarios.csv")

csv_files = list(input_dir.glob("*.csv"))

def scenario_sort_key(path):
    match = re.search(r"speed(\d+)_start(\d+)_dist(\d+)", path.stem)

    speed = int(match.group(1))
    start_x = int(match.group(2))
    gnb_distance = int(match.group(3))

    return speed, start_x, gnb_distance

csv_files = sorted(csv_files, key=scenario_sort_key)

all_runs = []
seen_run_ids = set()

for file_path in csv_files:

    match = re.search(
        r"speed(\d+)_start(\d+)_dist(\d+)",
        file_path.stem
    )

    speed = int(match.group(1))
    start_x = int(match.group(2))
    gnb_distance = int(match.group(3))

    data = pd.read_csv(file_path)

    if "run_id" not in data.columns:
        raise ValueError(
            f"{file_path.name} does not contain run_id"
        )
    
    run_ids = data["run_id"].unique()

    if len(run_ids) != 1:
        raise ValueError(
            f"{file_path.name} contains more than one run_id"
        )
    
    run_id = int(run_ids[0])

    if run_id in seen_run_ids:
        raise ValueError(
            f"Duplicate run_id found: {run_id}"
        )
    
    seen_run_ids.add(run_id)

    data["scenario_speed"] = speed
    data["scenario_start_x"] = start_x
    data["scenario_gnb_distance"] = gnb_distance

    all_runs.append(data)

combined_data = pd.concat(
    all_runs,
    ignore_index=True
)

output_file.parent.mkdir(
    parents = True,
    exist_ok = True
)

combined_data.to_csv(
    output_file,
    index=False
)

print("Number of simulation run:", len(csv_files))
print("Unique run IDs:", combined_data["run_id"].nunique())
print("Total rows:", len(combined_data))

print("\nRow per run:")
print(combined_data["run_id"].value_counts().sort_index())

print("\nCombined dataset saved to:")
print(output_file)