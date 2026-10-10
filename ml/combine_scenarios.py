import pandas as pd
from pathlib import Path
import re

project_dir = Path.home() / "5g-handover-work" / "5g-handover-ml"

input_dir = project_dir / "data" / "raw" / "scenarios"  
output_path = project_dir / "data" / "processed" / "combined-scenarios.csv"

output_path.parent.mkdir(parents=True, exist_ok=True)

pattern = re.compile(
    r"speed(\d+)_"
    r"start(\d+)_"
    r"dist(\d+)_"
    r"power([+-]\d+)_"
    r"rep(\d+)"
)

def scenario_sort_key(file_path):
    match = pattern.search(file_path.stem)

    if match is None:
        return (999, 999, 999, 999, 999)

    speed = int(match.group(1))
    start_x = int(match.group(2))
    gnb_distance = int(match.group(3))
    power_offset = int(match.group(4))
    repeat_id = int(match.group(5))

    return speed, start_x, gnb_distance, power_offset, repeat_id

csv_files = sorted(input_dir.glob("*.csv"), key=scenario_sort_key)

all_runs = []
seen_run_ids = set()

scenario_map = {}
next_scenario_id = 1

for file_path in csv_files:

    match = pattern.search(file_path.stem)

    if match is None: 
        print(f"Skipping unrecognized file: {file_path.name}")
        continue

    speed = int(match.group(1))
    start_x = int(match.group(2))
    gnb_distance = int(match.group(3))
    power_offset = int(match.group(4))
    repeat_id = int(match.group(5))

    scenario_key = (
        speed,
        start_x,
        gnb_distance,
        power_offset,
    )

    if scenario_key not in scenario_map:
        scenario_map[scenario_key] = next_scenario_id
        next_scenario_id += 1

    scenario_id = scenario_map[scenario_key]

    gnb1_power = 30
    gnb2_power = 30 + power_offset

    data = pd.read_csv(file_path)

    if "run_id" not in data.columns:
        raise ValueError(
            f"{file_path.name} does not contain run_id"
        )
    
    run_ids = data["run_id"].unique()

    if len(run_ids) != 1:
        raise ValueError(
            f"{file_path.name} contains more than one run_id: {run_ids}"
        )
    
    run_id = int(run_ids[0])

    if run_id in seen_run_ids:
        raise ValueError(
            f"Duplicate run_id found: {run_id}"
        )
    
    seen_run_ids.add(run_id)

    # Add scenatio/run metadata
    data["scenario_id"] = scenario_id
    data["repeat_id"] = repeat_id

    data["scenario_speed"] = speed
    data["scenario_start_x"] = start_x
    data["scenario_gnb_distance"] = gnb_distance
    data["scenario_power_offset"] = power_offset
    data["scenario_gnb1_tx_power"] = gnb1_power
    data["scenario_gnb2_tx_power"] = gnb2_power

    all_runs.append(data)

combined_data = pd.concat(
    all_runs,
    ignore_index=True
)

output_path.parent.mkdir(
    parents = True,
    exist_ok = True
)

combined_data.to_csv(
    output_path,
    index=False
)

print("\nCombination finished.")
print("Number of simulation run:", len(csv_files))
print("Unique run IDs:", combined_data["run_id"].nunique())
print("Scenario configurations:", combined_data["scenario_id"].nunique())
print("Total rows:", len(combined_data))
print("Total columns", len(combined_data.columns))
print("\nCombined dataset saved to:")
print(output_path)