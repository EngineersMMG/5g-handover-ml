import pandas as pd
from pathlib import Path
import re

input_dir = Path("data/raw/scenarios")
output_file = Path("data/processed/combined-scenarios.csv")

csv_files = list(input_dir.glob("*.csv"))

def scenario_sort_key(path):
    match = re.search(r"speed(\d+)_start(\d+)", path.stem)
    return int(match.group(1)), int(match.group(2))

csv_files = sorted(csv_files, key=scenario_sort_key)

all_runs = []

for run_id, file_path in enumerate(csv_files, start=1):

    match = re.search(
        r"speed(\d+)_start(\d+)",
        file_path.stem
    )

    speed = int(match.group(1))
    start_x = int(match.group(2))

    data = pd.read_csv(file_path)

    data["run_id"] = run_id
    data["scenario_speed"] = speed
    data["scenario_start_x"] = start_x

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
print("Total rows:", len(combined_data))

print("\nRow per run:")
print(combined_data["run_id"].value_counts().sort_index())

print("\nCombined dataset saved to:")
print(output_file)