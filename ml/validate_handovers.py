import pandas as pd
from pathlib import Path

input_dir = Path("data/raw/scenarios")

csv_files = list(input_dir.glob("*.csv"))

results = []

for file_path in csv_files:
    data = pd.read_csv(file_path)

    run_id = int(data["run_id"].iloc[0])
    handover_count = int(data["handover_event"].sum())

    results.append({
        "run_id": run_id,
        "file": file_path.name,
        "handover_count": handover_count
    })

summary = pd.DataFrame(results)

print("Total runs:", len(summary))

print("\nTotal handovers:")
print(summary["handover_count"].sum())

print("\nAverage handovers per run:")
print(summary["handover_count"].mean())

print("\nRuns with exactly one handover:")
print((summary["handover_count"] == 1).sum())

print("\nRuns with multiple handovers:")
print((summary["handover_count"] > 1).sum())

print("\nMaximum handovers in one run:")
print(summary["handover_count"].max())