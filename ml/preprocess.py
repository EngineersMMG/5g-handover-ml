import pandas as pd

input_path = "data/processed/combined-scenarios.csv"
output_path = "data/processed/ml-dataset.csv"

data = pd.read_csv(input_path)

# Remove rows before valid radio measurements are available
data = data[
    (data["serving_rsrp"] != 0) &
    (data["neighbor_rsrp"] != 0)
].copy()

# 0.2-second logging -> 5 future samples = 1 second
future_steps = 5

future_handover = pd.concat(
    [
        data.groupby("run_id")["handover_event"].shift(-step)
        for step in range(1, future_steps + 1) 
    ],
    axis=1
)

# Keep only rows where a complete next 1-second window exists
valid_rows = future_handover.notna().all(axis=1)

data = data.loc[valid_rows].copy()
future_handover = future_handover.loc[valid_rows]

# 1 if a handover happens anywhere in the next second
data["handover_next_1s"] = (
    future_handover.max(axis=1).astype(int)
)

data.to_csv(output_path, index=False)

print("Number of rows:", len(data))
print("Number of runs:", data["run_id"].nunique())

print("\nTarget distribution")
print(data["handover_next_1s"].value_counts())

print("\nDataset saved to: ")
print(output_path)