import pandas as pd
import numpy as np

input_path = "data/processed/combined-scenarios.csv"
output_path = "data/processed/ml-dataset.csv"

data = pd.read_csv(input_path)

print("Original shape:", data.shape)

# Keep rows after application traffic has started.
# Traffic starts at 1.0 s, so the first useful QoS sample is after that
data = data[data["time_s"] > 1.0].copy()

# Remove rows where radio measurements are not initialized
data = data[
    (data["serving_rsrp"] != 0) &
    (data["neighbor_rsrp"] != 0)
].copy()

# Make sure rows are correctly ordered within each simulation
data = data.sort_values(
    ["run_id", "time_s"]
).reset_index(drop=True)

#--------------------------------------------------
# Create target: handover occurs within next 1 second
# Logging interval = 0.2-s 
# 5 future rows = 1 second
#--------------------------------------------------

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

#-----------------------------------------
# Basic quality checks
#-----------------------------------------

data.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nMissing values:")
print(data.isna().sum()[data.isna().sum() > 0])

print("\nFinal rows:", len(data))
print("Final columns:", len(data.columns))

print("\nRuns:", data["run_id"].nunique())
print("Scenarios:", data["scenario_id"].nunique())
print("Repeats:", sorted(data["repeat_id"].unique()))

print("\nTarget distribution")
print(data["handover_next_1s"].value_counts())

print("\nTarget percentages:")
print(
    data["handover_next_1s"].value_counts(normalize=True).mul(100)
)

data.to_csv(output_path, index=False)

print("\nDataset saved to: ")
print(output_path)