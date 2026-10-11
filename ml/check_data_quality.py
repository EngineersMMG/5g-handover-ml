import pandas as pd
import numpy as np

input_path = "data/processed/ml-dataset.csv"

data = pd.read_csv(input_path)

print("Dataset shape:", data.shape)

#---------------------------------------------
# 1. Missing / invalid values
#---------------------------------------------

print("\n=== Missing Values ===")
print("Total NaN values:", data.isna().sum().sum())

numeric_data = data.select_dtypes(include=[np.number])

print("Infinite values:", np.isinf(numeric_data.to_numpy()).sum())

#---------------------------------------------
# 2. Duplicate checks
#---------------------------------------------

print("\n=== Duplicate Checks ===")

exact_duplicates = data.duplicated().sum()

duplicate_timestamps = data.duplicated(
    subset=["run_id", "time_s"]
).sum()

print("Exact duplicate rows:", exact_duplicates)
print("Duplicate run_id + time_s rows:", duplicate_timestamps)

#---------------------------------------------
# 3. Basic validity checks
#---------------------------------------------

print("\n=== Validity Checks ===")

print("Negative throughput:", (data["throughput_mbps"] < 0).sum())
print("Negative delay:", (data["average_delay_ms"] < 0).sum())
print("Packet loss < 0:", (data["packet_loss_rate"] < 0).sum())
print("Invalid prediction target:", (~data["handover_next_1s"].isin([0, 1])).sum())
print("Serving cells:", sorted(data["serving_cell"].unique()))
print("Target cells:", sorted(data["target_cell"].unique()))

#---------------------------------------------
# 4. Statistical inspection
#---------------------------------------------

features = [
    "speed_mps",
    "serving_rsrp",
    "neighbor_rsrp",
    "serving_rsrq",
    "neighbor_rsrq",
    "rsrp_difference",
    "serving_sinr_db",
    "throughput_mbps",
    "average_delay_ms",
    "packet_loss_rate"
]

print("\n=== Feature Statistics ===")

print(
    data[features].describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    ).T
)

#---------------------------------------------
# 5. Dataset structure
#---------------------------------------------

print("\n=== Dataset Structure ===")

print("Runs:", data["run_id"].nunique())
print("Scenarios:", data["scenario_id"].nunique())
print("\nRepeats per scenario:", data.groupby("scenario_id")["repeat_id"].nunique().value_counts())
print("\nTarget distribution:", data["handover_next_1s"].value_counts())
print("\nTarget percentages:", data["handover_next_1s"].value_counts(normalize=True).mul(100))
