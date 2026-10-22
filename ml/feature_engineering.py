import pandas as pd

input_path = "data/processed/ml-dataset.csv"
output_path = "data/processed/feature-dataset.csv"

data = pd.read_csv(input_path)

print("Original shape", data.shape)

# Always keep time order inside each simulation run
data = data.sort_values(["run_id", "time_s"]).reset_index(drop=True)

#----------------------------------------------
# 1. Domain-based features
#----------------------------------------------

# Neighbor quality relative to serving quality
data["rsrq_difference"] = (data["neighbor_rsrq"] - data["serving_rsrq"])

# serving_cell is categorical (cell 1 or cell 2)
# Convert it into a binary indicator
data["serving_cell_is_2"] = (data["serving_cell"] == 2).astype(int)

#----------------------------------------------
# 2. Short-term changes
#----------------------------------------------

change_features = [
    "serving_rsrp",
    "neighbor_rsrp",
    "rsrp_difference",
    "serving_rsrq",
    "neighbor_rsrq",
    "serving_sinr_db",
    "throughput_mbps",
    "average_delay_ms",
    "packet_loss_rate",
]

for feature in change_features:
    data[f"{feature}_change_0_2s"] = (
        data.groupby("run_id")[feature].diff()
    )

#-------------------------------------------------
# 3. Historical 1-second rolling averages
#-------------------------------------------------

# Logging interval = 0.2 s
#
# For the prediction target:
# next 5 feature samples reach t + 1.0 s
#
# For a historical window including the current sample,
# we use 6 samples:
# t, t-0.2, t-0.4, t-0.6, t-0.8, t-1.0

rolling_window = 6

rolling_features = [
    "serving_rsrp",
    "neighbor_rsrp",
    "rsrp_difference",
    "serving_rsrq",
    "neighbor_rsrq",
    "serving_sinr_db",
    "throughput_mbps",
    "average_delay_ms",
    "packet_loss_rate",
]

for feature in rolling_features:
    data[f"{feature}_mean_1s"] = (
        data.groupby("run_id")[feature]
        .transform(
            lambda x: x.rolling(
                window = rolling_window,
                min_periods = rolling_window
            ).mean()
        )
    )

#-------------------------------------------------
# 4. Select metadata, target, and ML feature
#-------------------------------------------------

metadata_columns = [
    "scenario_id",
    "run_id", 
    "repeat_id",
    "time_s",
    "handover_event"
]

base_features = [
    "speed_mps",
    "serving_cell_is_2",
    "serving_rsrp",
    "neighbor_rsrp",
    "rsrp_difference",
    "serving_rsrq",
    "neighbor_rsrq",
    "rsrq_difference",
    "serving_sinr_db",
    "throughput_mbps",
    "average_delay_ms",
    "packet_loss_rate",
]

engineered_features = (
    [f"{feature}_change_0_2s" for feature in change_features]
    +
    [f"{feature}_mean_1s" for feature in rolling_features]
)

target_column = ["handover_next_1s"]

final_columns = (metadata_columns + base_features + engineered_features + target_column)

data = data[final_columns].copy()

# The first few samples of each run do not yet have enough historical observations 
# for a full 1-second window.
data = data.dropna().reset_index(drop=True)

#-------------------------------------------------
# 5. Final checks
#-------------------------------------------------

print("\nFeature-engineered shape:", data.shape)
print("Runs:", data["run_id"].nunique())
print("Scenarios:", data["scenario_id"].nunique())

print("\nMissing values:", data.isna().sum().sum())

print("\nTarget distribution:", data["handover_next_1s"].value_counts())

print("\nNumber of ML input features:")
print(len(base_features) + len(engineered_features))

print("\nML features:")
for feature in base_features + engineered_features:
    print(feature)

data.to_csv(output_path, index=False)

print("\nFeature dataset saved to:", output_path)