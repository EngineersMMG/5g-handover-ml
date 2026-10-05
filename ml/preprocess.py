import pandas as pd

data = pd.read_csv("data/raw/handover-data.csv")

print("First 5 rows:")
print(data.head())

data["handover_next_1s"] = data["handover_event"].shift(-1)

data = data.dropna()

data["handover_next_1s"] = data["handover_next_1s"].astype(int)

print("\nAround handover:")

print(
    data.loc[
        (data["time_s"] >= 20) & (data["time_s"] <= 24),
        [
            "time_s",
            "serving_rsrp",
            "neighbor_rsrp",
            "rsrp_difference",
            "handover_event",
            "handover_next_1s"
        ]
    ]
)