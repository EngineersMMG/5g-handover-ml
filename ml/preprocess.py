import pandas as pd

input_path = "data/processed/combined-scenarios.csv"
output_path = "data/processed/ml-dataset.csv"

data = pd.read_csv(input_path)

data["handover_next_1s"] = (data.groupby("run_id")["handover_event"].shift(-1))

data = data.dropna(subset=["handover_next_1s"])

data["handover_next_1s"] = data["handover_next_1s"].astype(int)

data.to_csv(output_path, index=False)

print("Number of rows:", len(data))
print("Number of runs:", data["run_id"].nunique())

print("\n Target distribution")
print(data["handover_next_1s"].value_counts())

print("\n Dataset saved to: ")
print(output_path)