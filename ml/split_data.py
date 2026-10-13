import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

input_path = "data/processed/feature-dataset.csv"
split_dir = Path("data/splits")

split_dir.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(input_path)

# Get unique physical scenarios
scenario_ids = data["scenario_id"].unique()

print("Total scenarios:", len(scenario_ids))

#-------------------------------------------
# 70% Train, 15% Validation, 15% Train
#-------------------------------------------

train_scenarios, temp_scenarios = train_test_split(
    scenario_ids,
    test_size=0.30,
    random_state=42,
) 

validation_scenarios, test_scenarios = train_test_split(
    temp_scenarios,
    test_size=0.50,
    random_state=42,
)

#---------------------------------------------------
# Create row subsets only for validation/statistics
#---------------------------------------------------

train_data = data[data["scenario_id"].isin(train_scenarios)]
validation_data = data[data["scenario_id"].isin(validation_scenarios)]
test_data = data[data["scenario_id"].isin(test_scenarios)]

#---------------------------------------------------
# Save only scenarios IDs
#---------------------------------------------------

pd.DataFrame(
    {"scenario_id": sorted(train_scenarios)}
).to_csv(
    split_dir / "train_scenarios.csv",
    index=False
)

pd.DataFrame(
       {"scenario_id": sorted(validation_scenarios)}
).to_csv(
    split_dir / "validation_scenarios.csv",
    index=False
)

pd.DataFrame(
       {"scenario_id": sorted(test_scenarios)}
).to_csv(
    split_dir / "test_scenarios.csv",
    index=False
)

#---------------------------------------------------
# Verify no scenario leakage
#---------------------------------------------------

assert set(train_scenarios).isdisjoint(validation_scenarios)
assert set(test_scenarios).isdisjoint(validation_scenarios)
assert set(validation_scenarios).isdisjoint(test_scenarios)

# Verify each scenario's repeat stay together automatically 
print("\n=== Scenario Split ===")
print("Train scenarios:", len(train_scenarios))
print("Validation scenarios:", len(validation_scenarios))
print("Test scenarios:", len(test_scenarios))

print("\n=== Row Counts ===")
print("Train rows:", len(train_data))
print("Validation rows:", len(validation_data))
print("Test rows:", len(test_data))

print("\n=== Runs ===")
print("Train runs:", train_data["run_id"].nunique())
print("Validation runs:", validation_data["run_id"].nunique())
print("Test runs:", test_data["run_id"].nunique())

print("\n=== Positive Class Percentage ===")
print(
    "Train:",
    train_data["handover_next_1s"].mean() * 100
)
print(
    "Validation:",
    validation_data["handover_next_1s"].mean() * 100
)
print(
    "Test:",
    test_data["handover_next_1s"].mean() * 100
)

print("\nScenario leakage: NONE")
print("Split files saved to:", split_dir)