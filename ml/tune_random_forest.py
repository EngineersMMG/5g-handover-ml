import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

data = pd.read_csv("data/processed/feature-dataset.csv")

train_scenarios = pd.read_csv("data/splits/train_scenarios.csv")["scenario_id"]

train_data = data[data["scenario_id"].isin(train_scenarios)].copy()

target = "handover_next_1s"

metadata_columns = [
    "scenario_id",
    "run_id",
    "repeat_id",
    "time_s",
    "handover_event",
    target
]

X_train = train_data.drop(columns=metadata_columns)
y_train = train_data[target]

groups = train_data["scenario_id"]

cv = GroupKFold(n_splits=3)

print("Training rows:", len(X_train))
print("Features:", X_train.shape[1])
print("Unique scenarios:", groups.nunique())
print("Cross-validation folds:", cv.n_splits)

assert X_train.shape[1] == 30

configurations = [
    {"max_depth": 20, "min_sample_leaf": 1}, 
    {"max_depth": 12, "min_sample_leaf": 5}, 
]

results = []

for config in configurations:

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train, groups),
        start=1
    ):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=config["max_depth"],
            min_sample_leaf=config["min_sample_leaf"],
            class_weight="balanced_subsample",
            n_jobs=2, 
            random_state=42
        )

        model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx]
        )

        probabilities = model.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]