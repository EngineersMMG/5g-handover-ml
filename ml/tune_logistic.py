import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv("data/processed/feature-dataset.csv")

train_scenarios = pd.read_csv("data/splits/train_scenarios.csv")["scenario_id"]

validation_scenarios = pd.read_csv("data/splits/validation_scenarios.csv")["scenario_id"]

train_data = data[data["scenario_id"].isin(train_scenarios)].copy()

validation_data = data[data["scenario_id"].isin(validation_scenarios)].copy()

target = "handover_next_1s"

metadata_columns = ["scenario_id", "run_id", "repeat_id", "time_s", "handover_event", target]

X_train = train_data.drop(columns=metadata_columns)

y_train = train_data[target]

groups = train_data["scenario_id"]


X_validation = validation_data.drop(columns=metadata_columns)

y_validation = validation_data[target]

print("Training rows:", len(X_train))
print("Training scenarios:", groups.nunique())
print("Validation rows:", len(X_validation))
print("Number of features:", X_train.shape[1])

# ============================================================
# 2. TUNE C USING GROUPED CROSS-VALIDATION
# ============================================================

group_kfold = GroupKFold(n_splits=3)

c_values = [0.01, 0.1, 1.0, 10.0]

cv_results = []

for c_value in c_values:

    fold_scores = []

    print(
        "\nTesting C =", c_value)

    for fold_number, (train_index, val_index) in enumerate(
        group_kfold.split(X_train, y_train, groups=groups),
        start=1
    ):

        X_fold_train = X_train.iloc[train_index]

        X_fold_val = X_train.iloc[val_index]

        y_fold_train = y_train.iloc[train_index]

        y_fold_val = y_train.iloc[val_index]

        # Fit scaler ONLY on fold-training data
        scaler = StandardScaler()

        X_fold_train_scaled = (scaler.fit_transform(X_fold_train))

        X_fold_val_scaled = (scaler.transform(X_fold_val))

        model = LogisticRegression(
            C=c_value,
            max_iter=1000,
            class_weight=None,
            random_state=42,
        )

        model.fit(X_fold_train_scaled, y_fold_train)

        y_prob = model.predict_proba(X_fold_val_scaled)[:, 1]

        fold_ap = average_precision_score(y_fold_val, y_prob)

        fold_scores.append(fold_ap)

        print(
            "Fold",
            fold_number,
            "AP:",
            fold_ap
        )

    mean_ap = np.mean(fold_scores)

    cv_results.append({
        "C": c_value,
        "mean_ap": mean_ap,
    })

    print("Mean AP:", mean_ap)


cv_results_df = pd.DataFrame(cv_results)

print("\nCross-validation results:")

print(cv_results_df)

best_row = cv_results_df.loc[cv_results_df["mean_ap"].idxmax()]

best_c = best_row["C"]

print("\nBest C:", best_c)

# ============================================================
# 3. TRAIN FINAL LOGISTIC MODEL
# ============================================================

final_scaler = StandardScaler()

X_train_scaled = (final_scaler.fit_transform(X_train))

X_validation_scaled = (final_scaler.transform(X_validation))

final_model = LogisticRegression(
    C=best_c,
    max_iter=1000,
    class_weight=None,
    random_state=42,
)

final_model.fit(X_train_scaled, y_train) 


y_prob_validation = (final_model.predict_proba(X_validation_scaled)[:, 1])

# ============================================================
# 4. VALIDATION AVERAGE PRECISION
# ============================================================

validation_ap = average_precision_score(
    y_validation,
    y_prob_validation
)

print("\nValidation Average Precision:", validation_ap)

# ============================================================
# 5. CHOOSE THRESHOLD FOR >= 92.5% RECALL
# ============================================================

precisions, recalls, thresholds = (precision_recall_curve(
        y_validation,
        y_prob_validation
    )
)

target_recall = 0.925


valid_indices = np.where(
    recalls[:-1] >= target_recall
)[0]


best_index = valid_indices[
    np.argmax(
        precisions[:-1][
            valid_indices
        ]
    )
]


operating_threshold = thresholds[best_index]

# ============================================================
# 6. EVALUATE 92.5% RECALL OPERATING POINT
# ============================================================

y_pred_operating = (
    y_prob_validation >= operating_threshold
).astype(int)


tn, fp, fn, tp = confusion_matrix(
    y_validation,
    y_pred_operating
).ravel()


precision = precision_score(
    y_validation,
    y_pred_operating
)

recall = recall_score(
    y_validation,
    y_pred_operating
)

f1 = f1_score(
    y_validation,
    y_pred_operating
)

accuracy = accuracy_score(
    y_validation,
    y_pred_operating
)


print("\n===== Logistic Regression: 92.5% Recall Operating Point =====")

print("Threshold:", operating_threshold)

print("Precision:", precision)

print("Recall:", recall)

print("F1:", f1)

print("Accuracy:", accuracy)

print("True positives:", tp)

print("False alarms:", fp)

print("Missed handovers:", fn)

print("True negatives:", tn)