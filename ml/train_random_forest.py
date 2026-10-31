
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv("data/processed/feature-dataset.csv")

train_scenarios = pd.read_csv(
    "data/splits/train_scenarios.csv"
)["scenario_id"]

validation_scenarios = pd.read_csv(
    "data/splits/validation_scenarios.csv"
)["scenario_id"]

train_data = data[
    data["scenario_id"].isin(train_scenarios)
].copy()

validation_data = data[
    data["scenario_id"].isin(validation_scenarios)
].copy()

print("\n===== DATASET INFORMATION =====")
print("Train rows:", len(train_data))
print("Validation rows:", len(validation_data))
print("Training scenarios:", train_data["scenario_id"].nunique())
print("Validation scenarios:", validation_data["scenario_id"].nunique())


# ============================================================
# 2. PREPARE FEATURES AND TARGET
# ============================================================

target = "handover_next_1s"

metadata_columns = [
    "scenario_id",
    "run_id",
    "repeat_id",
    "time_s",
    "handover_event",
    target,
]

X_train = train_data.drop(columns=metadata_columns)
y_train = train_data[target]

X_validation = validation_data.drop(columns=metadata_columns)
y_validation = validation_data[target]

print("\n===== RANDOM FOREST FEATURE CHECK =====")
print("X train shape:", X_train.shape)
print("X validation shape:", X_validation.shape)
print("Training features:", X_train.shape[1])
print("Validation features:", X_validation.shape[1])

assert X_train.shape[1] == 30
assert X_train.columns.equals(X_validation.columns)

assert set(train_data["scenario_id"]).isdisjoint(
    set(validation_data["scenario_id"])
)

print("Feature and scenario checks passed!")


# ============================================================
# 3. TRAIN TUNED RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=16,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)

print("\n===== RANDOM FOREST TRAINING =====")
print(model)

model.fit(X_train, y_train)

print("Training complete!")


# ============================================================
# 4. PREDICT ON VALIDATION DATA
# ============================================================

y_pred_default = model.predict(X_validation)

y_prob = model.predict_proba(X_validation)[:, 1]

validation_data["handover_probability"] = y_prob

print("\n===== DEFAULT MODEL PERFORMANCE =====")

print(
    "Confusion matrix:\n",
    confusion_matrix(y_validation, y_pred_default),
)

print(
    "Default accuracy:",
    accuracy_score(y_validation, y_pred_default),
)

print(
    "Default precision:",
    precision_score(y_validation, y_pred_default),
)

print(
    "Default recall:",
    recall_score(y_validation, y_pred_default),
)

print(
    "Default F1:",
    f1_score(y_validation, y_pred_default),
)


# ============================================================
# 5. AVERAGE PRECISION
# ============================================================

average_precision = average_precision_score(
    y_validation,
    y_prob,
)

print("\n===== AVERAGE PRECISION =====")
print(f"Average Precision: {average_precision:.6f}")


# ============================================================
# 6. FIND BEST ROW-LEVEL F1 THRESHOLD
# ============================================================

precisions, recalls, thresholds = precision_recall_curve(
    y_validation,
    y_prob,
)

f1_scores = (
    2 * precisions[:-1] * recalls[:-1]
    /
    (precisions[:-1] + recalls[:-1] + 1e-12)
)

best_index = np.argmax(f1_scores)

best_threshold = float(thresholds[best_index])

y_pred_tuned = (y_prob >= best_threshold).astype(int)

print("\n===== BEST ROW-LEVEL F1 THRESHOLD =====")

print(f"Best threshold: {best_threshold:.6f}")
print(f"Best F1: {f1_scores[best_index]:.6f}")

print(
    f"Precision: {precision_score(y_validation, y_pred_tuned):.6f}"
)

print(
    f"Recall: {recall_score(y_validation, y_pred_tuned):.6f}"
)

print(
    f"F1: {f1_score(y_validation, y_pred_tuned):.6f}"
)

print(
    f"Accuracy: {accuracy_score(y_validation, y_pred_tuned):.6f}"
)

print(
    "Confusion matrix:\n",
    confusion_matrix(y_validation, y_pred_tuned),
)


# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False,
)

print("\n===== TOP 10 IMPORTANT FEATURES =====")
print(feature_importance.head(10).to_string(index=False))


# ============================================================
# 8. EVENT-LEVEL AND ALARM-EPISODE EVALUATION
# ============================================================

def evaluate_threshold(validation_data, threshold):

    # Apply threshold without retraining the model
    validation_data["prediction"] = (
        validation_data["handover_probability"] >= threshold
    ).astype(int)

    detected = 0
    useful = 0
    false = 0

    total_events = int(
        validation_data["handover_event"].sum()
    )

    # Evaluate each simulation run separately
    for run_id, run_data in validation_data.groupby("run_id"):

        run_data = run_data.sort_values("time_s")

        handover_times = run_data.loc[
            run_data["handover_event"] == 1,
            "time_s",
        ].to_numpy()

        # ----------------------------------------------------
        # A. EVENT DETECTION
        # ----------------------------------------------------

        for ho_time in handover_times:

            prior_window = run_data[
                (run_data["time_s"] >= ho_time - 1.0)
                &
                (run_data["time_s"] < ho_time)
            ]

            if (prior_window["prediction"] == 1).any():
                detected += 1

        # ----------------------------------------------------
        # B. ALARM EPISODES
        # ----------------------------------------------------

        alarms = run_data["prediction"] == 1

        episode_ids = alarms.ne(
            alarms.shift(fill_value=False)
        ).cumsum()

        # Process episodes once per run, not per handover
        for _, episode in run_data.loc[alarms].groupby(
            episode_ids[alarms]
        ):

            start_time = episode["time_s"].iloc[0]
            end_time = episode["time_s"].iloc[-1]

            is_useful = np.any(
                (start_time < handover_times)
                &
                (handover_times <= end_time + 1.0)
            )

            if is_useful:
                useful += 1
            else:
                false += 1

    # --------------------------------------------------------
    # C. METRICS
    # --------------------------------------------------------

    missed = total_events - detected
    total_episodes = useful + false

    event_recall = (
        detected / total_events
        if total_events > 0 else 0.0
    )

    episode_precision = (
        useful / total_episodes
        if total_episodes > 0 else 0.0
    )

    positive_rows = int(
        validation_data["prediction"].sum()
    )

    # --------------------------------------------------------
    # D. SAFETY CHECKS
    # --------------------------------------------------------

    assert 0 <= detected <= total_events

    assert useful + false <= positive_rows, (
        "Alarm episodes exceed positive prediction rows."
    )

    assert detected + missed == total_events

    return {
        "threshold": threshold,
        "event_recall": event_recall,
        "detected_handovers": detected,
        "missed_handovers": missed,
        "useful_episodes": useful,
        "false_episodes": false,
        "total_episodes": total_episodes,
        "episode_precision": episode_precision,
    }


# ============================================================
# 9. ORIGINAL BEST-F1 THRESHOLD: EVENT EVALUATION
# ============================================================

baseline_event_result = evaluate_threshold(
    validation_data,
    best_threshold,
)

print("\n===== RANDOM FOREST EVENT DETECTION =====")

print(
    "Detected handovers:",
    baseline_event_result["detected_handovers"],
)

print(
    "Missed handovers:",
    baseline_event_result["missed_handovers"],
)

print(
    f"Event recall: "
    f"{baseline_event_result['event_recall']:.4%}"
)

print("\n===== RANDOM FOREST ALARM ANALYSIS =====")

print(
    "Useful episodes:",
    baseline_event_result["useful_episodes"],
)

print(
    "False alarm episodes:",
    baseline_event_result["false_episodes"],
)

print(
    "Total episodes:",
    baseline_event_result["total_episodes"],
)

print(
    f"Episode precision: "
    f"{baseline_event_result['episode_precision']:.4%}"
)


# ============================================================
# 10. THRESHOLD COMPARISON
# ============================================================

candidate_thresholds = [
    0.680,
    0.682,
    0.684,
    0.686,
    0.688,
    0.690,
    0.695,
    0.700
]

threshold_results = []

print("\n===== EVALUATING CANDIDATE THRESHOLDS =====")

for threshold in candidate_thresholds:

    result = evaluate_threshold(
        validation_data,
        threshold,
    )

    threshold_results.append(result)

    print(
        f"Threshold: {threshold:.6f} | "
        f"Event recall: {result['event_recall']:.4%} | "
        f"False episodes: {result['false_episodes']} | "
        f"Episode precision: {result['episode_precision']:.4%}"
    )


# ============================================================
# 11. PRINT COMPLETE THRESHOLD RESULTS
# ============================================================

threshold_results_df = pd.DataFrame(threshold_results)

assert len(threshold_results_df) == len(candidate_thresholds)

print("\n===== RANDOM FOREST THRESHOLD COMPARISON =====")

print(
    threshold_results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)


# ============================================================
# 12. SELECT OPERATING THRESHOLD
# ============================================================

# Validation objective:
# Event recall >= 99.5%, minimizing false alarm episodes.

eligible_thresholds = threshold_results_df[
    threshold_results_df["event_recall"] >= 0.995
].copy()

print("\n===== RANDOM FOREST THRESHOLD SELECTION =====")

if eligible_thresholds.empty:

    print(
        "No candidate threshold satisfies "
        "the 99.5% event-recall requirement."
    )

    selected_threshold = None

else:

    eligible_thresholds = eligible_thresholds.sort_values(
        by=["false_episodes", "episode_precision"],
        ascending=[True, False],
    )

    selected = eligible_thresholds.iloc[0]

    selected_threshold = float(selected["threshold"])

    print(f"Selected threshold: {selected_threshold:.6f}")

    print(
        f"Event recall: {selected['event_recall']:.4%}"
    )

    print(
        f"Episode precision: {selected['episode_precision']:.4%}"
    )

    print(
        f"Detected handovers: "
        f"{int(selected['detected_handovers'])}"
    )

    print(
        f"Missed handovers: "
        f"{int(selected['missed_handovers'])}"
    )

    print(
        f"False alarm episodes: "
        f"{int(selected['false_episodes'])}"
    )


# ============================================================
# 13. FINAL RANDOM FOREST SUMMARY
# ============================================================

print("\n========== RANDOM FOREST SUMMARY ==========")

print("Trees:", model.n_estimators)
print("Maximum depth:", model.max_depth)
print("Minimum samples per leaf:", model.min_samples_leaf)

print(f"Average Precision: {average_precision:.6f}")

print(f"Best row-F1 threshold: {best_threshold:.6f}")

print(
    f"Row precision at best F1: "
    f"{precision_score(y_validation, y_pred_tuned):.4%}"
)

print(
    f"Row recall at best F1: "
    f"{recall_score(y_validation, y_pred_tuned):.4%}"
)

print(
    f"Row F1: "
    f"{f1_score(y_validation, y_pred_tuned):.6f}"
)

if selected_threshold is not None:

    print(
        f"Selected event threshold: "
        f"{selected_threshold:.6f}"
    )

else:

    print("Selected event threshold: None")

print("===========================================")
