import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

data = pd.read_csv("data/processed/feature-dataset.csv")
train_scenarios = pd.read_csv("data/splits/train_scenarios.csv")["scenario_id"]
validation_scenarios = pd.read_csv("data/splits/validation_scenarios.csv")["scenario_id"]

train_data = data[data["scenario_id"].isin(train_scenarios)].copy()
validation_data = data[data["scenario_id"].isin(validation_scenarios)].copy()

print("Train rows:", len(train_data))
print("Validation rows:", len(validation_data))

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

X_validation = validation_data.drop(columns=metadata_columns)
y_validation = validation_data[target]

print("X train shape:", X_train.shape)
print("y train shape:", y_train.shape)
print("X validation shape:", X_validation.shape)
print("y validation shape:", y_validation.shape)

print("\n===== RANDOM FOREST FEATURE CHECK ========")
print("Training features:", X_train.shape[1])
print("Validation features:", X_validation.shape[1])

assert X_train.shape[1] == 30
assert X_train.columns.equals(X_validation.columns)

print("Feature check passed!")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=16,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
    verbose=1
)
print(model)

print("Training Random Forest...")
model.fit(X_train, y_train)
print("Training complete.")

y_pred = model.predict(X_validation)
y_prob = model.predict_proba(X_validation)[:, 1]

print(y_pred[:10])
print(y_prob[:10])

cm = confusion_matrix(y_validation, y_pred)
print(cm)

average_precision = average_precision_score(y_validation, y_prob)
print("Average Precision:", average_precision)

precisions, recalls, thresholds = precision_recall_curve(
    y_validation,
    y_prob
)

f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
print("F1 scores:", f1_scores)

best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]

print("Best threshold:", best_threshold)
print("Best F1:", f1_scores[best_index])
print("Precision at best threshold:", precisions[best_index])
print("Recall at best threshold:", recalls[best_index])

y_pred_tuned = (y_prob >= best_threshold).astype(int)

cm_tuned = confusion_matrix(y_validation, y_pred_tuned)
print(cm)
print(cm_tuned)

print(
    "Tuned precision:", 
    precision_score(y_validation, y_pred_tuned)
)
print(
    "Tuned recall:", 
    recall_score(y_validation, y_pred_tuned)
)
print(
    "Tuned F1:", 
    f1_score(y_validation, y_pred_tuned)
)
print(
    "Default accuracy:",
    accuracy_score(y_validation, y_pred)
)
print(
    "Tuned accuracy:",
    accuracy_score(y_validation, y_pred_tuned)
)

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})
feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)
print(feature_importance.head(10))

validation_data["prediction"] = y_pred_tuned

detected_events = 0
missed_events = 0

for run_id, run_data in validation_data.groupby("run_id"):

    handover_times = run_data.loc[
        run_data["handover_event"] == 1,
        "time_s"
    ]

    for handover_time in handover_times:

        prior_window = run_data[
            (run_data["time_s"] >= handover_time - 1.0)
            &
            (run_data["time_s"] < handover_time)
        ]

        if (prior_window["prediction"] == 1).any():
            detected_events += 1

        else:
            missed_events += 1

total_events = missed_events + detected_events

print("\n======= RANDOM FOREST EVENT DETECTION ========")
print("Detected handovers:", detected_events)
print("Missed handovers:", missed_events)
print(f"Event recall: {detected_events / total_events:.4%}")

useful_episodes = 0 
false_episodes = 0 

for run_id, run_data in validation_data.groupby("run_id"):

    run_data = run_data.sort_values("time_s")

    alarms = run_data["prediction"] == 1

    episode_ids = alarms.ne(
        alarms.shift(fill_value=False)
    ).cumsum()

    handover_times = run_data.loc[
        run_data["handover_event"] == 1,
        "time_s"
        ].to_numpy()

    for _, episode in run_data.loc[alarms].groupby(episode_ids[alarms]):

            start_time = episode["time_s"].iloc[0]
            end_time = episode["time_s"].iloc[-1]

            useful = any(
                start_time < ho_time <= end_time + 1.0
                for ho_time in handover_times
            )

            if useful: 
                useful_episodes += 1
            
            else: 
                false_episodes += 1

total_episodes = false_episodes + useful_episodes

print("\n======== RANDOM FOREST ALARM ANALYSIS =======")
print("Useful episodes:", useful_episodes)
print("False alarm episodes:", false_episodes)
print("Total alarm episodes:", total_episodes)

if total_episodes > 0:
    print(
        f"Episode precision: "
        f"{100 * useful_episodes / total_episodes:.2f}%"
    )

candidate_thresholds = [best_threshold, 0.68, 0.72, 0.76, 0.80]

threshold_results = []

for threshold in candidate_thresholds:

    validation_data["prediction"] = (y_prob >= threshold).astype(int)

    detected = 0
    useful = 0 
    false = 0

    for run_id, run_data in validation_data.groupby("run_id"):

        run_data = run_data.sort_values("time_s")

        handover_times = run_data.loc[
            run_data["handover_event"] == 1, 
            "time_s"
        ].to_numpy()

        for ho_time in handover_times:

            prior_window = run_data[
                (run_data["time_s"] >= ho_time - 1.0)
                &
                (run_data["time_s"] < ho_time)
            ]

            if (prior_window["prediction"] == 1).any():
                detected += 1
            
            alarms = run_data["prediction"] == 1

            episode_ids = alarms.ne(alarms.shift(fill_value=False)).cumsum()

            for _, episode in run_data.loc[alarms].groupby(episode_ids[alarms]):

                start = episode["time_s"].iloc[0]
                end = episode["time_s"].iloc[-1]

                is_useful = any(
                    start < ho_time <= end + 1.0
                    for ho_time in handover_times
                )

                if is_useful: 
                    useful += 1
            
                else: 
                    false += 1

            threshold_results.append({
                "threshold": threshold,
                "event_recall": detected / total_events,
                "detected_handovers": detected,
                "missed_handovers": total_events - detected,
                "useful_episodes": useful,
                "false_episodes": false,
                "episode_precision": useful / (useful + false)
                if useful + false > 0 else 0
            })        

threshold_results_df = pd.DataFrame(threshold_results)

print("\n======= RANDOM FOREST THRESHOLD COMPARISON =======")
print(threshold_results_df.to_string(index=False))