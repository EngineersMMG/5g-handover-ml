import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

def handover_was_detected(
    run_data, 
    handover_time,
    prediction_window = 1.0
): 

    prediction_window_rows = run_data[
        (run_data["time_s"] >= handover_time - prediction_window)
        &
        (run_data["time_s"] < handover_time)
    ]

    detected = (
        prediction_window_rows["prediction"] == 1
    ).any()

    return detected

def get_alarm_episodes(run_data):

    run_data = run_data.sort_values("time_s").copy()

    episodes = []

    in_episodes = False
    start_time = None
    end_time = None

    for _, row in run_data.iterrows():

        if row["prediction"] == 1:

            if not in_episodes:
                start_time = row["time_s"]
                in_episodes = True
            
            end_time = row["time_s"]

        elif in_episodes:

            episodes.append(
                (start_time, end_time)
            )
            in_episodes = False
        
    if in_episodes:
        episodes.append(
            (start_time, end_time)
        )

    return episodes   

def alarm_episode_is_useful(
    start_time,
    end_time,
    handover_times,
    prediction_window=1.0
):

    for handover_time in handover_times:
        if (
            start_time < handover_time
            <= end_time + prediction_window
        ):
         
            return True
    
    return False

data = pd.read_csv("data/processed/feature-dataset.csv")
train_scenarios = pd.read_csv("data/splits/train_scenarios.csv")["scenario_id"]
validation_scenarios = pd.read_csv("data/splits/validation_scenarios.csv")["scenario_id"]

train_data = data[
    data["scenario_id"].isin(train_scenarios)
    ].copy()
validation_data = data[
    data["scenario_id"].isin(validation_scenarios)
    ].copy()

target = "handover_next_1s"

metadata_columns = [
    "scenario_id",
    "run_id",
    "repeat_id",
    "time_s",
    "handover_event",
    target
]

X_train = train_data.drop(
    columns=metadata_columns
)

y_train = train_data[target]

X_validation = validation_data.drop(
    columns=metadata_columns
)

y_validation = validation_data[target]

print("Train shape:", X_train.shape)
print("Validation shape:", X_validation.shape)
print("Train features:", X_train.shape[1])
print("Validation runs:", validation_data["run_id"].nunique())

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)

model = LogisticRegression(
    C=10.0,
    max_iter=1000,
    class_weight=None,
    random_state=42
)

print("Training tuned Logistic Regression...")
model.fit(X_train_scaled, y_train)
print("Training complete")

y_prob_validation = model.predict_proba(
    X_validation_scaled
)[:, 1]

validation_data["handover_probability"] = y_prob_validation

precisions, recalls, thresholds = precision_recall_curve(
    y_validation,
    y_prob_validation
)

recall_targets = [0.62, 0.621, 0.622, 0.623, 0.624, 0.625, 0.626, 0.627, 0.628, 0.629, 0.63]
operating_thresholds = {}

for target_recall in recall_targets:
    valid_indices = np.where(
        recalls[:-1] >= target_recall
    )[0]

    best_index = valid_indices[
        np.argmax(
            precisions[:-1][valid_indices]
        )
    ]

    threshold = thresholds[best_index]
    operating_thresholds[target_recall] = {
        "threshold": threshold, 
        "row_precision": precisions[best_index],
        "row_recall" : recalls[best_index]
    }

    print(
        "Recall target:",
        target_recall,
        "| Threshold:",
        threshold,
        "| Precision:",
        precisions[best_index],
        "| Actual recall:",
        recalls[best_index]
    )

print("Probability rows:", len(y_prob_validation))
print("Validation rows:", len(validation_data))
print(validation_data[
    [
        "run_id",
        "time_s",
        "handover_event",
        "handover_next_1s",
        "handover_probability"
    ]
].head(10)
)

event_results = []

for target_recall, operating_point in operating_thresholds.items():

    threshold = operating_point["threshold"]
    row_precision = operating_point["row_precision"]
    row_recall = operating_point["row_recall"]

    validation_data["prediction"] = (
        validation_data["handover_probability"]
        >= threshold
    ).astype(int)

    detected_handovers = 0
    missed_handovers = 0 
    useful_alarm_episodes = 0
    false_alarm_episodes = 0 

    for run_id, run_data in validation_data.groupby("run_id"):

        handover_times = (
            run_data.loc[
                run_data["handover_event"] == 1,
                "time_s"
            ]
            .sort_values()
            .to_numpy()
        )

        for handover_time in handover_times:

            if handover_was_detected(run_data, handover_time):
                detected_handovers += 1
            
            else:
                missed_handovers += 1
        
        episodes = get_alarm_episodes(run_data)

        for start_time, end_time in episodes:

            if alarm_episode_is_useful(start_time, end_time, handover_times):
                useful_alarm_episodes += 1
            
            else: 
                false_alarm_episodes += 1

    event_recall = (
        detected_handovers
        /(detected_handovers + missed_handovers)
    )

    episode_precision = (
        useful_alarm_episodes
        /(useful_alarm_episodes + false_alarm_episodes)
    )

    total_alarm_episodes = useful_alarm_episodes + false_alarm_episodes

    event_results.append({
        "target_recall": target_recall,
        "threshold": threshold,
        "row_precision": row_precision,
        "row_recall": row_recall,
        "detected_handovers": detected_handovers,
        "missed_handovers": missed_handovers,
        "event_recall": event_recall,
        "useful_alarm_episodes": useful_alarm_episodes,
        "false_alarm_episodes": false_alarm_episodes,
        "episode_precision": episode_precision,
        "total_alarm_episodes": total_alarm_episodes
    })

event_results_df = pd.DataFrame(event_results)
print("\n======= Event-Level Threshold Comparison =====")
print(event_results_df.to_string(index=False))

candidate_threshold = operating_thresholds[0.624]["threshold"]

validation_data["prediction"] = (
    validation_data["handover_probability"]
    >= candidate_threshold
).astype(int)

print("\nInspecting missed handovers at threhold:", candidate_threshold)

missed_event_details = []

for run_id, run_data in validation_data.groupby("run_id"):

    handover_times = (
        run_data.loc[
            run_data["handover_event"] == 1, 
            "time_s"
        ]
        .sort_values()
        .to_numpy()
    )

    for handover_time in handover_times:

        if not handover_was_detected(run_data, handover_time):

            missed_event_details.append({
                "run_id": run_id,
                "handover_time": handover_time
            })

missed_event_df = pd.DataFrame(missed_event_details)
print("\nMissed handovers:", missed_event_df.to_string(index=False))
print("\nTotal missed:", len(missed_event_df))

miss_analysis = []

for _, missed_event in missed_event_df.iterrows():

    run_id = missed_event["run_id"]
    handover_time = missed_event["handover_time"]

    # Skip the too early-events
    if handover_time <= 2.2: 
        continue

    run_data = validation_data[validation_data["run_id"] == run_id]

    prior_window = run_data[
        (run_data["time_s"] >= handover_time - 1.0)
        &
        (run_data["time_s"] < handover_time)
    ]

    max_probability = prior_window[
        "handover_probability"
    ].max()

    miss_analysis.append({
        "run_id": run_id,
        "handover_time": handover_time, 
        "max_prior_probability": max_probability,
        "distance_from_threshold": candidate_threshold - max_probability
    })

miss_analysis_df = pd.DataFrame(miss_analysis) 

print("\n===== Evaluable Missed Handover Analysis =====")
print(miss_analysis_df.to_string(index=False))