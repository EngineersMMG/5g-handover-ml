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

X_train = train_data.drop(
    columns=["scenario_id", "run_id", "repeat_id", target]
)
y_train = train_data[target]

X_validation = validation_data.drop(
    columns=["scenario_id", "run_id", "repeat_id", target]
)
y_validation = validation_data[target]

print("X train shape:", X_train.shape)
print("y train shape:", y_train.shape)
print("X validation shape:", X_validation.shape)
print("y validation shape:", y_validation.shape)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
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