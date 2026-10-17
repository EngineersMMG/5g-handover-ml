import pandas as pd
from xgboost import XGBClassifier
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.inspection import permutation_importance

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

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = negative_count / positive_count
print("Negative samples:", negative_count)
print("Positive samples:", positive_count)
print("scale_pos_weight:", scale_pos_weight)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)
print(model)

print("Training XGBoost...")
model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=10)
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

sample_indices = X_validation.sample(
    n=30000,
    random_state=42
).index

X_perm = X_validation.loc[sample_indices]
y_perm = y_validation.loc[sample_indices]
print("Permutation sample", X_perm.shape)
print("Target sample", y_perm.shape)

perm_result = permutation_importance(
    model,
    X_perm,
    y_perm,
    scoring="average_precision",
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)

perm_importance = pd.DataFrame({
    "feature": X_perm.columns,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std
})
perm_importance = perm_importance.sort_values(
    "importance_mean",
    ascending=False
)
print(perm_importance.head(10))

X_train_ablation = X_train.drop(columns=["serving_cell_is_2"])
X_validation_ablation = X_validation.drop(columns=["serving_cell_is_2"])
print(X_train_ablation.shape)
print(X_validation_ablation.shape)

model_ablation = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)
print("Training ablation XGBoost...")
model_ablation.fit(X_train_ablation, y_train, eval_set=[(X_validation_ablation, y_validation)], verbose=50)
print("Training complete.")

y_prob_ablation = model_ablation.predict_proba(X_validation_ablation)[:, 1]
ap_ablation = average_precision_score(y_validation, y_prob_ablation)
print("Original AP:", average_precision)
print("Ablation AP:", ap_ablation)

precisions_ablation, recalls_ablation, thresholds_ablation = precision_recall_curve(
    y_validation,
    y_prob_ablation
)

f1_scores_ablation = (
    2 
    * precisions_ablation[:-1] 
    * recalls_ablation[:-1] 
    / (
        precisions_ablation[:-1]
        + recalls_ablation[:-1] 
        + 1e-12
    )
)
print("F1 scores ablation:", f1_scores_ablation)

best_index_ablation = np.argmax(f1_scores_ablation)

best_threshold_ablation = thresholds_ablation[best_index_ablation]

print("Ablation best threshold:", best_threshold_ablation)
print("Ablation best F1:", f1_scores_ablation[best_index_ablation])
print("Ablation precision:", precisions_ablation[best_index_ablation])
print("Ablation recall:", recalls_ablation[best_index_ablation])

y_pred_ablation_tuned = (
    y_prob_ablation >= best_threshold_ablation
).astype(int)
print("Verified ablation F1:",
f1_score(y_validation,y_pred_ablation_tuned))