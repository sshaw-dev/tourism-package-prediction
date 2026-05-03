# train.py — Hyperparameter-tuned model with MLflow tracking; saves best model to HF Hub
import pandas as pd
import numpy as np
import joblib, os, mlflow, mlflow.sklearn
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, classification_report
)
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ── Constants ──────────────────────────────────────────────────────────
DATASET_REPO = "shawsushant/tourism-package-prediction"
MODEL_REPO   = "shawsushant/tourism-package-prediction-model"
MODEL_FILE   = "best_tourism_model_v1.joblib"

# ── Load train/test from Hugging Face ──────────────────────────────────
Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").squeeze()
ytest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").squeeze()
print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")

# ── MLflow setup ───────────────────────────────────────────────────────
mlflow.set_experiment("Tourism_Package_Prediction")

# ── Candidates and their hyperparameter grids ──────────────────────────
candidates = {
    "DecisionTree": (
        DecisionTreeClassifier(class_weight='balanced', random_state=42),
        {"max_depth": [3, 5, 7], "min_samples_split": [2, 5]}
    ),
    "RandomForest": (
        RandomForestClassifier(class_weight='balanced', random_state=42),
        {"n_estimators": [100, 200], "max_depth": [5, 8]}
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]}
    ),
    "Bagging": (
        BaggingClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_samples": [0.7, 1.0]}
    ),
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                      scale_pos_weight=ytrain.value_counts()[0]/ytrain.value_counts()[1],
                      random_state=42),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
    ),
}

best_model, best_name, best_f1 = None, "", 0.0
results = {}

for name, (model, param_grid) in candidates.items():
    print(f"\nTuning {name}...")
    gs = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1, refit=True)
    gs.fit(Xtrain, ytrain)
    tuned = gs.best_estimator_

    y_pred  = tuned.predict(Xtest)
    y_proba = tuned.predict_proba(Xtest)[:, 1]
    acc  = accuracy_score(ytest, y_pred)
    f1   = f1_score(ytest, y_pred)
    rec  = recall_score(ytest, y_pred)
    auc  = roc_auc_score(ytest, y_proba)
    results[name] = {"Accuracy": acc, "F1": f1, "Recall": rec, "ROC-AUC": auc}

    # Log everything to MLflow
    with mlflow.start_run(run_name=name):
        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics({"accuracy": acc, "f1": f1, "recall": rec, "roc_auc": auc})
        mlflow.sklearn.log_model(tuned, name)

    print(f"  Best params : {gs.best_params_}")
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  Recall={rec:.4f}  AUC={auc:.4f}")
    print(classification_report(ytest, y_pred, target_names=['Not Purchased', 'Purchased']))

    if f1 > best_f1:
        best_f1, best_model, best_name = f1, tuned, name

# ── Summary table ──────────────────────────────────────────────────────
import pandas as pd
print("\n── Results Summary ──")
print(pd.DataFrame(results).T.sort_values('F1', ascending=False))
print(f"\nBest model: {best_name}  (F1={best_f1:.4f})")

# ── Save best model locally ────────────────────────────────────────────
joblib.dump(best_model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# ── Register best model on Hugging Face Model Hub ─────────────────────
api = HfApi(token=os.getenv("HF_TOKEN"))
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
    print(f"Model repo '{MODEL_REPO}' created.")

api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Best model '{best_name}' registered on Hugging Face Model Hub: {MODEL_REPO}")
