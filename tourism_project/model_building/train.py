# train.py
# Trains a Random Forest model using:
#   make_column_transformer  — scales numerics, one-hot encodes categoricals
#   make_pipeline            — bundles preprocessor + model into one object
#   GridSearchCV             — tunes hyperparameters with 5-fold CV
# The best pipeline is saved with joblib and uploaded to Hugging Face Model Hub.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ── Constants ──────────────────────────────────────────────────────────
DATASET_REPO = "shawsushant/tourism-package-prediction"
MODEL_REPO   = "shawsushant/tourism-package-prediction-model"
MODEL_FILE   = "best_tourism_model_v1.joblib"

# ── 1. Load train / test splits from Hugging Face ──────────────────────
Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").squeeze()
ytest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").squeeze()
print(f"Xtrain: {Xtrain.shape}  |  Xtest: {Xtest.shape}")
print(f"Xtrain dtypes:\n{Xtrain.dtypes}")

# ── 2. Class weight to handle imbalance ────────────────────────────────
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class weight ratio: {class_weight:.2f}")

# ── 3. Pipeline: StandardScaler -> RandomForestClassifier ──────────────
# All features are numeric (label-encoded in prep.py) so a single
# StandardScaler over the full feature set is correct here.
rf_model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
model_pipeline = make_pipeline(StandardScaler(), rf_model)

# ── 4. Hyperparameter grid ─────────────────────────────────────────────
param_grid = {
    'randomforestclassifier__n_estimators':      [100, 200],
    'randomforestclassifier__max_depth':         [5, 8, None],
    'randomforestclassifier__max_features':      ['sqrt', 'log2'],
    'randomforestclassifier__min_samples_split': [2, 5],
}

# ── 5. GridSearchCV with 5-fold cross-validation ───────────────────────
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)
grid_search.fit(Xtrain, ytrain)

# ── 6. Best model ──────────────────────────────────────────────────────
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# ── 7. Predict on training and test sets ───────────────────────────────
y_pred_train = best_model.predict(Xtrain)
y_pred_test  = best_model.predict(Xtest)

# ── 8. Evaluation ──────────────────────────────────────────────────────
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train,
      target_names=['Not Purchased', 'Purchased']))

print("Test Classification Report:")
print(classification_report(ytest, y_pred_test,
      target_names=['Not Purchased', 'Purchased']))

# ── 9. Save best model ─────────────────────────────────────────────────
joblib.dump(best_model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# ── 10. Upload to Hugging Face Model Hub ──────────────────────────────
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repo '{MODEL_REPO}' not found. Creating new repo...")
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
    print(f"Model repo '{MODEL_REPO}' created.")

api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Model registered on Hugging Face Model Hub: {MODEL_REPO}")
