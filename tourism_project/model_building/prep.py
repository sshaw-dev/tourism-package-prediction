# prep.py — Load from HF Hub, clean, split, and re-upload train/test sets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
import pickle
import os

# ── Constants ──────────────────────────────────────────────────────────
REPO_ID      = "shawsushant/tourism-package-prediction"
DATASET_PATH = f"hf://datasets/{REPO_ID}/tourism.csv"

api = HfApi(token=os.getenv("HF_TOKEN"))

# ── 1. Load dataset directly from Hugging Face ─────────────────────────
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded. Shape: {df.shape}")

# ── 2. Drop unnecessary columns ────────────────────────────────────────
drop_cols = [c for c in ['Unnamed: 0', 'CustomerID'] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print(f"Dropped columns: {drop_cols}")

# ── 3. Impute missing values ────────────────────────────────────────────
num_cols  = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols  = df.select_dtypes(include=['object']).columns.tolist()
target    = 'ProdTaken'
num_feats = [c for c in num_cols if c != target]

for col in num_feats:
    df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("Missing values imputed.")

# ── 4. Label-encode categorical columns and save encoders ──────────────
# Encoders are saved so the Streamlit app can apply the same transformation
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print(f"Encoders saved for columns: {list(encoders.keys())}")

# ── 5. Train-test split ─────────────────────────────────────────────────
X = df.drop(columns=[target])
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {Xtrain.shape}, Test size: {Xtest.shape}")

# ── 6. Save splits locally ─────────────────────────────────────────────
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv",  index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv",  index=False)
print("Train/test CSVs saved locally.")

# ── 7. Upload splits + encoders back to Hugging Face ───────────────────
for fname in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv", "encoders.pkl"]:
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=fname,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded {fname} to {REPO_ID}")
print("Data preparation complete.")
