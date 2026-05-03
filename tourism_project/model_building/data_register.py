# data_register.py — Register the raw dataset on Hugging Face Datasets Hub
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

REPO_ID   = "shawsushant/tourism-package-prediction"
REPO_TYPE = "dataset"

# Authenticate using the HF_TOKEN environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the dataset repo if it does not exist yet
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"Dataset repo '{REPO_ID}' created.")

# Upload the entire data folder to the dataset repo
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)
print("Raw dataset uploaded to Hugging Face Dataset Hub successfully.")
