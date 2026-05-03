# data_register.py — Register the raw dataset on Hugging Face Datasets Hub
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

REPO_ID   = "shawsushant/tourism-package-prediction"
REPO_TYPE = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"Dataset repo '{REPO_ID}' created.")

# Upload   data folder to the dataset repo
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)
