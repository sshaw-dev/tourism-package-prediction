# hosting.py — Push all deployment files to the Hugging Face Space
from huggingface_hub import HfApi
import os

SPACE_REPO = "shawsushant/tourism-package-prediction"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload everything inside the deployment folder to the HF Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print(f"Deployment files pushed to HF Space: shawsushant/tourism-package-prediction")
print(f"Space URL: https://huggingface.co/spaces/shawsushant/tourism-package-prediction")
