"""Upload the dataset to the Hugging Face Hub.

Auth via the HF_TOKEN env var (or a prior `huggingface-cli login`).
Usage:
  HF_TOKEN=hf_xxx python scripts/upload_hf.py [repo-name]

Creates <your-username>/<repo-name> (default: amc-tutor-sft) and uploads hf_dataset/.
The username is derived from the token, so you don't pass it.
"""
import os, sys
from huggingface_hub import HfApi

name = sys.argv[1] if len(sys.argv) > 1 else "amc-tutor-sft"
api = HfApi(token=os.environ.get("HF_TOKEN"))
user = api.whoami()["name"]
repo = f"{user}/{name}"
api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(
    folder_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_dataset"),
    repo_id=repo, repo_type="dataset",
    commit_message="Add AMC-tutor decontaminated SFT dataset + card",
)
print(f"done -> https://huggingface.co/datasets/{repo}")
