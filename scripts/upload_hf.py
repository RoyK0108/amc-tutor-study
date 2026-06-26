"""Upload the dataset to the Hugging Face Hub.

Usage:
  huggingface-cli login            # or: hf auth login   (paste a WRITE token)
  python scripts/upload_hf.py <your-hf-username>

Creates a PRIVATE dataset repo <username>/amc-tutor-sft and uploads hf_dataset/.
Flip it to public on the Hub website when you're ready.
"""
import sys
from huggingface_hub import HfApi

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/upload_hf.py <hf-username> [repo-name]")
user = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "amc-tutor-sft"
repo = f"{user}/{name}"

api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(
    folder_path="/Users/YNA/amc-tutor/hf_dataset",
    repo_id=repo,
    repo_type="dataset",
    commit_message="Add AMC-tutor decontaminated SFT dataset + card",
)
print(f"done -> https://huggingface.co/datasets/{repo}  (currently PRIVATE)")
