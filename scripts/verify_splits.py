"""Read split sizes from dataset metadata (fast, no full download) so we never
mistake a test set for training data."""
from datasets import get_dataset_config_names, load_dataset_builder

names = [
    "EleutherAI/hendrycks_math",
    "nlile/hendrycks-MATH-benchmark",
    "HuggingFaceH4/MATH-500",
    "AI-MO/aimo-validation-amc",
    "AI-MO/aimo-validation-aime",
]
for name in names:
    print("\n#", name)
    try:
        cfgs = get_dataset_config_names(name)
        print("  configs:", cfgs)
        for c in cfgs:
            b = load_dataset_builder(name, c)
            splits = {k: v.num_examples for k, v in (b.info.splits or {}).items()}
            print(f"  [{c}] splits:", splits)
    except Exception as e:
        print("  ERR:", type(e).__name__, e)
