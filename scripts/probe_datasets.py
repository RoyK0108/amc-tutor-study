"""Probe candidate datasets: report which load, their columns, and a sample row.
Uses streaming so we don't download full files just to inspect schemas.
"""
import datasets
from datasets import load_dataset

print("datasets version:", datasets.__version__)


def probe(name, split="train", config=None, n=2):
    tag = f"{name}" + (f" [{config}]" if config else "")
    print(f"\n===== {tag} (split={split}) =====")
    try:
        kwargs = dict(split=split, streaming=True)
        if config:
            ds = load_dataset(name, config, **kwargs)
        else:
            ds = load_dataset(name, **kwargs)
        rows = []
        for i, ex in enumerate(ds):
            rows.append(ex)
            if i + 1 >= n:
                break
        if not rows:
            print("  (no rows)")
            return
        print("  columns:", list(rows[0].keys()))
        ex = rows[0]
        for k, v in ex.items():
            s = str(v).replace("\n", " ")
            if len(s) > 220:
                s = s[:220] + "..."
            print(f"    {k}: {s}")
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


# --- MATH training set: try several mirrors (original uses a loading script) ---
probe("hendrycks/competition_math", split="train")
probe("EleutherAI/hendrycks_math", split="train", config="algebra")
probe("nlile/hendrycks-MATH-benchmark", split="train")
probe("lighteval/MATH", split="train", config="all")
probe("HuggingFaceH4/MATH-500", split="test")  # 500-problem eval subset

# --- NuminaMath (large competition CoT corpus) ---
probe("AI-MO/NuminaMath-CoT", split="train")

# --- Contamination-controlled test sets ---
probe("AI-MO/aimo-validation-amc", split="train")
probe("AI-MO/aimo-validation-aime", split="train")
