"""Build the AMC-tutor train/val/eval data.

Pipeline:
  1. Load MATH train (EleutherAI/hendrycks_math, 7 subject configs) -> filter to
     levels 3-5 -> extract \\boxed{} answers.
  2. (optional, --numina) add NuminaMath-CoT rows from AMC-style sources.
  3. Load held-out EVAL sets: aimo-validation-amc (83), aimo-validation-aime (90),
     MATH-500 (500, contaminated-tier reference).
  4. DECONTAMINATE train against every eval problem (exact / 13-gram / 8-gram
     Jaccard) and drop overlaps. Log how many were removed.
  5. Exact-dedup train, split train/val, render to chat format, write JSONL.

Outputs (under --out):
  data/processed/train.jsonl, val.jsonl     ({"messages": [...]})
  eval/amc.jsonl, eval/aime.jsonl, eval/math500.jsonl   ({"problem","answer",...})
  results/data_report.json
"""
import argparse
import json
import random
from collections import defaultdict, Counter

from datasets import load_dataset, get_dataset_config_names
from common import (SYSTEM_PROMPT, extract_boxed, normalize_text, word_ngrams,
                    normalize_answer)

MATH_NAME = "EleutherAI/hendrycks_math"
NUMINA_NAME = "AI-MO/NuminaMath-CoT"
NUMINA_SOURCES = {"amc_aime", "synthetic_amc"}  # AMC-distribution only


def parse_level(level_str):
    # "Level 3" -> 3 ; unknown -> None
    try:
        return int(str(level_str).strip().split()[-1])
    except (ValueError, IndexError):
        return None


def load_math(levels):
    records = []
    for cfg in get_dataset_config_names(MATH_NAME):
        ds = load_dataset(MATH_NAME, cfg, split="train")
        for i, ex in enumerate(ds):
            lvl = parse_level(ex.get("level"))
            if lvl is None or lvl not in levels:
                continue
            sol = (ex.get("solution") or "").strip()
            ans = extract_boxed(sol)
            if not sol or not ans:
                continue
            records.append({
                "id": f"math/{cfg}/{i}",
                "source": "MATH",
                "subject": ex.get("type") or cfg,
                "level": lvl,
                "problem": (ex.get("problem") or "").strip(),
                "solution": sol,
                "answer": ans,
            })
    return records


def load_numina(cap):
    ds = load_dataset(NUMINA_NAME, split="train")
    ds = ds.filter(lambda x: x.get("source") in NUMINA_SOURCES,
                   desc="filter numina sources")
    records = []
    for i, ex in enumerate(ds):
        sol = (ex.get("solution") or "").strip()
        ans = extract_boxed(sol)
        if not sol or not ans:
            continue
        records.append({
            "id": f"numina/{ex.get('source')}/{i}",
            "source": f"numina:{ex.get('source')}",
            "subject": None,
            "level": None,
            "problem": (ex.get("problem") or "").strip(),
            "solution": sol,
            "answer": ans,
        })
    random.shuffle(records)
    return records[:cap]


def load_eval_sets():
    """Return dict name -> list of {problem, answer, ...} and a flat list of all
    eval problem strings (for decontamination)."""
    out = {}

    amc = load_dataset("AI-MO/aimo-validation-amc", split="train")
    out["amc"] = [{"id": f"amc/{r['id']}", "problem": r["problem"].strip(),
                   "answer": normalize_answer(r["answer"]), "url": r.get("url")}
                  for r in amc]

    aime = load_dataset("AI-MO/aimo-validation-aime", split="train")
    out["aime"] = [{"id": f"aime/{r['id']}", "problem": r["problem"].strip(),
                    "answer": normalize_answer(r["answer"]), "url": r.get("url")}
                   for r in aime]

    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out["math500"] = [{"id": r.get("unique_id", f"math500/{i}"),
                       "problem": r["problem"].strip(),
                       "answer": normalize_answer(r["answer"]),
                       "subject": r.get("subject"), "level": r.get("level")}
                      for i, r in enumerate(m500)]
    return out


# ---------------- contamination index ----------------
def build_test_index(problems, n_long=13, n_short=8):
    exact, long_grams, short_sets, inv = set(), set(), [], defaultdict(set)
    for i, p in enumerate(problems):
        norm = normalize_text(p)
        exact.add(norm)
        for g in word_ngrams(norm, n_long):
            long_grams.add(g)
        sg = set(word_ngrams(norm, n_short))
        short_sets.append(sg)
        for g in sg:
            inv[g].add(i)
    return exact, long_grams, short_sets, inv


def contam_reason(problem, idx, n_long=13, n_short=8, jacc=0.6):
    exact, long_grams, short_sets, inv = idx
    norm = normalize_text(problem)
    if norm in exact:
        return "exact"
    for g in word_ngrams(norm, n_long):
        if g in long_grams:
            return "ngram13"
    sg = set(word_ngrams(norm, n_short))
    if not sg:
        return None
    cand = set()
    for g in sg:
        c = inv.get(g)
        if c:
            cand |= c
    for i in cand:
        ts = short_sets[i]
        union = len(sg | ts)
        if union and len(sg & ts) / union >= jacc:
            return "jaccard"
    return None


def to_chat(rec):
    assistant = rec["solution"].strip()
    assistant += f"\n\nFinal answer: \\boxed{{{rec['answer']}}}"
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": rec["problem"]},
        {"role": "assistant", "content": assistant},
    ]}


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Users/YNA/amc-tutor")
    ap.add_argument("--levels", default="3,4,5")
    ap.add_argument("--numina", action="store_true")
    ap.add_argument("--numina-cap", type=int, default=6000)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    levels = {int(x) for x in args.levels.split(",")}

    print(f"[1/5] Loading MATH train, levels={sorted(levels)} ...")
    train = load_math(levels)
    print(f"      MATH records: {len(train)}")
    if args.numina:
        print(f"[1b ] Loading NuminaMath ({NUMINA_SOURCES}), cap={args.numina_cap} ...")
        nm = load_numina(args.numina_cap)
        print(f"      NuminaMath records: {len(nm)}")
        train += nm

    print("[2/5] Loading eval sets ...")
    evals = load_eval_sets()
    for k, v in evals.items():
        print(f"      {k}: {len(v)}")

    print("[3/5] Building contamination index from all eval problems ...")
    all_test_problems = [e["problem"] for v in evals.values() for e in v]
    idx = build_test_index(all_test_problems)

    print("[4/5] Decontaminating + dedup ...")
    seen = set()
    kept, removed = [], Counter()
    for rec in train:
        norm = normalize_text(rec["problem"])
        if norm in seen:
            removed["dup"] += 1
            continue
        reason = contam_reason(rec["problem"], idx)
        if reason:
            removed[f"contam:{reason}"] += 1
            continue
        seen.add(norm)
        kept.append(rec)
    print(f"      kept {len(kept)} / {len(train)}  | removed {dict(removed)}")

    print("[5/5] Splitting + writing ...")
    random.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_frac))
    val, tr = kept[:n_val], kept[n_val:]
    write_jsonl(f"{args.out}/data/processed/train.jsonl", [to_chat(r) for r in tr])
    write_jsonl(f"{args.out}/data/processed/val.jsonl", [to_chat(r) for r in val])
    for name, rows in evals.items():
        write_jsonl(f"{args.out}/eval/{name}.jsonl", rows)

    report = {
        "train": len(tr), "val": len(val),
        "removed": dict(removed),
        "by_source": dict(Counter(r["source"] for r in kept)),
        "by_level": dict(Counter(r["level"] for r in kept)),
        "by_subject": dict(Counter(r["subject"] for r in kept)),
        "eval_sizes": {k: len(v) for k, v in evals.items()},
        "config": vars(args),
    }
    with open(f"{args.out}/results/data_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n=== SUMMARY ===")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
