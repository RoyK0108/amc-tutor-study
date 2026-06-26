"""Build a side-by-side base-vs-fine-tuned markdown for blind teaching-quality
grading. Reads two results/*.json (from evaluate.py), matches by problem id, and
writes the problem + gold + both full solutions so a human can score pedagogy.

Usage:
  python scripts/compare.py --base results/amc_base_zeroshot.json \
      --ft results/amc_ft.json --problems eval/amc.jsonl --n 15 \
      --out results/compare_amc.md
"""
import argparse, json, os, random


def load_results(path):
    d = json.load(open(path))
    return {str(r["id"]): r for r in d["results"]}, d["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base, bsum = load_results(args.base)
    ft, fsum = load_results(args.ft)
    probs = {str(r["id"]): r for r in (json.loads(l) for l in open(args.problems))}
    ids = [i for i in probs if i in base and i in ft]
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.n]

    out = [f"# Base vs Fine-tuned — qualitative comparison",
           f"Base `{os.path.basename(args.base)}` acc {bsum['accuracy']*100:.1f}% | "
           f"Fine-tuned `{os.path.basename(args.ft)}` acc {fsum['accuracy']*100:.1f}%",
           "Grade each solution 1-5: steps-correct / clarity / key-idea / pedagogy.\n"]
    for k, i in enumerate(ids, 1):
        p, b, f = probs[i], base[i], ft[i]
        out += [f"\n---\n## {k}. `{i}` (gold: `{p['answer']}`)",
                f"**Problem:** {p['problem']}\n",
                f"### Base — pred `{b['pred']}` {'OK' if b['correct'] else 'WRONG'}",
                b["generation"], "",
                f"### Fine-tuned — pred `{f['pred']}` {'OK' if f['correct'] else 'WRONG'}",
                f["generation"], ""]
    with open(args.out, "w") as fh:
        fh.write("\n".join(out))
    print(f"wrote {args.out} with {len(ids)} problems")


if __name__ == "__main__":
    main()
