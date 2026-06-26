"""Aggregate results/*.json into a comparison table (markdown).

Usage: python scripts/report.py
Reads every results/<name>.json that has a 'summary' block and prints a table
of accuracy +/- 95% CI, no-answer rate, and base->fine-tuned deltas per set.
"""
import glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_of(data_path):
    return os.path.splitext(os.path.basename(data_path))[0]  # amc / aime / math500


def main():
    rows = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        try:
            s = json.load(open(fp)).get("summary")
        except Exception:
            continue
        if not s:
            continue
        s["set"] = set_of(s["data"])
        s["kind"] = "fine-tuned" if s.get("adapter") else "base"
        rows.append(s)

    if not rows:
        print("No results yet.")
        return

    print("| run | set | kind | shots | n | accuracy | 95% CI | no-ans |")
    print("|---|---|---|---|---|---|---|---|")
    for s in sorted(rows, key=lambda r: (r["set"], r["kind"], r.get("few_shot", 0))):
        lo, hi = s["ci95"]
        print(f"| {s['name']} | {s['set']} | {s['kind']} | {s.get('few_shot',0)} | "
              f"{s['n']} | {s['accuracy']*100:.1f}% | "
              f"[{lo*100:.1f}, {hi*100:.1f}] | {s.get('no_answer',0)} |")

    # base -> fine-tuned deltas per set (zero-shot only)
    print("\n### Base -> fine-tuned (zero-shot) deltas")
    by = {}
    for s in rows:
        if s.get("few_shot", 0) == 0:
            by.setdefault(s["set"], {})[s["kind"]] = s
    print("| set | base | fine-tuned | delta |")
    print("|---|---|---|---|")
    for st, d in sorted(by.items()):
        if "base" in d and "fine-tuned" in d:
            b, f = d["base"]["accuracy"], d["fine-tuned"]["accuracy"]
            print(f"| {st} | {b*100:.1f}% | {f*100:.1f}% | {(f-b)*100:+.1f} pts |")
        elif "base" in d:
            print(f"| {st} | {d['base']['accuracy']*100:.1f}% | (pending) | - |")


if __name__ == "__main__":
    main()
