"""Evaluate an MLX model on an AMC/AIME/MATH eval set.

Metric: pass@1 final-answer accuracy. The model's last \\boxed{} is compared to
the gold answer with math_verify (symbolic equivalence), falling back to string
match. Reports accuracy with a bootstrap 95% CI and writes per-problem results.

Examples:
  # base model, zero-shot, on the clean AMC set
  python eval/evaluate.py --model mlx-community/Qwen3-1.7B-4bit \
      --data eval/amc.jsonl --name amc_base_zeroshot
  # fine-tuned (base + LoRA adapter)
  python eval/evaluate.py --model mlx-community/Qwen3-1.7B-4bit \
      --adapter adapters/ --data eval/amc.jsonl --name amc_ft
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from common import SYSTEM_PROMPT, extract_boxed  # noqa: E402

from mlx_lm import load, generate              # noqa: E402
from mlx_lm.sample_utils import make_sampler   # noqa: E402
from math_verify import parse, verify          # noqa: E402


def is_correct(gold, pred):
    if pred is None:
        return False
    if str(gold).strip() == str(pred).strip():
        return True
    try:
        return bool(verify(parse(str(gold)), parse(str(pred))))
    except Exception:
        return False


def bootstrap_ci(flags, n_boot=10000, seed=0):
    arr = np.asarray(flags, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def load_fewshot(path, k, seed):
    import random
    rows = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(rows)
    shots = []
    for r in rows[:k]:
        msgs = {m["role"]: m["content"] for m in r["messages"]}
        shots.append((msgs["user"], msgs["assistant"]))
    return shots


def build_prompt(tok, problem, shots):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in shots:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": problem})
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (fine-tuned)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--fewshot-src", default="data/processed/train.jsonl")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    shots = load_fewshot(args.fewshot_src, args.few_shot, args.seed) if args.few_shot else []

    print(f"loading {args.model}" + (f" + adapter {args.adapter}" if args.adapter else ""))
    load_kw = {"adapter_path": args.adapter} if args.adapter else {}
    model, tok = load(args.model, **load_kw)
    sampler = make_sampler(temp=args.temp)

    results, flags = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        prompt = build_prompt(tok, r["problem"], shots)
        try:
            gen = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens,
                           sampler=sampler, verbose=False)
        except Exception as e:
            gen = f"<ERROR: {e}>"
        pred = extract_boxed(gen)
        ok = is_correct(r["answer"], pred)
        flags.append(ok)
        results.append({"id": r.get("id", i), "gold": r["answer"], "pred": pred,
                        "correct": ok, "gen_chars": len(gen),
                        "generation": gen})
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            acc = sum(flags) / len(flags)
            el = time.time() - t0
            print(f"  [{i+1}/{len(rows)}] acc={acc:.3f} "
                  f"({el:.0f}s, {el/(i+1):.1f}s/q)", flush=True)

    acc = sum(flags) / len(flags) if flags else 0.0
    lo, hi = bootstrap_ci(flags)
    summary = {"name": args.name, "model": args.model, "adapter": args.adapter,
               "data": args.data, "n": len(rows), "few_shot": args.few_shot,
               "temp": args.temp, "accuracy": acc, "ci95": [lo, hi],
               "n_correct": int(sum(flags)),
               "no_answer": int(sum(1 for r in results if r["pred"] is None)),
               "seconds": time.time() - t0}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/{args.name}.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print("\n=== RESULT ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
