"""Generates notebooks/self_distillation_colab.ipynb programmatically (avoids
hand-escaping notebook JSON). Run once: python scripts/_make_notebook.py"""
import json, os

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# AMC Tutor — Phase 2: Self-Distillation (STaR) on a free GPU

The local study ([repo](https://github.com/RoyK0108/amc-tutor-study)) found that **naive SFT on terse
competition-math solutions *regresses* small-model reasoning** (Qwen3-1.7B/4B both got *worse* on held-out
AMC; validation loss hid it). This notebook implements the principled fix — **self-distillation / STaR** —
on a free Colab GPU.

**Idea:** don't imitate terse human solutions; train the model on **its own *correct* reasoning**:
1. sample several solutions per training problem from the base model,
2. keep only those whose final answer is correct (verified with `math_verify`),
3. fine-tune (LoRA) on those correct, reasoning-rich traces,
4. evaluate on the same contamination-controlled AMC/AIME held-out sets.

> Runtime → Change runtime type → **GPU** (T4 works; L4/A100 faster).""")

md("""## Why this avoids the regression
Naive SFT taught the model to mimic a *terse style* and skip reasoning. STaR reinforces the model's *own*
verbose, **correct** chains instead of overwriting them — the small-scale version of the rejection-sampling
used to bootstrap strong open math models.""")

code("""!pip install -q unsloth math_verify datasets""")

code("""import torch, random, re
random.seed(42)

SYSTEM_PROMPT = ("You are an expert AMC (American Mathematics Competitions) math tutor. "
                 "Solve with clear, step-by-step reasoning, then give the final result on its "
                 "own line as 'Final answer: \\\\boxed{...}'.")
BASE = "unsloth/Qwen2.5-Math-7B-Instruct"   # strong small math base; or "unsloth/Qwen3-4B"
MAX_SEQ   = 2048
N_PROBLEMS = 800     # training problems to distill from (raise if you have GPU time)
N_SAMPLES  = 4       # candidate solutions sampled per problem
GEN_TEMP   = 0.8""")

code("""from math_verify import parse, verify

def extract_boxed(t):
    i = t.rfind("\\\\boxed")
    if i < 0: return None
    i += 6
    while i < len(t) and t[i] != "{":
        if t[i].isspace(): i += 1
        else: return None
    if i >= len(t) or t[i] != "{": return None
    d, s = 0, i
    for j in range(i, len(t)):
        if t[j] == "{": d += 1
        elif t[j] == "}":
            d -= 1
            if d == 0: return t[s+1:j].strip()
    return None

def is_correct(gold, pred):
    if pred is None: return False
    if str(gold).strip() == str(pred).strip(): return True
    try: return bool(verify(parse(str(gold)), parse(str(pred))))
    except Exception: return False""")

code("""from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None)
tok.padding_side = "left"   # correct for batched generation
FastLanguageModel.for_inference(model)""")

md("""## Step 1 — sample candidate solutions, keep the correct ones (the STaR filter)""")

code("""from datasets import load_dataset, get_dataset_config_names

probs = []
for cfg in get_dataset_config_names("EleutherAI/hendrycks_math"):
    for ex in load_dataset("EleutherAI/hendrycks_math", cfg, split="train"):
        if ex.get("level") in ("Level 3", "Level 4", "Level 5"):
            g = extract_boxed(ex.get("solution", ""))
            if g: probs.append({"problem": ex["problem"], "gold": g})
random.shuffle(probs); probs = probs[:N_PROBLEMS]
print(len(probs), "problems to distill from")""")

code("""def build_prompt(p):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

traces = []
for k, item in enumerate(probs):
    ins = tok([build_prompt(item["problem"])] * N_SAMPLES, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**ins, max_new_tokens=1024, do_sample=True, temperature=GEN_TEMP, top_p=0.95)
    for o in out:
        text = tok.decode(o[ins["input_ids"].shape[1]:], skip_special_tokens=True)
        if is_correct(item["gold"], extract_boxed(text)):
            traces.append({"problem": item["problem"], "solution": text})
            break   # one correct trace per problem is enough
    if (k + 1) % 50 == 0:
        print(f"{k+1}/{len(probs)} processed, {len(traces)} correct traces")
print("kept", len(traces), "correct self-distilled traces")""")

md("""## Step 2 — LoRA fine-tune on the correct self-generated traces""")

code("""from datasets import Dataset

def to_text(t):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": t["problem"]},
            {"role": "assistant", "content": t["solution"]}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}

train_ds = Dataset.from_list(traces).map(to_text)

FastLanguageModel.for_training(model)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)""")

code("""from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=train_ds,
    args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=20, num_train_epochs=2, learning_rate=1e-4, lr_scheduler_type="cosine",
        optim="adamw_8bit", logging_steps=10, seed=42, output_dir="outputs",
        max_seq_length=MAX_SEQ, dataset_text_field="text"))
trainer.train()
model.save_pretrained("amc_star_adapter"); tok.save_pretrained("amc_star_adapter")""")

md("""## Step 3 — evaluate vs base on held-out AMC / AIME
Compare against the study's baselines (Qwen3-4B base: **47.0% AMC**). A *successful* STaR run should
**match or beat** the base — unlike naive SFT, which collapsed it.""")

code("""FastLanguageModel.for_inference(model)
def norm(a):
    a = str(a).strip()
    try:
        f = float(a); return str(int(f)) if f == int(f) else a
    except Exception:
        return a

def evaluate(repo, n=None):
    rows = load_dataset(repo, split="train")
    if n: rows = rows.select(range(min(n, len(rows))))
    c = 0
    for r in rows:
        ins = tok([build_prompt(r["problem"])], return_tensors="pt").to(model.device)
        out = model.generate(**ins, max_new_tokens=2048, do_sample=False)
        text = tok.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True)
        c += is_correct(norm(r["answer"]), extract_boxed(text))
    print(f"{repo}: {c}/{len(rows)} = {100*c/len(rows):.1f}%")

evaluate("AI-MO/aimo-validation-amc")
evaluate("AI-MO/aimo-validation-aime")""")

md("""## Optional next: GRPO (RL with verifiable rewards)
The other principled route is **GRPO** — RL where the reward is *answer correctness* (via `math_verify`),
which raises reasoning rather than overwriting it (the DeepSeek-R1 recipe). Unsloth ships a free GRPO
Colab notebook for small Qwen models; plug in the same `is_correct` reward and the AMC eval above.

### Caveats
- **Generation is the slow part** — raise `N_PROBLEMS` / `N_SAMPLES` as GPU time allows.
- Keep the eval *contamination-controlled* (the AI-MO sets post-date MATH).
- Save & download `amc_star_adapter/` to keep your fine-tune.""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs("/Users/YNA/amc-tutor/notebooks", exist_ok=True)
with open("/Users/YNA/amc-tutor/notebooks/self_distillation_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebook with", len(cells), "cells")
