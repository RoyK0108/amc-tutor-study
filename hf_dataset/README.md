---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- mathematics
- competition-math
- amc
- aime
- reasoning
- chain-of-thought
- decontaminated
pretty_name: AMC Tutor — decontaminated competition-math SFT set
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
---

# AMC Tutor — decontaminated competition-math SFT dataset

Chat-formatted, **decontaminated** supervised-fine-tuning data for AMC 10/12-style
competition mathematics. Built for a reproducible **$0, local (MacBook M4)** study of QLoRA
fine-tuning small models. Each row is a tutor system prompt + problem + step-by-step solution
ending in `Final answer: \boxed{...}`.

Companion study & code: **https://github.com/RoyK0108/amc-tutor-study**

## ⚠️ This is a study artifact — read the finding first
The study's **headline result is negative**: naive SFT on these (terse) solutions *degrades*
small-model reasoning — Qwen3-1.7B and 4B both **regressed** on held-out AMC after fine-tuning —
and **validation loss completely hides it** (it fell the whole time while accuracy collapsed).
Use the data, but prefer reasoning-distillation / RL over naive SFT (see the repo).

## Composition
| source | n | license |
|---|---|---|
| MATH (Hendrycks), levels 3–5 | 5,397 | MIT |
| NuminaMath-CoT · `synthetic_amc` | 7,533 | Apache-2.0 |
| NuminaMath-CoT · `amc_aime` | 443 | Apache-2.0 |
| **train / validation** | **12,839 / 534** | |

## Decontamination (the careful part)
Every training problem was dropped if it overlapped *any* evaluation problem
(`AI-MO/aimo-validation-amc`, `AI-MO/aimo-validation-aime`, `HuggingFaceH4/MATH-500`) by exact
match, a shared 13-gram, or 8-gram Jaccard ≥ 0.6. This notably removed **most of NuminaMath's
`amc_aime`** — it is 2022–24 AMC/AIME and overlaps the eval years (the decontamination is doing
real work, not theater).

## Format
```json
{"messages": [
  {"role": "system", "content": "You are an expert AMC math tutor. ..."},
  {"role": "user", "content": "<problem>"},
  {"role": "assistant", "content": "<step-by-step solution>\n\nFinal answer: \\boxed{<ans>}"}
]}
```

## Evaluation sets (referenced, not redistributed here)
`AI-MO/aimo-validation-amc` (83), `AI-MO/aimo-validation-aime` (90), `HuggingFaceH4/MATH-500` (500).

## Sources & attribution
- **MATH** — Hendrycks et al. 2021 (MIT), via `EleutherAI/hendrycks_math`.
- **NuminaMath-CoT** — Project Numina (Apache-2.0).
- Base models referenced in the study: **Qwen3** (Apache-2.0).

## Reproduce
`scripts/build_dataset.py --numina` in the companion repo rebuilds this exact set.

## License
Apache-2.0 for the curated dataset (compatible with both source licenses); retain the source
attributions above. Built with AI assistance (Claude Code).
