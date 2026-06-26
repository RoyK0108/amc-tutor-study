# CLAUDE.md — repo guide

AMC 10/12 competition-math tutor **fine-tuning study**. Built locally on a MacBook Air M4
with MLX, at $0. The headline is a **negative result** (see `README.md`).

## Layout
- `scripts/build_dataset.py` — build + **decontaminate** train/eval (`--numina` adds NuminaMath-CoT).
- `scripts/common.py` — shared `SYSTEM_PROMPT`, `extract_boxed`, n-gram decontam helpers (used by build + eval).
- `eval/evaluate.py` — pass@1 eval: last `\boxed{}` vs gold via `math_verify`, + bootstrap 95% CI. `--adapter` for fine-tuned.
- `scripts/diag_ft.py` — per-problem base-vs-fine-tuned diagnostic (**how the regression was caught**).
- `scripts/report.py`, `compare.py`, `make_plot.py` — aggregation, side-by-side, figure (`assets/`).
- `configs/*.yaml` — MLX LoRA configs (`1.7b`, `1.7b_v2`, `4b`).
- `notebooks/self_distillation_colab.ipynb` — **Phase 2**: real-GPU STaR fix (the path to an *improved* model).
- `hf_dataset/` — Hugging Face dataset card; `scripts/upload_hf.py` publishes it.
- `data/processed/`, `eval/` — data; `results/` — eval JSONs + logs.

## Env
```bash
uv venv .venv --python 3.12 && uv pip install -r requirements.txt
```

## Key methodological point
The eval verdict is **generated-answer accuracy vs. the base on identical items** — NOT validation
loss (which fell the whole time while accuracy collapsed). Always compare per-problem against base.

## Ops gotchas
Fanless M4 thermal-throttles under sustained load; **closing the lid sleeps the Mac and hangs
in-flight GPU jobs** — keep the lid open or `sudo pmset -a disablesleep 1`. Recover hung training
from the last `save_every` checkpoint.

Built with AI assistance (Claude Code).
