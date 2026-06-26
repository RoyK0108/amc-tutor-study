"""Smoke test: load a small Qwen3 MLX model and solve one AMC problem.
Confirms MLX works on this Mac, shows tokens/sec, and checks chat-template +
(disabled) thinking behavior."""
import sys, time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from common import SYSTEM_PROMPT, extract_boxed

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-1.7B-4bit"
print("loading:", MODEL, flush=True)
t0 = time.time()
model, tok = load(MODEL)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

problem = (r"$\frac{m}{n}$ is the Irreducible fraction value of "
           r"\[3+\frac{1}{3+\frac{1}{3+\frac13}}\], what is the value of $m+n$?")
messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}]
try:
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
except TypeError:
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     tokenize=False)

t1 = time.time()
out = generate(model, tok, prompt=prompt, max_tokens=600,
               sampler=make_sampler(temp=0.0), verbose=True)
print("\n--- parsed ---")
print("extracted boxed:", extract_boxed(out), "| gold: 142")
print(f"gen wall time: {time.time()-t1:.1f}s")
