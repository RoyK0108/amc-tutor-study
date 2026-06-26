"""Diagnose the fine-tuned 4B regression: generate on an easy AMC problem under
both thinking modes and inspect output + prompt formatting."""
import sys, json
sys.path.insert(0, '/Users/YNA/amc-tutor/scripts')
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from common import SYSTEM_PROMPT, extract_boxed

model, tok = load("mlx-community/Qwen3-4B-4bit",
                  adapter_path="/Users/YNA/amc-tutor/adapters_4b")
rows = [json.loads(l) for l in open('/Users/YNA/amc-tutor/eval/amc.jsonl')]
r = rows[0]  # continued fraction, gold 142 (base solved it)
msgs = [{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": r['problem']}]
for think in [False, True]:
    p = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                tokenize=False, enable_thinking=think)
    out = generate(model, tok, prompt=p, max_tokens=1024,
                   sampler=make_sampler(temp=0.0), verbose=False)
    print("=" * 72)
    print(f"enable_thinking={think} | gold={r['answer']} | boxed={extract_boxed(out)} | gen_len={len(out)}")
    print("PROMPT tail:", repr(p[-200:]))
    print("--- GEN first 800 ---")
    print(out[:800])
    print("--- GEN last 200 ---")
    print(out[-200:])
