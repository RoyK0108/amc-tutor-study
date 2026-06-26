"""Shared utilities for the AMC tutor project.

Imported by both the dataset builder and the evaluator so that the system
prompt, answer-extraction, and text-normalization logic are *identical* across
training and evaluation (a common source of silent eval bugs).
"""
import re

# The tutor persona. Used for BOTH training targets and eval prompts so the
# model sees the same instruction at train and test time.
SYSTEM_PROMPT = (
    "You are an expert AMC (American Mathematics Competitions) math tutor. "
    "Solve the problem with clear, step-by-step reasoning that a student can "
    "learn from: state the key idea, show each step and why it works, then give "
    "the final result on its own line as 'Final answer: \\boxed{...}'."
)


def extract_boxed(text: str):
    """Return the contents of the LAST \\boxed{...} in `text`, handling nested
    braces. Returns None if there is no well-formed \\boxed{}."""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        if text[i].isspace():
            i += 1
        else:
            return None  # something like \boxedX -> malformed
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j].strip()
    return None  # unbalanced braces


_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9\\]+")


def normalize_text(text: str) -> str:
    """Lowercase, drop $ delimiters, collapse whitespace. Used for dedup and
    contamination matching (not for training text)."""
    if not text:
        return ""
    t = text.lower().replace("$", " ")
    t = _NONWORD.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def word_ngrams(norm_text: str, n: int):
    """Yield word n-grams (as tuples) from already-normalized text. If the text
    has fewer than n words, yield the whole word tuple once so short problems
    still get a fingerprint."""
    words = norm_text.split()
    if len(words) < n:
        if words:
            yield tuple(words)
        return
    for i in range(len(words) - n + 1):
        yield tuple(words[i:i + n])


def normalize_answer(ans) -> str:
    """Light normalization of a gold answer for storage: turn 142.0 -> '142'."""
    if ans is None:
        return ""
    s = str(ans).strip()
    # collapse integral floats: "142.0" -> "142"
    m = re.fullmatch(r"(-?\d+)\.0+", s)
    if m:
        return m.group(1)
    return s
