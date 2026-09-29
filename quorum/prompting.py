"""Single-forward option scoring for the 24-shot reader ("answer-token" / M1 scoring).

Every candidate label gets a single-token letter marker (A–Z, a–z, then AA–ZZ). The prompt lays out the task,
the markered options, the retrieved in-context examples, and the query; the model's *next-token* logits are
then read at exactly those marker tokens — giving a K-way distribution over the labels from ONE forward pass.
Options are shuffled per query so a marker carries no label information; `unpermute` maps positions back to
label ids.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Sequence

SYSTEM = ("You are a decision model. Read the task, the options and the message, then answer with the single "
          "option marker that fits best. Answer with the marker only.")
THINK_OFF = "<think>\n\n</think>\n\n"

_U = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_L = [chr(c) for c in range(ord("a"), ord("z") + 1)]
MARKER_POOL = _U + _L + [a + b for a in _U for b in _U]      # single-token letters, verified per tokenizer


@dataclass
class Example:
    task_desc: str
    options: Sequence[str]                                    # verbalized label per class, index == label id
    text: str
    label_idx: int | None = None
    shots: Sequence[tuple[str, int]] = field(default_factory=tuple)   # (text, label_idx) retrieved examples


def verify_markers(tok, pool: Sequence[str] = MARKER_POOL, prefix: str = " ") -> list[str]:
    """Markers that tokenize to exactly one token in the leading-space form the model emits after 'Answer:'."""
    ok = []
    for m in pool:
        if len(tok(prefix + m, add_special_tokens=False).input_ids) == 1:
            ok.append(m)
    return ok


def marker_token_ids(tok, markers: Sequence[str], prefix: str = " ") -> list[int]:
    ids = []
    for m in markers:
        t = tok(prefix + m, add_special_tokens=False).input_ids
        assert len(t) == 1, f"marker {m!r} is not a single token for this tokenizer"
        ids.append(t[0])
    return ids


def build_prompt(ex: Example, markers: Sequence[str], shuffle: bool = True,
                 rng: random.Random | None = None) -> tuple[str, list[int]]:
    """Return (prompt, order) where order[j] is the original label id shown at position j (marker markers[j])."""
    K = len(ex.options)
    if K > len(markers):
        raise ValueError(f"K={K} options but only {len(markers)} single-token markers available")
    order = list(range(K))
    if shuffle:
        (rng or random).shuffle(order)
    lines = [f"Task: {ex.task_desc}", "Options:"]
    for j, oi in enumerate(order):
        lines.append(f"{markers[j]}) {ex.options[oi]}")
    if ex.shots:
        lines.append("Examples:")
        pos_of = {oi: j for j, oi in enumerate(order)}
        for t, li in ex.shots:
            lines.append(f"Message: {t}\nAnswer: {markers[pos_of[li]]}")
    lines.append(f"Message: {ex.text}")
    user = "\n".join(lines)
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
              f"<|im_start|>assistant\n{THINK_OFF}Answer:")
    return prompt, order


def unpermute(logits_by_pos, order: Sequence[int]):
    """logits_by_pos[j] scored the option shown at position j; return logits indexed by label id."""
    import torch
    inv = torch.empty(len(order), dtype=torch.long)
    for j, oi in enumerate(order):
        inv[oi] = j
    return logits_by_pos[inv]
