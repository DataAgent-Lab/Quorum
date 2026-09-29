"""Quorum live demo — a jury of tiny open models decides an intent and tells you how sure it is.

Type a message and a few candidate intent labels; the zero-shot ensemble (PrismNLI-0.4B + bge-large +
bge-base, geometric mean, no training, no calibration) returns a calibrated distribution + each member's vote.
Runs on CPU. The 24-shot pipeline that beats Jev needs the gated adapter — see the repo.
"""
import gradio as gr
from quorum import ZeroShotEnsemble

MAX_LABELS = 20
CLF = ZeroShotEnsemble(device="cpu")           # loads the three models once at startup (~20 s)
MEMBER_NAMES = ["PrismNLI-0.4B (NLI)", "bge-large (cosine)", "bge-base (cosine)"]

DEFAULT_MSG = "when will my new card arrive?"
DEFAULT_LABELS = "\n".join([
    "card arrival", "card delivery estimate", "lost or stolen card",
    "change pin", "top up by card", "exchange rate",
])


def decide(message: str, labels_text: str):
    message = (message or "").strip()
    labels = []
    for line in (labels_text or "").splitlines():
        s = line.strip()
        if s and s not in labels:
            labels.append(s)
    if not message:
        raise gr.Error("Enter a message.")
    if len(labels) < 2:
        raise gr.Error("Enter at least 2 candidate labels (one per line).")
    if len(labels) > MAX_LABELS:
        raise gr.Error(f"Please keep it to {MAX_LABELS} labels or fewer for this CPU demo "
                       f"(you gave {len(labels)}).")
    members, ens = CLF.member_and_ensemble_proba(message, list(range(len(labels))), label_texts=labels)
    dist = {labels[i]: float(ens[i]) for i in range(len(labels))}
    breakdown = "\n".join(f"- **{MEMBER_NAMES[m]}** → {labels[int(members[m].argmax())]}"
                          for m in range(len(members)))
    top = max(dist, key=dist.get)
    summary = f"### {top}\ncalibrated confidence **{dist[top]:.1%}**\n\n**How the jury voted**\n{breakdown}"
    return dist, summary


with gr.Blocks(title="Quorum") as demo:
    gr.Markdown(
        "# Quorum — a jury of tiny open models\n"
        "Type a message and a few candidate intent labels. A **zero-shot** ensemble of three sub-1B open "
        "models (PrismNLI-0.4B + bge-large + bge-base, geometric mean) picks the intent and tells you how "
        "sure it is — **no training, no calibration**, on CPU.\n\n"
        "*This is the zero-shot tier. The 24-shot pipeline that beats the closed API Jev (0.932 vs 0.924) "
        "uses a gated adapter — see the "
        "[repo](https://github.com/DataAgent-Lab/Quorum) and the "
        "[Quorum collection](https://huggingface.co/collections/DataAgent/quorum-6abb3255e0fab20d66add39b).*"
    )
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(label="Message", value=DEFAULT_MSG, lines=2)
            labs = gr.Textbox(label="Candidate labels (one per line, 2–20)", value=DEFAULT_LABELS, lines=8)
            btn = gr.Button("Decide", variant="primary")
        with gr.Column():
            out_label = gr.Label(label="Distribution", num_top_classes=10)
            out_md = gr.Markdown()
    btn.click(decide, [msg, labs], [out_label, out_md])
    gr.Examples(
        [["my card was stolen last night", DEFAULT_LABELS],
         ["what exchange rate did I get?", DEFAULT_LABELS],
         ["I need to reset my PIN", DEFAULT_LABELS]],
        [msg, labs],
    )

if __name__ == "__main__":
    demo.launch()
