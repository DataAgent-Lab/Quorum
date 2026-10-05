#!/usr/bin/env python3
"""R10 — upper reference: Qwen3-4B fine-tuned (LoRA) on Banking77 train only, evaluated once on the full test.

Formulation (approved): the same marker-scoring prompt as the 24-shot reader (`quorum/prompting.py`), with class
names as options and NO in-context examples, so the reference differs from the reader only in training data and
budget. The target is the gold option's marker; the loss is cross-entropy over the K marker logits at the answer
position. Options are shuffled per training example (seeded).

Data: train = pinned Banking77 train minus the clean selection split S (9,079 rows); S (924 rows) is used ONLY for
the early-stopping curve and checkpoint choice; the test split is scored exactly once, with the chosen checkpoint.

    python scripts/r10_finetune.py

Standard LoRA recipe: r=16, alpha=32, dropout 0.05 on q,k,v,o,gate,up,down; lr 2e-4, 3% linear warm-up then linear
decay; 2 epochs; micro-batch 4 x grad-accum 2 (effective 8); bf16; gradient checkpointing; seed 0. S is scored
every half epoch and at the final step with one fixed option order (seed 1000); the checkpoint with the highest S
accuracy is kept (ties: the earlier one). The full curve is recorded in the meta. Writes results/predictions/banking77_finetune_lora.jsonl.gz + _meta.json.
"""
import copy, gzip, hashlib, importlib.util, json, math, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("clean24", ROOT / "scripts" / "clean_24shot.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
from quorum import prompting as P

HP = {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
      "lr": 2e-4, "warmup_frac": 0.03, "epochs": 2, "micro_batch": 4, "grad_accum": 2, "seed": 0,
      "eval_points_per_epoch": 2, "eval_order_seed": 1000, "max_tokens": 4096, "precision": "bf16"}
OUTP = ROOT / "results" / "predictions"


def main():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from quorum.data import _humanize
    random.seed(HP["seed"]); np.random.seed(HP["seed"]); torch.manual_seed(HP["seed"])
    tr = load_dataset(C.DATASET, revision=C.DATASET_REV, split="train")
    names = [_humanize(n) for n in tr.features["label"].names]; K = len(names)
    T = [{"text": r["text"], "label": int(r["label"])} for r in tr]
    S = C.load_split(); Sset = set(S)
    train = [T[i] for i in range(len(T)) if i not in Sset]; val = [T[i] for i in S]
    tok = AutoTokenizer.from_pretrained(C.BASE, revision=C.BASE_REV, padding_side="left")
    markers = P.verify_markers(tok); mids = torch.tensor(P.marker_token_ids(tok, markers)[:K])
    base = AutoModelForCausalLM.from_pretrained(C.BASE, revision=C.BASE_REV, torch_dtype=torch.bfloat16).to("cuda")
    base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=HP["lora_r"], lora_alpha=HP["lora_alpha"], lora_dropout=HP["lora_dropout"],
                                            target_modules=HP["target_modules"], task_type="CAUSAL_LM"))

    def batch_logits(texts, rngs_or_rng, labels=None):
        prompts, orders = [], []
        for t in texts:
            p, o = P.build_prompt(P.Example(C.TASK_DESC, names, t, None, shots=()), markers, shuffle=True, rng=rngs_or_rng)
            prompts.append(p); orders.append(o)
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to("cuda")
        assert enc["input_ids"].shape[1] <= HP["max_tokens"], "prompt exceeds the token bound"
        last = model(**enc).logits[:, -1, :].float()
        return last[:, mids.to(last.device)], orders

    def score(rows):
        """Label-ordered probabilities with one fixed option order (seeded), no gradient."""
        model.eval(); rng = random.Random(HP["eval_order_seed"]); out = np.zeros((len(rows), K))
        with torch.no_grad():
            for i in range(0, len(rows), 8):
                lg, orders = batch_logits([r["text"] for r in rows[i:i + 8]], rng)
                for b, o in enumerate(orders):
                    out[i + b] = P.unpermute(lg[b].cpu(), o).numpy()
        model.train(); z = out - out.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    steps_per_epoch = math.ceil(len(train) / (HP["micro_batch"] * HP["grad_accum"]))
    total = steps_per_epoch * HP["epochs"]; warm = max(1, int(HP["warmup_frac"] * total))
    eval_every = steps_per_epoch // HP["eval_points_per_epoch"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=HP["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    yv = np.array([r["label"] for r in val]); curve, best = [], None
    rng_order, rng_data = random.Random(HP["seed"]), random.Random(HP["seed"] + 1)
    model.train(); step = 0; t0 = time.time()
    for ep in range(HP["epochs"]):
        idx = list(range(len(train))); rng_data.shuffle(idx); mb = HP["micro_batch"]
        for s in range(0, len(idx), mb * HP["grad_accum"]):
            opt.zero_grad(set_to_none=True); loss_sum = 0.0
            for a in range(HP["grad_accum"]):
                chunk = [train[j] for j in idx[s + a * mb: s + (a + 1) * mb]]
                if not chunk: continue
                lg, orders = batch_logits([r["text"] for r in chunk], rng_order)
                tgt = torch.tensor([o.index(r["label"]) for r, o in zip(chunk, orders)], device=lg.device)
                loss = torch.nn.functional.cross_entropy(lg, tgt) / HP["grad_accum"]; loss.backward(); loss_sum += float(loss)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                print(f"step {step}/{total} loss {loss_sum:.4f} {round(time.time() - t0)}s", flush=True)
            if step % eval_every == 0 or step == total:
                pv = score(val); acc = float((pv.argmax(1) == yv).mean())
                nll = float(-np.mean(np.log(pv[np.arange(len(yv)), yv] + 1e-12)))
                curve.append({"step": step, "epoch": round(step / steps_per_epoch, 2), "S_accuracy": round(acc, 4),
                              "S_nll": round(nll, 4), "elapsed_s": round(time.time() - t0)})
                print("EVAL", json.dumps(curve[-1]), flush=True)
                if best is None or acc > best["S_accuracy"]:
                    best = {**curve[-1], "state": {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}}
    train_s = time.time() - t0
    set_peft_model_state_dict(model, best.pop("state"))
    test = [{"text": r["text"], "label": int(r["label"])}
            for r in load_dataset(C.DATASET, revision=C.DATASET_REV, split="test")]            # scored once
    t1 = time.time(); pt = score(test); gold = np.array([r["label"] for r in test]); pred = pt.argmax(1)
    g7 = lambda v: [float(f"{x:.7g}") for x in v]; OUTP.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTP / "banking77_finetune_lora.jsonl.gz", "wt") as f:
        for i, r in enumerate(test):
            f.write(json.dumps({"idx": i, "text": r["text"], "gold": int(gold[i]), "pred": int(pred[i]), "probs": g7(pt[i])}) + "\n")
    hit = pred == gold; res = {"accuracy": round(float(hit.mean()), 4), "correct": int(hit.sum())}
    try:
        import csv, io, urllib.request
        txt = urllib.request.urlopen(C.REPRO_URL, timeout=60).read().decode()
        rep = {int(x["id"].split("-")[1]): x for x in csv.DictReader(io.StringIO(txt))}
        theirs = np.array([rep[i]["correct"] == "True" for i in range(3080)])
        b, c = int((hit & ~theirs).sum()), int((~hit & theirs).sum())
        res["mcnemar_vs_reproduction"] = {"b": b, "c": c, "p_two_sided_exact": C.mcnemar_exact(b, c)}
    except Exception as e:
        res["mcnemar_vs_reproduction"] = {"error": f"{type(e).__name__}: {e}"}
    import torch as _t, transformers, peft
    meta = {"description": "Qwen3-4B + LoRA fine-tuned on Banking77 train minus the clean selection split; "
                           "marker-scoring prompt, class names, no in-context examples",
            "hyper_parameters": HP, "train_rows": len(train), "selection_rows_for_early_stopping": len(val),
            "steps_per_epoch": steps_per_epoch, "total_steps": total, "early_stopping_curve": curve,
            "chosen_checkpoint": best, "result": res, "base": C.BASE, "base_revision": C.BASE_REV,
            "dataset": C.DATASET, "dataset_revision": C.DATASET_REV,
            "gpu_hours": {"train": round(train_s / 3600, 2), "test_scoring": round((time.time() - t1) / 3600, 3)},
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "git_head": C._read_git_head(),
            "versions": {"torch": _t.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
            "device": _t.cuda.get_device_name(0), "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUTP / "banking77_finetune_lora_meta.json").write_text(json.dumps(meta, indent=1))
    print("R10_DONE", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
