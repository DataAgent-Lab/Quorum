#!/usr/bin/env python3
"""Decision-model study, block C analysis (`provenance/decision13_protocol.md` §7, amendments 8–9) and the §7 primary
score over blocks E + C.

    python scripts/decision13/analyze_block_c.py      # after analyze_block_e.py -> results/decision13/analysis_block_c.json

Unit: one `labels-consensus` row of the WorkflowEvals harness's scores.json (case x policy). Metric: the workflow's own
(`exact_actions`; `primary_action` for agent_trace_observability). `prediction_error` rows count as wrong (the
harness's rule; includes release-code length refusals). Exact McNemar per workflow x model pair on identical rows,
Holm across the 12 tests of block C. The §7 primary score adds block C's Holm-significant wins - losses to block E's.

Inputs: invoice_processing per-decision rows (results/decision13/workflowevals/invoice_processing_<model>.jsonl.gz:
case id, policy id, label set, status, metric flags; no case text). The other three workflows carry no licence, so
only their counts are published (restricted_aggregates.json) and their tests are recomputed from those counts.
"""
import gzip, json, sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import holm, mcnemar_exact

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "decision13"
MODELS = ["pplx_v11", "clef_flash", "clef"]
WORKFLOWS = {"invoice_processing": "exact_actions", "customer_service": "exact_actions",
             "security_incidents": "exact_actions", "agent_trace_observability": "primary_action"}
EXPOSED_ALL = {"patchnet", "secvuleval"}


def rows(wf, m, label_set="labels-consensus"):
    out, metric = {}, WORKFLOWS[wf]
    for r in map(json.loads, gzip.open(RES / "workflowevals" / f"{wf}_{m}.jsonl.gz", "rt")):
        if r["label_set_id"] != label_set or r["status"] not in ("scored", "prediction_error"):
            continue
        out[(r["case_id"], r["policy_id"])] = (r["status"] == "scored" and r.get(metric) is True, r["status"])
    return out


def per_item_entry(wf, metric, tests):
    R = {m: rows(wf, m) for m in MODELS}
    keys = sorted(R["pplx_v11"])
    assert all(sorted(R[m]) == keys for m in MODELS), f"{wf}: rows differ across models"
    entry = {"metric": metric, "n": len(keys), "models": {}, "context_label_sets": {}, "pairs": {}}
    for m in MODELS:
        st = Counter(R[m][k][1] for k in keys)
        entry["models"][m] = {"accuracy": sum(R[m][k][0] for k in keys) / len(keys), "prediction_error": st["prediction_error"],
                              "scored": st["scored"]}
        entry["context_label_sets"][m] = {ls: sum(v[0] for v in rows(wf, m, ls).values()) / max(1, len(rows(wf, m, ls)))
                                          for ls in ("labels-openai", "labels-anthropic")}
    answered_all = [k for k in keys if all(R[m][k][1] == "scored" for m in MODELS)]
    entry["all_answered"] = {"n": len(answered_all), **{m: (sum(R[m][k][0] for k in answered_all) / len(answered_all)) if answered_all else None for m in MODELS}}
    for x, y in combinations(MODELS, 2):
        b = sum(R[x][k][0] and not R[y][k][0] for k in keys); c = sum(R[y][k][0] and not R[x][k][0] for k in keys)
        p = mcnemar_exact(b, c); entry["pairs"][f"{x}__{y}"] = {"b_x_only": b, "c_y_only": c, "p": p}
        tests.append([wf, x, y, p, x if b > c else y])
    return entry


def restricted_entry(wf, agg, tests):
    entry = {k: agg[k] for k in ("metric", "n", "models", "context_label_sets", "all_answered", "dataset_revision_sha256")}
    entry["per_item_data"] = "no licence; counts only (results/decision13/restricted_aggregates.json)"
    entry["pairs"] = {}
    for x, y in combinations(MODELS, 2):
        b, c = agg["pairs"][f"{x}__{y}"]["b_x_only"], agg["pairs"][f"{x}__{y}"]["c_y_only"]
        p = mcnemar_exact(b, c); entry["pairs"][f"{x}__{y}"] = {"b_x_only": b, "c_y_only": c, "p": p}
        tests.append([wf, x, y, p, x if b > c else y])
    return entry


def score(ts, drop=()):
    sc = {m: 0 for m in MODELS}
    for t in ts:
        if t["set"] in drop or t["p_holm"] >= 0.05:
            continue
        worse = t["b"] if t["better"] == t["a"] else t["a"]; sc[t["better"]] += 1; sc[worse] -= 1
    return sc


def main():
    restricted = json.loads((RES / "restricted_aggregates.json").read_text())["block_c"]
    res, tests = {"workflows": {}}, []
    for wf, metric in WORKFLOWS.items():
        res["workflows"][wf] = (restricted_entry(wf, restricted[wf], tests) if wf in restricted
                                else per_item_entry(wf, metric, tests))
    for t, p_adj in zip(tests, holm([t[3] for t in tests])):
        t.append(p_adj)
    res["holm_accuracy"] = [dict(zip(["set", "a", "b", "p", "better", "p_holm"], t)) for t in tests]
    E = json.loads((RES / "analysis_block_e.json").read_text())
    res["score_block_c"] = score(res["holm_accuracy"])
    res["primary_score_E_plus_C"] = {m: score(E["holm_accuracy"])[m] + res["score_block_c"][m] for m in MODELS}
    res["primary_score_E_plus_C_without_exposed"] = {m: score(E["holm_accuracy"], EXPOSED_ALL)[m] + res["score_block_c"][m] for m in MODELS}
    (RES / "analysis_block_c.json").write_text(json.dumps(res, indent=1))
    for wf, e in res["workflows"].items():
        print(f"{wf:26s} {e['metric']:14s} n={e['n']:4d} " + " ".join(
            f"{m}={e['models'][m]['accuracy'] * 100:.1f}%(err {e['models'][m]['prediction_error']})" for m in MODELS))
    print("block C:", res["score_block_c"], "| E+C:", res["primary_score_E_plus_C"],
          "| E+C without exposed:", res["primary_score_E_plus_C_without_exposed"])


if __name__ == "__main__":
    main()
