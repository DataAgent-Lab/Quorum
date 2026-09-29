# Phase 1.0 — Tasks (Quorum live demo Space)

> Spec: `spec_details.md`. Conversation log: `../../archive/conversation-phase1.0-hf-space-demo.txt`.

## T0 — Fact-first gate
- [x] T0.1 Verify CPU feasibility + latency of `ZeroShotEnsemble` (load 22 s, ~2–5 s small-label, RAM 3.3 GB;
  fits free CPU tier). → conversation DECISION #1.

> **Revised architecture (DECISION #3):** compute (Gradio) Spaces need a paid HF plan → ship a FREE **static**
> Space that calls a **self-hosted** Quorum API. `serve/` (FastAPI) + `space/` (static index.html).

## T1 — Build the service + static frontend
- [x] T1.1 `serve/app.py` — FastAPI wrapper around `quorum.ZeroShotEnsemble`: `POST /predict`, `GET /health`,
  configurable CORS; input guards. `pyproject.toml` `serve` extra (fastapi, uvicorn).
- [x] T1.2 `serve/README.md` — deploy cheat-sheet (install → uvicorn → HTTPS tunnel → CORS → report URL).
- [x] T1.3 `space/index.html` — static frontend: API-URL field (localStorage), message + labels, renders
  distribution + top label + confidence + per-member votes; error handling for API-down / CORS.
- [x] T1.4 `space/README.md` — static Space card (`sdk: static`, links, honest note). Removed the Gradio
  `space/app.py` + `space/requirements.txt` (compute Space — paid).

## T2 — Tests (one per AC)
- [x] T2.1 (AC-F1/F3) core decide() on CPU → distribution sums ~1, "card arrival" top (verified).
- [x] T2.2 (AC-F2) per-member breakdown shows 3 members' picks (verified).
- [x] T2.3 (AC-F4) <2 labels / empty msg / >20 labels guards (verified in decide() test + serve validation).
- [x] T2.4 (AC-P1) 6-label prediction ~2 s warm on CPU (verified).
- [x] T2.5 (AC-I3) internal-trace scan of `serve/` + `space/` = clean.
- [x] T2.6 `serve/app.py` verified via FastAPI TestClient: /health OK, POST /predict 200 -> "card arrival" 0.999 + 3 member picks, <2-labels guard -> 422 (fastapi 0.125).
- [ ] T2.7 (AC-I2) static Space is listed in the Quorum collection (verified post-deploy).

## T3 — Deploy
- [ ] T3.1 Create HF **static** Space `DataAgent/Quorum-Demo` (sdk static, FREE), push `index.html` + card.
- [ ] T3.2 Add the Space to the `Quorum` collection.
- [ ] T3.3 Link the Space from the GitHub README.
- [ ] T3.4 (other machine) run `serve/app.py`, expose HTTPS + CORS, set the Space's default API URL.

## T4 — Docs sync
- [ ] T4.1 Update spec/tasks statuses; append `[STATUS SUMMARY]` to the conversation file.
