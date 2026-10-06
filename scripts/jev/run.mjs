// Jev retrieval-swap run — implements provenance/jev_retrieval_swap_protocol.md exactly (pre-registered).
//   cd scripts/jev && npm ci && node run.mjs --cache <dir> [--limit N]
// Reads JEV_API_KEY from service/.env (never printed). Writes results/jev/{armA,armB}.jsonl + ledger.jsonl.
import { readFileSync, writeFileSync, existsSync, mkdirSync, appendFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { experimental_evaluate as evaluate, createGateway } from "ai";

const ROOT = new URL("../../", import.meta.url).pathname;
const REPRO = "https://raw.githubusercontent.com/simonmesmith/jev-banking77-experiment/5cac4ff/runs/test-v1";
const BYTE_LIMIT = 30000, CONCURRENCY = 4, ATTEMPTS = 3, CAP_USD = 5.0, RATE = 0.042 / 1e6;
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const CACHE = arg("--cache"); const LIMIT = Number(arg("--limit", 3080));
if (!CACHE) throw new Error("--cache <dir> required (reproduction files are cached outside the repo)");

const env = Object.fromEntries(readFileSync(ROOT + "service/.env", "utf8").split("\n").filter(l => l.includes("="))
  .map(l => [l.slice(0, l.indexOf("=")).trim(), l.slice(l.indexOf("=") + 1).trim().replace(/^["']|["']$/g, "")]));
if (!env.JEV_API_KEY) throw new Error("JEV_API_KEY missing in service/.env");
const model = createGateway({ apiKey: env.JEV_API_KEY }).evaluationModel("typesafe-ai/jev");
const redact = s => String(s).split(env.JEV_API_KEY).join("[REDACTED]");

// Pinned train texts/labels and the clean run's retrieved indices (both from this repository).
const train = JSON.parse(readFileSync(ROOT + "docs/phases/2.0/jev/train_pinned.json", "utf8"));   // moved 2026-10-06   // {text:[], label_name:[]}
const clean = gunzipSync(readFileSync(ROOT + "results/predictions/banking77_24shot_clean.jsonl.gz")).toString()
  .trim().split("\n").map(l => JSON.parse(l));

const OUT = ROOT + "results/jev/"; mkdirSync(OUT, { recursive: true }); mkdirSync(CACHE, { recursive: true });
const done = { A: new Set(), B: new Set() };
for (const arm of ["A", "B"]) if (existsSync(OUT + `arm${arm}.jsonl`))
  for (const l of readFileSync(OUT + `arm${arm}.jsonl`, "utf8").trim().split("\n").filter(Boolean)) done[arm].add(JSON.parse(l).idx);
let spent = 0;
if (existsSync(OUT + "ledger.jsonl")) for (const l of readFileSync(OUT + "ledger.jsonl", "utf8").trim().split("\n").filter(Boolean))
  spent += JSON.parse(l).cost_usd || 0;

async function request(i) {
  const f = `${CACHE}/request-${String(i).padStart(5, "0")}.json`;
  if (!existsSync(f)) {
    const r = await fetch(`${REPRO}/test-${String(i).padStart(5, "0")}/request.json`);
    if (!r.ok) throw new Error(`reproduction request ${i}: HTTP ${r.status}`);
    writeFileSync(f, await r.text());
  }
  return JSON.parse(readFileSync(f, "utf8"));
}

function swap(req, i) {
  const out = structuredClone(req);
  let ex = clean[i].retrieved_idx.map(j => ({ message: train.text[j], intent: train.label_name[j] }));
  out.state.labeled_examples = ex;
  while (Buffer.byteLength(JSON.stringify(out)) > BYTE_LIMIT && ex.length) { ex = ex.slice(0, -1); out.state.labeled_examples = ex; }
  return { payload: out, kept: ex.length };
}

async function call(arm, i, payload, extra) {
  for (let a = 1; a <= ATTEMPTS; a++) {
    if (spent + 0.01 > CAP_USD) throw new Error("budget cap reached");
    const t0 = Date.now();
    try {
      const r = await evaluate({ model, state: payload.state, questions: payload.questions, maxRetries: 0 });
      const ans = r.answers.intent, toks = r.usage?.inputTokens ?? r.usage?.input_tokens ?? 0, cost = toks * RATE;
      spent += cost;
      appendFileSync(OUT + "ledger.jsonl", JSON.stringify({ arm, idx: i, attempt: a, ok: true, ms: Date.now() - t0, input_tokens: toks, cost_usd: cost,
        model: r.response?.modelId ?? null }) + "\n");
      return { idx: i, arm, choice: ans.choice, confidence: ans.confidence ?? null, probabilities: ans.probabilities ?? null,
        model: r.response?.modelId ?? null, input_tokens: toks, attempts: a, ...extra };
    } catch (e) {
      const status = e.statusCode ?? e.cause?.statusCode ?? null;
      appendFileSync(OUT + "ledger.jsonl", JSON.stringify({ arm, idx: i, attempt: a, ok: false, status, error: redact(e.message).slice(0, 300) }) + "\n");
      const retryable = status === null || [429, 500, 502, 503, 504, 529].includes(status);
      if (!retryable || a === ATTEMPTS) return { idx: i, arm, failed: true, status, ...extra };
      await new Promise(s => setTimeout(s, 2000 * 2 ** (a - 1)));
    }
  }
}

const queue = Array.from({ length: LIMIT }, (_, i) => i);
async function worker() {
  while (queue.length) {
    const i = queue.shift(); const req = await request(i);
    if (!done.A.has(i)) appendFileSync(OUT + "armA.jsonl", JSON.stringify(await call("A", i, req, { kept: req.state.labeled_examples?.length ?? 0 })) + "\n");
    if (!done.B.has(i)) { const { payload, kept } = swap(req, i); appendFileSync(OUT + "armB.jsonl", JSON.stringify(await call("B", i, payload, { kept })) + "\n"); }
  }
}
await Promise.all(Array.from({ length: CONCURRENCY }, worker));
console.log(JSON.stringify({ done: LIMIT, spent_usd: Number(spent.toFixed(4)) }));
