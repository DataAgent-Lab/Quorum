# Decision memo — AI use in writing the TACL paper (Phase 2.0)

> 2026-10-06. Policy text quoted verbatim from the primary sources (fetched today). Hours are estimates for a
> 10-page TACL paper whose evidence is already complete (claims-ledger.md). Decision owner: the author.

## The binding rule (ACL Policy on Publication Ethics → "Guidelines for Generative Assistance in Authorship")

- "Any use of generative AI tools and technologies to create content, whether to support research or in review
  rebuttals, should be appropriate and should be fully disclosed in the Acknowledgements section." · "Any use not
  listed below is considered inappropriate use." · "Authors are responsible for all content submitted."
- Allowed list (verbatim, abridged to the operative sentence):
  - **(a)** "Assistance purely with the language of the paper. This covers models used for paraphrasing or polishing the
    author's original content, rather than for suggesting new content…" (grammar/spell checkers need no disclosure)
  - **(b)** predictive keyboards (no disclosure)
  - **(c)** "Literature search… The usual requirements for citation accuracy… apply."
  - **(d)** "Low-novelty text. This covers the automatic generation of text about pre-existing ideas. Authors should
    specify where such automatically generated text was used, and convince the reviewers that the generation was
    checked to be accurate and is accompanied by relevant and appropriate citations."
  - **(e)** "New ideas… generative model output [that] reads to the authors as new research ideas… which the authors
    then develop themselves… The authors should disclose if models were used in this manner."
  - **(f)** "New ideas + new text: ACL does not consider a generative model to be an entity that can fulfill the
    requirements of co-authorship." (i.e. not an allowed use)
- **Translation is not named** anywhere in the list. **TACL's submission page does not mention AI at all**; the ACL
  policy is the operative rule for ACL publications, but the gap is a reason to ask the editor (option G).

## What this means for this paper, plainly

- The **results, numbers and protocols** are the author's research, run with AI coding agents under the author's
  direction → must be disclosed ("to support research … fully disclosed"); the label descriptions written by an LLM
  must also be disclosed in the method section (claims-ledger P4).
- The **core text** (contributions, method rationale, interpretation of results, discussion, limitations) is *new
  content*. An LLM writing it from scratch is (f) → **not allowed at TACL even with disclosure**.
- An LLM **may**: polish/translate the author's own text (a, translation by interpretation), draft text about
  pre-existing ideas with verified citations (d: related work, standard definitions such as BM25/McNemar/ECE, dataset
  descriptions), help search literature (c), and suggest ideas the author then develops (e).

## Options

| | Workflow | Policy basis | Author hours (est.) | Risk |
|---|---|---|---|---|
| **A** | **Dictation:** the author explains each core section (in Chinese is fine, 15–20 min each); the LLM only transcribes, translates sentence-faithfully and fixes language; the author fixes the structure first (outline approval) and edits/approves every paragraph | (a) — "polishing the author's original content"; translation is an interpretation of (a), so disclose it explicitly | 6 core sections × ~20 min dictation (2 h) + editing/approving (5 h) + full read (2 h) ≈ **9 h** | Low–medium: the "translation" reading is not explicit; keep the LLM from adding claims or structure |
| **B** | **Split by novelty:** the LLM drafts only pre-existing-idea text with verified citations (related work, background, metric/dataset definitions, ≈30–35% of the paper), disclosed by section; the author writes the core (~6 pages) | (d) for the drafted parts; (a) for polishing the author's core | writing 6 pages directly ≈ **18–25 h** (non-native, from scratch); ≈ **10 h** combined with A | Low |
| **A+B** (recommended) | A for the core, B for the low-novelty parts, author reviews everything; Acknowledgements lists exactly which sections were LLM-drafted (d) and that the core was dictated by the author and translated/polished by an LLM (a) | (a) + (d), both disclosed | **≈ 10–12 h** over 1–2 weeks | Low |
| **C** | Add a human co-author who writes the core, with authorship credit | standard authorship | author: review only (~4 h); co-author: ~20 h | Low policy risk; recruiting/credit logistics |
| **D** | Change venue to one that allows LLM-written text with responsibility: **TMLR** — "LLMs may be used as general-purpose assistive tools. Whichever tools are used, authors are fully responsible for content… LLMs are not eligible for authorship." (low-quality, largely LLM-generated submissions face misconduct scrutiny) · NeurIPS 2025 — "We welcome authors to use any tool that is suitable for preparing high-quality papers… authors are responsible for the entire content" | TMLR / NeurIPS policy | review + editing ≈ **6 h** | Recognition drops from Tier 1 (TACL) to Tier 2 (TMLR); NeurIPS main is a poor fit/novelty bar |
| **G** | Before choosing, **email the TACL editor** describing the exact workflow (AI-run experiments under the author's direction; A+B writing) and ask whether it is acceptable | removes the ambiguity (translation; AI-run research) | ~1 h | None; editors already know the authors, so blind review is unaffected |

## Not acceptable under any option

- LLM-written core text submitted to TACL — disclosed or not ((f); "any use not listed … inappropriate").
- Undisclosed AI use of any kind; AI listed as an author.
- "Humanizer" / paraphrase-to-evade-detection tools (misrepresent authorship; not an allowed use).
- Hidden text or instructions aimed at reviewers or LLM reviewers.
- Any reference not verified by hand + `paper/tools/checks.py bib` + the title/author check (fabricated or mismatched
  citations are the main cause of real desk rejections).

## Recommendation

**G first (1 h), then A+B at TACL (≈10–12 author hours).** If the author cannot commit ~10 hours, choose **D (TMLR)**
with full disclosure rather than bending TACL's rule. In every option: keep an AI-use log (which tool, which section,
what it did), and the Acknowledgements text is drafted from that log.
