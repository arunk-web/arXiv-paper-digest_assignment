# Autonomous arXiv Paper Digest & QA Agent

An agent that takes a research topic or a specific arXiv ID, retrieves and
parses the paper, produces a structured executive briefing, and answers
follow-up questions grounded in the paper's own text (RAG).

## 1. Architecture — Explicit State Graph

This is implemented as a sequence of pure `State -> State` node functions
with explicit conditional edges (see `src/graph.py`), not a single prompt.

```
query_understanding
        |
        v
  arxiv_retrieval  --(0 results)--> HALT, ask user to rephrase
        |
        v
     selection      (topic search: rank candidates by embedding
        |             similarity to query; paper ID: skip straight through)
        v
   fetch_parse      --(download/parse failure)--> HALT
        |           --(scanned/sparse PDF)--> fallback to abstract-only,
        |             continue with a warning attached to state
        v
   chunk_embed      (chunk text, embed with sentence-transformers,
        |            store in a local FAISS index, persisted to disk)
        v
   summarize        (LLM call constrained to JSON schema, grounded in
        |            parsed sections)
        v
  [briefing printed + saved to disk]
        |
        v
     QA loop        (repeatable node, not part of the linear pipeline:
                      embed question -> FAISS top-k -> RAG prompt ->
                      answer constrained to retrieved context)
```

### State shape (`src/state.py`)

A single `AgentState` dataclass carries everything downstream nodes need:

- `raw_query`, `mode`, `parsed_arxiv_id` — query understanding output
- `candidates: List[PaperMeta]`, `selected_paper` — retrieval/selection output
- `pdf_path`, `parsed_sections`, `full_text`, `parse_ok`, `parse_warning` — parsing output
- `chunks: List[Chunk]`, `vectorstore_path` — chunking/embedding output
- `briefing: dict` — the structured executive briefing
- `qa_history: List[QATurn]` — full conversation log for the QA loop
- `errors`, `halt`, `halt_reason` — control-flow / failure signaling

**Persistence across summarize -> QA:** the FAISS index + chunk metadata are
saved to disk (`data/vectorstores/<arxiv_id>.faiss` / `.meta.pkl`) and *also*
kept in-memory on the state object for the current run. This means the QA
loop works whether it's called in the same process right after summarization,
or in a fresh process later by re-loading the index from disk — no re-parsing
or re-embedding needed.

## 2. Failure handling implemented

- **Zero arXiv results** for a topic query -> pipeline halts early with a
  clear message instead of continuing with garbage input.
- **Corrupted / undownloadable PDF** -> halts with a clear reason.
- **Scanned or broken-layout PDF** (detected via average extracted
  characters per page) -> does **not** crash; falls back to briefing from
  the abstract only, flags this in `briefing["parsing_note"]`, and QA
  grounding quality is reduced but the agent stays usable.
- **Huge papers (>80 pages)** -> only the first 40 + last 10 pages are
  parsed, to keep latency/cost bounded, with a note added to the briefing.
- **QA hallucination guard** -> the RAG prompt explicitly instructs the
  model to say `"I couldn't find that in the paper."` when retrieved
  chunks don't answer the question, and the agent tracks `grounded: bool`
  per QA turn.

## 3. Setup & Run (all free tools)

```bash
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set LLM_PROVIDER=groq and GROQ_API_KEY=<your free key from https://console.groq.com/keys>
# (or LLM_PROVIDER=gemini with a free key from https://aistudio.google.com/app/apikey)
```

Run:

```bash
python main.py "recent work on KV-cache compression for LLMs"
# or
python main.py 2401.12345
# or
python main.py https://arxiv.org/abs/2401.12345
```

**Free-tier rate limits to be aware of when testing:** Groq's free tier
(default model here: `openai/gpt-oss-20b`) allows roughly 30 requests/min
and ~14,400/day at time of writing. Groq periodically retires/renames
models — if you get a `model_not_found` error, run
`curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`
to see which model IDs are currently active on your account and update
`GROQ_MODEL` in `.env` accordingly — more than enough for this pipeline (1 summarize call +
1 call per QA question). Gemini's free tier (`gemini-1.5-flash`) is
similarly generous. No paid key is required for either path. The
`sentence-transformers` embedding model runs 100% locally (no API calls,
no key needed) after its first download (~80MB).

## 4. Example run

```
$ python main.py "KV-cache compression for LLMs"
[graph] running node: query_understanding
[graph] running node: arxiv_retrieval
[graph] running node: selection
[graph] running node: fetch_parse
[graph] running node: chunk_embed
[graph] running node: summarize

======================================================================
TITLE:    EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving
AUTHORS:  Shaoting Feng, Yuhan Liu, Hanchen Li, Xiaokun Chen, Samuel Shen, Kuntai Du,
          Zhuohan Gu, Rui Zhang, Yuyang Huang, Yihua Cheng, Jiayi Yao, Qizheng Zhang,
          Ganesh Ananthanarayanan, Junchen Jiang
ARXIV ID: 2512.14946v1   (2025-12-16)
LINK:     https://arxiv.org/abs/2512.14946v1
----------------------------------------------------------------------
WHY IT MATTERS:
EVICPRESS jointly compresses and evicts entries from the LLM KV cache
instead of treating compression and eviction as separate mechanisms,
which lets it keep more useful context in a fixed memory budget while
serving LLM requests faster.

PROBLEM STATEMENT:
Serving LLMs with long contexts is bottlenecked by the size of the
KV cache; existing systems either only compress or only evict cache
entries, leaving efficiency gains on the table.

METHOD:
  - Profiles context sensitivity to decide which parts of the KV cache
    can be safely compressed vs. must be preserved.
  - Applies a joint compression + eviction policy (using methods such as
    SnapKV, KeyDiff, KVzip, KNorm) instead of choosing one mechanism only.
  - Uses a heuristic scheduler to balance cache-hit rate against
    generation-quality risk per request.

KEY RESULTS:
  - Achieves up to 2.19x faster time-to-first-token (TTFT) at equivalent
    generation quality.
  - Higher KV-cache hit rates on fast devices compared to baselines that
    only compress or only evict.
  - Preserves high generation quality by conservatively compressing
    contexts sensitive to errors.
  - Evaluated on 12 datasets and 5 models, demonstrating consistent
    improvements.

LIMITATIONS:
  - Not stated in the paper.

SUGGESTED FOLLOW-UP QUESTIONS:
  - How does EVICPRESS adapt to changes in context sensitivity over time?
  - Which specific lossy compression algorithms and rates are supported
    by the system?
  - What is the computational overhead of the profiling and heuristic
    phases, especially for large numbers of concurrent contexts?
======================================================================
[saved] briefing -> ./data/briefings/2512.14946v1.json

Ask questions about this paper (blank line to quit):
Q> What compression method does EVICPRESS use?
A> EVICPRESS employs a variety of compression methods, including SnapKV,
   KeyDiff, KVzip, and KNorm (from the distribution figure in the
   evaluation section).
   (sources: background, evaluation, introduction, method)
Q> Does the paper report results on multi-GPU inference?
A> I couldn't find that in the paper.
   (sources: introduction, method, related work)
Q>
```

Note the second QA example: the agent correctly refuses to answer instead of
guessing when the retrieved chunks don't contain the answer — this is the
grounding guard from §2 working as intended, not a bug.

## 5. Design Decisions & Tradeoffs

**Custom Python state machine instead of LangGraph/CrewAI.** For a 7-node
mostly-linear pipeline with one branch point, a hand-rolled state machine
is more transparent to review than framework machinery, and keeps the
dependency footprint (and thus setup friction) minimal. With more time I
would port this to LangGraph to get built-in checkpointing/resume and
visual graph export for free.

**Local embeddings (sentence-transformers) + local FAISS** instead of a
hosted vector DB or hosted embedding API — keeps the whole retrieval path
free, offline-capable, and fast to set up, at the cost of slightly lower
embedding quality than e.g. OpenAI's embedding models.

**Section-header regex chunking** rather than a layout-aware parser
(unstructured.io, Grobid). This is simpler and dependency-light but is
sensitive to PDF sections that use unconventional headers or two-column
layouts that PyMuPDF sometimes reads out of order. Chunking then also
applies a fixed word-window (220 words, 40-word overlap) within each
detected section, which is a reasonable default for RAG but not tuned per
paper.

**Auto-pick top candidate on topic search** rather than presenting a list
and asking the user to choose. This matches "no UI beyond CLI" scope; the
tradeoff is the agent commits to one interpretation of an ambiguous topic.
`state.candidates` still holds the full ranked list, so extending this to
an interactive picker is a small change to `main.py`.

**Grounding strategy:** the QA prompt restricts the model to only the
top-k retrieved chunks and explicitly instructs it to refuse rather than
guess. This is a prompt-level guardrail, not a hard constraint — a more
robust version (given more time) would add a second LLM call that verifies
each answer sentence is entailed by the retrieved context before returning
it to the user (a lightweight NLI-based grounding check).

**What I'd do differently / next with more time:**
- Add a verification/self-check pass on the summarize node output (does
  each bullet actually appear/traceable in the source text?).
- Table/figure-aware parsing — currently tables and captions get flattened
  into body text.
- A small eval set of (paper, question, expected-answer) pairs to
  regression-test the RAG pipeline instead of eyeballing outputs.
- Persist `qa_history` to disk per paper so a QA session can resume across
  separate CLI invocations, not just within one process.

## 6. Known Limitations

- Section detection is heuristic; papers with non-standard section naming
  may be split imperfectly.
- No OCR fallback for scanned PDFs — they degrade to abstract-only mode.
- Only English-language section headers are recognized.
- Single-paper QA only; no cross-paper comparison in this scope.

## 7. Out of scope (per assessment instructions)

No frontend beyond this CLI, no auth/deployment, no non-arXiv sources, no
fine-tuning — all intentionally omitted.
