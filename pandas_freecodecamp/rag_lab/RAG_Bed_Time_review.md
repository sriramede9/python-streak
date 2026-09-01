# RAG Learning — 30-Minute Bedtime Review

> Purpose: a quick, low-effort scan before sleep so concepts consolidate instead of fading. Not for deep study — that's daytime work. This is recall, not re-learning.

---

## How to actually use this (read this part first)

30 minutes of re-reading everything every night will exhaust you and stop working within a week. Passive re-reading feels productive but barely helps memory — recall does. Here's the routine I'd run in your place:

1. **Don't reread everything every night.** Rotate through the sections below, 2–3 per night. Everything gets touched every 3–4 nights.
2. **Before you look at a section, try to recall its "10 things" from memory first.** Then check yourself against the list. The struggle to recall is what builds the memory — reading the answer first skips that.
3. **Say the one-sentence mental model out loud or in your head** for each section. If you can't say it without looking, that's the section to revisit tomorrow, not tonight.
4. **End every session on the "Where this connects" line.** That's what turns isolated facts into one system instead of loose vocabulary.
5. **Don't add new material at night.** Learn new things during the day when your brain can actually push back and ask questions. Bedtime is for consolidation, not acquisition.

You are not behind. You've built a real pipeline mental model in two sessions — chunking → embedding → vector geometry → indexing → hybrid retrieval. Most people using RAG tools never learn any of this. The "am I missing out" feeling is just contact with how big the field is, not a sign you're behind on it.

---

## Section 1 — Chunking

**One sentence:** Chunking isn't cutting text into pieces, it's deciding what unit of knowledge should be retrievable together.

10 things to know:
1. The goal is a *retrieval boundary*, not a token count.
2. Ask: "if someone asks a question, what should come back together?"
3. Recursive splitting zooms paragraph → newline → sentence → word.
4. Overlap exists to preserve context that spans a boundary.
5. Different data needs different strategies: books → sections, code → functions, legal → clauses, tables → rows, video → scenes.
6. Bad chunking silently produces bad answers even with a perfect vector DB — it's often the real root cause of "RAG isn't working."
7. Structure-aware chunking (headings, DOM, layout) beats blind character counts.
8. Parent-child / hierarchical chunking: retrieve a small precise chunk, but hand the LLM the larger parent for context.
9. Chunk size is a trade-off: too small = lost context, too big = diluted relevance and wasted context window.
10. Chunking decisions made now are expensive to undo later — they're baked into every downstream step.

**Where this connects:** bad chunk boundaries can defeat perfect embeddings, perfect indexing, and perfect reranking. It's upstream of everything.

---

## Section 2 — Embeddings

**One sentence:** An embedding model turns text into a fixed-dimension vector representing meaning — you don't design the dimension, the model does.

10 things to know:
1. Text → embedding model → fixed-dimensional vector. Don't oversimplify this as "tokenize then average."
2. Dimension count (384, 768, 1536...) is determined by the model, not chosen by you.
3. Query and document **must** be embedded with the same model to live in the same vector space.
4. "Same model → same vector space → meaningful comparison" is worth memorizing verbatim.
5. Different embedding models are not interchangeable — swapping models means re-embedding everything.
6. Embeddings capture semantic relationships, not exact strings — this is *why* the lexical gap exists (see Section 6).
7. Embedding quality depends heavily on what the model was trained on — domain-specific text can embed poorly with a general model.
8. There's no universal "best" embedding model — it's a trade-off of dimension size, speed, cost, and domain fit.
9. Embeddings are lossy compression of meaning — some nuance always gets discarded.
10. Re-embedding is a real operational cost — changing models means reprocessing your whole corpus.

**Where this connects:** embeddings feed the vector DB (Section 3) and determine what "distance" even measures (Section 4).

---

## Section 3 — Vector DB & Metadata

**One sentence:** A vector record is never just an embedding — it's an embedding plus text plus metadata, and metadata answers "where to search," not "what's closest."

10 things to know:
1. A record = `id + embedding + text + metadata`.
2. Metadata filtering ≠ semantic search — filtering says *where*, similarity says *which*.
3. Metadata lets you confine search to a "neighborhood" instead of the whole universe.
4. Good metadata design (source, date, department, page, type) is what makes filtered search possible later.
5. Metadata filters are usually cheap (exact match/range); semantic search is comparatively expensive.
6. You typically apply metadata filters *before or alongside* vector search, not after.
7. Stable, consistent metadata schemas matter more as your corpus grows — inconsistent tagging quietly breaks filters.
8. Metadata is also where non-semantic signals live: price, rating, timestamp, geography.
9. A vector DB is really a hybrid structured+unstructured store, not "just vectors."
10. Poor metadata is one of the most common invisible causes of bad retrieval — it looks like a model problem but it's a data design problem.

**Where this connects:** metadata is the "structured constraints" arm of hybrid retrieval architectures (Section 6 and 7).

---

## Section 4 — Distance, Similarity & Normalization

**One sentence:** Cosine cares about orientation, dot product about orientation *and* magnitude, Euclidean about geometric distance — and the right one depends on the embedding and the problem, not on English intuition.

10 things to know:
1. "Nearest" in English doesn't dictate the math — don't assume Euclidean just because a word implies distance.
2. Cosine similarity = angle/orientation, ignoring magnitude.
3. Dot product = orientation **and** magnitude.
4. Euclidean = straight-line geometric distance.
5. If vectors are L2-normalized (‖x‖=1), cosine similarity and dot product become equivalent.
6. Normalization changes geometry — it's not automatically a speed optimization, it's a design decision tied to what the model/index expects.
7. Never normalize "because it's best practice" without knowing what your embedding model assumes.
8. Different data modalities (image/audio/video) often need different distance metrics than text.
9. The metric is chosen based on the representation and the retrieval goal, not a fixed law.
10. Metric choice affects both retrieval *quality* and what indexing methods are efficient to use.

**Where this connects:** the metric you pick determines how the index (Section 5) is even allowed to organize the search space.

---

## Section 5 — Indexing (ANN, HNSW, IVF)

**One sentence:** With millions of vectors you can't brute-force compare against everything, so an index gives you likely neighbors without scanning it all — HNSW navigates a graph, IVF searches within pre-partitioned clusters.

10 things to know:
1. ANN = Approximate Nearest Neighbor — trades a little accuracy for a lot of speed.
2. HNSW = graph navigation; think "which neighbor should I visit next?"
3. IVF = partition first (like store aisles), then search only within relevant clusters.
4. HNSW has memory overhead because it stores graph connectivity on top of the vectors — not simply "HNSW = RAM monster."
5. IVF's quality depends heavily on how well the space was clustered ahead of time.
6. Both are approximate — you're trading perfect recall for practical speed.
7. Index choice affects recall, latency, and memory very differently at different scales.
8. Techniques like quantization, compression, disk-based indexes, and sharding exist specifically to manage the memory/speed trade-off at scale.
9. Indexing solves "how do I avoid comparing against everything," not "which candidate is actually best" — that's ranking's job (Section 8).
10. Index tuning is a real, ongoing engineering task, not a one-time setup.

**Where this connects:** indexing produces *candidates*; it never claims to produce the *final* answer — that requires ranking.

---

## Section 6 — Retrieval Pipeline (the "next wall")

**One sentence:** Retrieval ≠ nearest-neighbor search — nearest neighbors are just the raw candidate pool.

10 things to know:
1. Full pipeline: query → query processing/embedding → metadata filtering → ANN retrieval → top-K → reranking → top-N evidence → context construction → LLM.
2. "High similarity score" doesn't guarantee "best evidence for the LLM."
3. Retrieval and generation are separate layers — RAG is not the LLM, it's a knowledge-finding system feeding the LLM.
4. Recall (did we find the right doc at all) and precision (is the top result actually right) are different failure modes to debug separately.
5. Context construction (what you actually hand the LLM) is its own design problem, not automatic.
6. "Lost in the middle" — LLMs pay less attention to information buried in the middle of a long context.
7. Compression and parent-child retrieval exist to give the LLM tight, relevant evidence instead of raw chunk dumps.
8. Retrieval evaluation and generation evaluation are two separate questions: did retrieval work? did the LLM use it correctly?
9. A demo that "looks right" can still be retrieving the wrong evidence for the wrong reasons — this is why evaluation matters.
10. This pipeline view is the difference between "assembling LangChain components" and actually engineering retrieval.

**Where this connects:** this is the skeleton everything else (hybrid search, reranking, evaluation) attaches to.

---

## Section 7 — Geospatial, Ranking & "Don't Embed Everything"

**One sentence:** Semantic questions need vectors, exact constraints need metadata filters, geography needs geospatial distance, and "best/first" needs ranking — real systems combine signals instead of forcing everything through embeddings.

10 things to know:
1. "Nearest coffee shop" is a geospatial problem, not automatically a RAG/embedding problem.
2. Real-world lat/lon distance typically wants Haversine (spherical), not naive flat Euclidean.
3. Retrieval (who's a candidate) and ranking (who's best) are different operations — don't collapse them.
4. IVF has nothing to do with "having ratings" — that was a genuine early misconception worth remembering as a corrected one.
5. Structured attributes (price, rating, distance) don't need to be embedded — use the database for what databases are good at.
6. A checklist for any search problem: (1) what am I searching over, (2) what determines candidates, (3) what constraints exist, (4) how do I rank.
7. Real ranking often blends multiple signals with weights: `score = w1*semantic + w2*rating + w3*proximity + ...`.
8. This blended score is literally what a "ranking model" is — the next concept beyond simple retrieval.
9. Multiple independent relevance *signals* (semantic, lexical, structural, metadata, geospatial, business) is a more mature mental model than "one similarity score."
10. The trap to avoid: shoving numeric/structured data into vectors and hoping cosine similarity does the work of a database query.

**Where this connects:** this reframes hybrid search — vectors + BM25 is really just two signals out of many possible ones.

---

## Section 8 — Hybrid Search: BM25

**One sentence:** Dense search asks "what does this mean," BM25 asks "what exact words are here" — and BM25 wins on identifiers, SKUs, acronyms, and proper nouns that embeddings weren't built to memorize precisely.

10 things to know:
1. The Lexical Gap: dense embeddings can miss exact strings like IDs, part numbers, ticket numbers.
2. BM25 = classical sparse/keyword retrieval, still the gold standard for exact token matches.
3. Three ingredients: Term Frequency (TF), Inverse Document Frequency (IDF), document length normalization.
4. TF: how often the query term appears — more occurrences, more relevance signal (with diminishing returns).
5. IDF: rare terms across the corpus are more discriminative than common ones ("the" tells you nothing; "XQ-17B" tells you a lot).
6. Length normalization: prevents long documents from winning purely by having more chances to contain a term.
7. BM25 is essentially "does this doc contain my terms, how important/rare are they, how often, and how long is the doc."
8. Tokenizer design matters a lot — naive `.split(" ")` can mangle hyphenated IDs, SKUs, and punctuation-heavy identifiers.
9. BM25 doesn't understand synonyms or paraphrase — "medical center" won't match "hospital" unless the exact word is present.
10. Sparse and dense retrieval fail in *opposite* ways, which is exactly why combining them is valuable.

**Where this connects:** BM25's weakness (no semantic understanding) is dense retrieval's strength, and vice versa — that asymmetry is the whole justification for hybrid search.

---

## Section 9 — Hybrid Search: Reciprocal Rank Fusion (RRF)

**One sentence:** RRF ignores raw scores entirely and fuses results using only *rank position*, because a BM25 score and a cosine similarity score live on completely incompatible scales.

10 things to know:
1. The Score-Scale Incompatibility Problem: BM25 might output 0–25+, cosine outputs -1 to 1 — you cannot just add them.
2. There is no universal formula converting a BM25 score into a cosine-equivalent score.
3. RRF formula: `RRF(d) = Σ 1/(k + rank_in_retriever(d))` summed across each retrieval system.
4. RRF requires no score normalization at all — it sidesteps the incompatibility problem entirely by using rank, not magnitude.
5. Documents ranking well in *multiple* retrievers get a "consensus boost" — this is the real value of RRF.
6. `k` (commonly 60) is a smoothing constant, not a magic number — there's no universal correct value.
7. Smaller `k` → sharper emphasis on top ranks. Larger `k` → flattens the difference between rank positions.
8. Mental split: **score fusion** = "how do I combine incompatible numbers" (hard); **rank fusion** = "forget the numbers, compare positions" (RRF's trick).
9. Production vector DBs (Qdrant, Milvus, Weaviate, Pinecone) increasingly support sparse + dense natively in one engine, rather than stitching together two separate systems.
10. A common real mistake: running BM25 and vector search as two separate systems without keeping chunk IDs aligned across both — deletions/updates then desync silently.

**Where this connects:** RRF's output is a candidate pool, *not* the final answer — it feeds into reranking (cross-encoders, Episode 008), which judges query+document pairs jointly rather than just voting on rank agreement.

---

## The one big through-line to fall asleep on

```
DATA → CHUNKS → EMBEDDINGS → VECTOR DB (+metadata)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
                 SEMANTIC       LEXICAL       STRUCTURED
                (dense/ANN)     (BM25)      (metadata/geo)
                    │              │               │
                    └──────────────┼───────────────┘
                                   ▼
                              CANDIDATES
                                   │
                          FUSION (RRF) → RANKING
                                   │
                          RERANKING (next: cross-encoder)
                                   │
                             TOP EVIDENCE
                                   │
                                  LLM
                                   │
                                ANSWER
```

Every concept you've learned so far is a labeled box or arrow in this one diagram. That's the thing to actually hold onto — not the individual facts.