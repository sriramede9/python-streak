# RAG — The Mental Model I Want in My Head

> **Purpose:** A quick-read reference for understanding Retrieval-Augmented Generation deeply enough that I can explain it without mixing up embeddings, retrieval, context, prompting, chunking, reranking, and generation.
>
> Read this on the bus, before bed, during a coffee break, or whenever the RAG neurons decide to leave the building.

---

# 0. The One-Sentence Definition

## RAG is:

> **Find the right external information, put it into the model's context, and ask the model to answer using that information.**

The fundamental loop is:

```text
USER QUESTION
     ↓
   RETRIEVE
     ↓
RELEVANT CONTEXT
     ↓
   GENERATE
     ↓
   ANSWER
```

Everything else is engineering around these two big jobs:

```text
1. Retrieval: What information should I give the model?
2. Generation: What should the model say using that information?
```

If I get lost in a RAG conversation, return here.

---

# 1. The Master Mental Model

Think of an LLM as a **very intelligent chef who cannot walk into the warehouse**.

The warehouse contains:

```text
company policies
PDFs
product docs
tickets
emails
research papers
code
```

The chef knows how to reason and communicate, but the warehouse is external.

RAG builds the supply chain:

```text
                 COMPANY KNOWLEDGE
                        │
                        ▼
                  INGEST / PARSE
                        │
                        ▼
                     CHUNKS
                        │
                        ▼
                   EMBEDDINGS
                        │
                        ▼
                 VECTOR DATABASE
                        │
                        │
USER QUESTION ──────────┘
       │
       ▼
   EMBEDDING
       │
       ▼
    RETRIEVAL
       │
       ▼
 RELEVANT CHUNKS
       │
       ▼
     PROMPT
       │
       ▼
      LLM
       │
       ▼
     ANSWER
```

### Remember this:

> **Embeddings do not answer questions.**
>
> **The vector database does not answer questions.**
>
> **Retrieval does not answer questions.**
>
> **The LLM generates the answer.**

This single separation prevents an embarrassing amount of RAG confusion.

---

# 2. RAG Is Not One Thing

RAG is a system made of stages.

```text
             RAG SYSTEM
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
 INGEST       RETRIEVE      GENERATE
    │            │            │
    ▼            ▼            ▼
 parse        search       prompt
 chunk        rank         reason
 embed        filter       answer
 store        rerank      cite
```

A bad answer can therefore come from many different places.

```text
Bad answer
   │
   ├── bad document parsing
   ├── bad chunking
   ├── bad embedding
   ├── bad query formulation
   ├── bad retrieval
   ├── bad ranking
   ├── bad context selection
   ├── bad prompt
   └── bad generation
```

Never jump straight to “the LLM is bad.”

The LLM may simply have been handed the wrong evidence.

---

# 3. Offline vs Online: The Most Important Architecture Split

A mature RAG system has two broad moments.

## Offline / ingestion time

Usually happens when documents enter the system.

```text
Document
   ↓
Parse
   ↓
Clean / normalize
   ↓
Structure-aware processing
   ↓
Chunk
   ↓
Embed
   ↓
Store
```

This can be expensive because it happens ahead of user queries.

## Online / query time

Happens when the user asks something.

```text
Question
   ↓
Query transformation
   ↓
Retrieve
   ↓
Filter
   ↓
Rerank
   ↓
Select context
   ↓
Prompt LLM
   ↓
Answer
```

### Memory hook

> **Ingestion prepares the library. Query time finds the books. Generation reads the selected pages.**

---

# 4. Why Chunking Exists

Imagine a 300-page PDF.

We usually do not want one giant vector representing the whole PDF.

Why?

Because a single embedding would blur many topics together.

Instead:

```text
300-page document
      ↓
small meaningful units
      ↓
chunk 1
chunk 2
chunk 3
...
chunk 850
```

Now retrieval can find the relevant region.

### Chunking question to memorize

Do NOT start with:

> “How many tokens should my chunk be?”

Start with:

> **“What should one retrieval result represent?”**

That question drives the chunking strategy.

---

# 5. Chunking Mental Map

There is no sacred chunk size handed down from the mountain.

Common strategies:

```text
STRUCTURAL
   ↓
respect headings / sections / paragraphs

RECURSIVE
   ↓
split using a hierarchy of separators

SEMANTIC
   ↓
split when meaning changes

HIERARCHICAL
   ↓
parent context + smaller child retrieval units

HYBRID
   ↓
combine structure + size + semantics
```

## Structural chunking

Best mental model:

> **“The author already told me where the boundaries are.”**

Example:

```text
# Authentication
...

# Authorization
...

# Troubleshooting
...
```

Do not use machine learning to rediscover obvious headings just because machine learning exists.

Humans invented headings for a reason.

## Recursive chunking

Mental model:

> **“Try the biggest sensible boundary first, then break smaller only when necessary.”**

Example hierarchy:

```text
Document
  ↓
Sections
  ↓
Paragraphs
  ↓
Sentences
  ↓
Words
```

## Semantic chunking

Mental model:

> **“Find places where the topic changes.”**

One simple intuition:

```text
Sentence 1 ── similar ── Sentence 2
Sentence 2 ── similar ── Sentence 3
Sentence 3 ── VERY DIFFERENT ── Sentence 4
                                      ↑
                                 boundary
```

Embeddings turn sentences into vectors.

Cosine similarity tells us how close they are.

Semantic drift is essentially:

```text
distance(sentence/window A, sentence/window B)
                         ↑
                    bigger jump
                         ↓
                 likely topic shift
```

### Crucial distinction

Semantic chunking is **not** “embed everything and magically get perfect chunks.”

It is an algorithm that uses semantic similarity/distance as a signal for where boundaries may belong.

---

# 6. Embeddings: What They Actually Do

An embedding is a numerical representation of meaning.

Conceptually:

```text
“dog”
  ↓
[0.12, -0.44, 0.83, ...]
```

Another sentence:

```text
“A puppy is an animal commonly kept as a pet.”
  ↓
[0.10, -0.41, 0.79, ...]
```

They may be close in vector space because their meanings are related.

### The key idea

> **Embedding converts language into a geometry we can search.**

This is why vector databases exist.

---

# 7. What Cosine Similarity Means

A common similarity measure is:

```text
cosine_similarity(A, B)
```

Conceptually:

```text
same direction        → high similarity
very different        → low similarity
```

The exact values depend on the model and setup.

Do not develop mystical beliefs such as:

> “0.82 always means relevant.”

Similarity scores are meaningful **within a system**, not universal laws of physics.

---

# 8. Vector Database Mental Model

Think of a vector database as:

> **A warehouse optimized for finding vectors near another vector.**

You store:

```text
vector
+
chunk text
+
metadata
```

Example:

```json
{
  "id": "doc-17-chunk-4",
  "text": "Employees may work remotely...",
  "embedding": [0.12, -0.44, ...],
  "metadata": {
    "document_id": "hr-policy",
    "section": "Remote Work",
    "page": 7
  }
}
```

The vector is for mathematical search.

The text is what the LLM eventually reads.

The metadata is what lets your system filter, trace, cite, and reason about the source.

---

# 9. Query-Time Embedding

Suppose the user asks:

> “Can employees work from home three days a week?”

We embed the query too.

```text
Question
   ↓
query embedding
   ↓
search vector database
   ↓
nearest chunks
```

### Core symmetry

```text
DOCUMENT CHUNK → embedding
USER QUERY     → embedding

then compare them
```

But there is an important caveat:

> **Query and document text are not always naturally shaped the same way.**

That is the **query-document asymmetry** problem.

This motivates techniques like:

```text
query rewriting
multi-query expansion
step-back prompting
HyDE
```

---

# 10. Retrieval: The Actual Search Problem

Retrieval asks:

> **Which pieces of the corpus should enter the model's context?**

A simple pipeline:

```text
Query
  ↓
Embed
  ↓
Vector search
  ↓
top K
```

For example:

```text
Top 10 candidates
      ↓
  relevance check
      ↓
Top 4 useful chunks
      ↓
     LLM
```

Do not confuse **candidate retrieval** with **final context selection**.

---

# 11. Dense Retrieval vs Sparse Retrieval

## Dense retrieval

Uses embeddings.

Good at semantic relationships.

```text
“car”
≈
“automobile”
```

## Sparse / lexical retrieval

Uses exact or near-exact terms.

BM25 is a classic example.

Good when exact wording matters.

```text
“ERR_CONNECTION_RESET”
```

An embedding might understand the concept.

BM25 can directly notice the exact error string.

---

# 12. Hybrid Retrieval

A strong mental model is:

```text
                QUERY
                  │
          ┌───────┴────────┐
          ▼                ▼
       BM25             VECTOR
          │                │
          └───────┬────────┘
                  ▼
               FUSION
                  ▼
              RERANKER
                  ▼
             FINAL CONTEXT
```

This combines lexical and semantic strengths.

---

# 13. Reciprocal Rank Fusion (RRF)

RRF is a way of combining rankings from different retrieval systems.

Mental model:

> **“I have two judges. Reward documents that both judges rank highly.”**

A simplified formula is:

```text
score(d) = Σ 1 / (k + rank(d))
```

The exact choice of `k` is an implementation detail.

What matters conceptually:

```text
appears high in multiple rankings
          ↓
      gets rewarded
```

RRF does not magically understand relevance.

It combines rankings.

---

# 14. Reranking

Initial retrieval is usually optimized for speed.

That means we may retrieve more candidates than we finally need.

Example:

```text
100,000 documents
      ↓
vector/BM25 retrieval
      ↓
Top 50
      ↓
expensive reranker
      ↓
Top 5
      ↓
LLM
```

A cross-encoder reranker can look at:

```text
(query, candidate chunk)
```

and score their relevance together.

### Mental model

```text
Retriever = “Who might be relevant?”
Reranker  = “Which of those are actually the best?”
```

This distinction is extremely important.

---

# 15. Cross-Encoder vs Bi-Encoder

## Bi-encoder

Encode separately:

```text
query → vector
chunk → vector
```

Then compare vectors.

Fast because chunk embeddings can be precomputed.

## Cross-encoder

Put both together:

```text
[QUERY + CHUNK]
       ↓
     MODEL
       ↓
 relevance score
```

Potentially more accurate, but much more expensive to run across many candidates.

### Memory hook

> **Bi-encoder: compare fingerprints.**
>
> **Cross-encoder: actually read the two things together.**

---

# 16. Late Interaction / ColBERT Mental Model

Late interaction sits between simple vector similarity and full cross-encoding.

Instead of reducing the entire text to one vector, it keeps token-level representations.

Mental model:

```text
Query tokens:   q1 q2 q3 q4
                   ↓
                token vectors

Document tokens: d1 d2 d3 ...
                   ↓
                token vectors

compare token-level matches
```

The major idea:

> **Keep more fine-grained information than one vector, while avoiding full query-document encoding from scratch.**

---

# 17. Query Transformation

Users ask messy questions.

Documents are written differently.

Example:

```text
User:
“Can I cancel this thing before renewal?”

Document:
“Subscribers may terminate recurring service prior to the
renewal date in accordance with Section 8.2.”
```

A good retrieval system may transform the query before searching.

---

# 18. Query Rewriting

Rewrite the user's question into a search-friendly form.

```text
Original:
“Can I cancel this thing before renewal?”

Rewritten:
“What is the cancellation policy before the subscription renewal date?”
```

The goal is not to make the user sound smarter.

The goal is:

> **better alignment with the language likely to occur in the corpus.**

---

# 19. Multi-Query Expansion

One query may have multiple plausible formulations.

```text
Original query
      ↓
 ┌────┼────┐
 ▼    ▼    ▼
Q1   Q2   Q3
 \    |   /
  \   |  /
   retrieval
      ↓
    fusion
```

Example:

```text
“How long is parental leave?”

→ “parental leave duration”
→ “employee parental leave entitlement”
→ “maximum parental leave period”
```

This increases recall by searching different semantic/lexical angles.

---

# 20. Step-Back Prompting

Instead of searching the narrow question directly, ask a broader conceptual question.

Example:

```text
Original:
“What penalty applies to this clause?”

Step back:
“What is the company's policy regarding violations of this type?”
```

Mental model:

> **Move one conceptual level upward before retrieving.**

Useful when the original question is too specific, underspecified, or dependent on background context.

---

# 21. HyDE

HyDE = **Hypothetical Document Embeddings**.

The core trick:

```text
user question
      ↓
LLM writes a hypothetical answer/document
      ↓
embed that hypothetical document
      ↓
search real documents
```

Why might this help?

Because the user query may be short and unlike the prose in the actual corpus.

Example:

```text
Query:
“What happens after a failed deployment?”
```

HyDE might generate a hypothetical passage containing terms like:

```text
rollback
incident response
deployment failure
health checks
release recovery
```

Then retrieve using that richer representation.

### Crucial warning

The hypothetical document is a **retrieval aid**, not evidence.

Never confuse:

```text
generated hypothesis
```

with:

```text
retrieved source of truth
```

---

# 22. Recall vs Precision

These two ideas appear everywhere in retrieval.

## Recall

> Did I retrieve the relevant information at all?

## Precision

> Of the things I retrieved, how many were actually relevant?

Example:

```text
Need: 3 relevant chunks

Retrieved:
✅ relevant
✅ relevant
❌ irrelevant
❌ irrelevant
❌ irrelevant
```

You have some recall, but poor precision.

Another case:

```text
Retrieved:
✅ relevant
✅ relevant
✅ relevant
```

Excellent precision, but perhaps poor recall if an important fourth chunk was missed.

### Mental sequence

```text
FIRST: don't miss important evidence
        ↓
THEN: remove irrelevant evidence
```

That naturally motivates:

```text
retriever → reranker
```

---

# 23. Context Is a Budget

The LLM has a context window.

That does not mean:

> “Put the whole database in there.”

The context window is a budget.

```text
         CONTEXT BUDGET
┌───────────────────────────────┐
│ system prompt                 │
│ user question                 │
│ retrieved chunks              │
│ conversation history          │
│ instructions                  │
└───────────────────────────────┘
```

Every extra chunk consumes tokens.

Too little context:

```text
missing evidence
```

Too much context:

```text
noise
conflicts
latency
cost
attention dilution
```

### Important principle

> **More context is not automatically better context.**

---

# 24. The LLM Is Not the Retrieval Engine

This distinction matters enormously.

The LLM receives something like:

```text
System instructions
+
User question
+
Retrieved context
```

Then generates an answer.

The LLM itself is not guaranteed to:

```text
search your vector DB
find the right policy
verify every claim
```

Those are responsibilities of the surrounding application, unless tools/agents are explicitly used.

---

# 25. Grounding

Grounding means the answer is tied to supplied evidence.

A grounded prompt might say:

```text
Answer using only the provided context.
If the answer is not supported, say you don't know.
```

But prompting alone does not create truth.

If retrieval is wrong:

```text
wrong chunks
   ↓
LLM
   ↓
confident wrong answer
```

This is why:

> **RAG quality is upstream-heavy.**

---

# 26. The RAG Failure Equation

A useful mental model:

```text
Overall RAG quality
≈
Retrieval quality
×
Context quality
×
Generation quality
```

This is not a literal scientific equation.

It is an engineering intuition:

> A brilliant generator cannot rescue consistently terrible retrieval.

If the correct evidence never enters the prompt, the model has nothing reliable to work with.

---

# 27. Why Hallucinations Happen in RAG

RAG does not automatically eliminate hallucinations.

Possible failure chain:

```text
Question
  ↓
bad query formulation
  ↓
wrong retrieval
  ↓
missing evidence
  ↓
LLM fills the gap
  ↓
hallucination
```

Or:

```text
correct evidence
      ↓
conflicting evidence
      ↓
poor context ordering
      ↓
LLM chooses badly
```

Or:

```text
correct retrieval
      ↓
bad prompt
      ↓
overconfident generation
```

RAG is not an anti-hallucination spell.

It is an **evidence-supply architecture**.

---

# 28. Metadata Is Not Decoration

Each chunk should usually carry useful metadata.

```json
{
  "document_id": "policy-2026",
  "source": "employee-handbook.pdf",
  "page": 17,
  "section": "Parental Leave",
  "chunk_id": 42
}
```

Metadata enables:

```text
filtering
security
citations
traceability
debugging
analytics
source linking
```

A strong RAG system knows not only:

> “Here is text.”

but also:

> “Here is exactly where this text came from.”

---

# 29. Security Mental Model

Retrieval must respect authorization.

Imagine two employees:

```text
Alice → Finance documents
Bob   → Engineering documents
```

If the vector database retrieves Alice's confidential chunk for Bob, you have a security failure even if the answer is factually perfect.

Therefore:

```text
USER
 ↓
AUTHORIZATION
 ↓
FILTERABLE CORPUS
 ↓
RETRIEVAL
```

Do not treat metadata filtering as an optional optimization when permissions are involved.

---

# 30. Chunk Size Is a Retrieval Tradeoff

Small chunks:

```text
+ precise
+ focused
- less context
- may lose relationships
```

Large chunks:

```text
+ more context
+ preserve relationships
- less precise
- more noise
```

Think:

```text
precision ←────────────→ context
 small chunk             large chunk
```

There is no universal winner.

---

# 31. Overlap Is a Context Bridge

Suppose a sentence at the boundary depends on the previous paragraph.

Overlap can preserve continuity.

```text
CHUNK A
[A B C D E]

CHUNK B
        [D E F G H]
```

The repeated region is the bridge.

But excessive overlap creates duplication.

Use it because boundary continuity needs it, not because “20% overlap” sounds scientifically respectable.

---

# 32. Parent-Child Retrieval

Sometimes we want:

```text
small child chunks for precise retrieval
```

but:

```text
large parent context for generation
```

Architecture:

```text
Document
   │
   ├── Parent section
   │     ├── Child chunk 1
   │     ├── Child chunk 2
   │     └── Child chunk 3
   │
   └── Parent section 2
         ├── Child chunk 4
         └── Child chunk 5
```

Retrieve the child.

Then expand to the parent.

Mental model:

> **Search with a microscope. Read with the surrounding paragraph.**

---

# 33. Lost in the Middle

Large contexts can contain the correct evidence but still perform poorly if important material is buried among lots of irrelevant material.

This creates a practical lesson:

> **Retrieval is not just “get more documents.” Context ordering and selection matter.**

Possible responses include:

```text
reranking
context compression
deduplication
better chunking
ordering strategies
parent-child retrieval
```

---

# 34. Deduplication

Overlapping retrieval systems can return nearly identical chunks.

Example:

```text
BM25 → chunk 10
Vector → chunk 10
Vector → chunk 11
Reranker → chunk 10
```

The LLM does not need the same evidence repeated five times.

Deduplication protects:

```text
token budget
latency
clarity
```

---

# 35. Retrieval Is a Ranking Problem

A useful abstraction is:

```text
                    DOCUMENT CORPUS
                         │
                         ▼
                       query
                         │
                         ▼
                    candidate set
                         │
                         ▼
                  scoring / ranking
                         │
                         ▼
                    top candidates
                         │
                         ▼
                      reranking
                         │
                         ▼
                   final evidence
```

The system is continuously answering:

> **“Which evidence deserves scarce context-window space?”**

That is a powerful way to think about modern RAG.

---

# 36. Evaluation: Stop Guessing

Do not judge RAG only by reading a few answers and thinking:

> “Seems pretty good.”

Build a test set.

Example:

```text
question
expected evidence
expected answer
source
```

Then test retrieval separately from generation.

---

# 37. Retrieval Evaluation

Useful conceptual metrics include:

```text
Recall@K
Precision@K
MRR
nDCG
```

### Recall@K

Did the relevant item appear somewhere in the top K?

### MRR

How high was the first relevant result?

### nDCG

How good was the ranking when multiple relevance levels matter?

The precise metric depends on the task.

The important principle:

> **Evaluate retrieval independently from answer generation.**

Otherwise you cannot tell which layer is broken.

---

# 38. End-to-End Evaluation

You also want to evaluate:

```text
faithfulness / groundedness
answer relevance
completeness
citation correctness
latency
cost
```

Think in layers:

```text
        RAG EVALUATION
              │
      ┌───────┴────────┐
      ▼                ▼
 Retrieval          Generation
   quality             quality
      │                │
      ▼                ▼
right evidence?     right answer?
```

---

# 39. Debugging RAG: Follow the Evidence

When an answer is wrong, debug from left to right.

```text
1. What did the user ask?
2. What query did we actually search?
3. What candidates were retrieved?
4. Was the correct chunk present?
5. What did the reranker choose?
6. What context entered the prompt?
7. What did the model answer?
```

This is much better than staring at the final answer and blaming the LLM.

---

# 40. The Six Most Common RAG Mistakes

## Mistake 1: Treating embeddings as magic

Embedding ≠ truth.

Embedding ≠ answer.

Embedding = representation useful for similarity/search.

## Mistake 2: Optimizing chunk size without an evaluation set

You are tuning numbers against vibes.

## Mistake 3: Retrieving too much

More context can become noise.

## Mistake 4: Forgetting metadata

No source, no traceability, no useful filtering.

## Mistake 5: Mixing retrieval failure with generation failure

A bad answer can originate before the LLM even sees the prompt.

## Mistake 6: Assuming one retrieval method is enough

Dense, lexical, metadata, reranking, and query transformation solve different problems.

---

# 41. The Complete Modern RAG Map

Keep this diagram in your head:

```text
                    ┌──────────────────────┐
                    │      DOCUMENTS       │
                    └──────────┬───────────┘
                               │
                         PARSE / CLEAN
                               │
                         STRUCTURE
                               │
                            CHUNK
                               │
                            EMBED
                               │
                       VECTOR DATABASE
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 │       USER QUERY          │
                 │             │             │
                 │      REWRITE / EXPAND     │
                 │             │             │
                 │        QUERY EMBED        │
                 │             │             │
                 │    ┌────────┴─────────┐   │
                 │    ▼                  ▼   │
                 │  VECTOR             BM25 │
                 │    │                  │   │
                 │    └────────┬─────────┘   │
                 │             ▼             │
                 │            RRF             │
                 │             ▼             │
                 │         TOP CANDIDATES     │
                 │             ▼             │
                 │          RERANK           │
                 │             ▼             │
                 │       CONTEXT SELECTION   │
                 │             ▼             │
                 └──────────► PROMPT ◄────────┘
                               │
                               ▼
                              LLM
                               │
                               ▼
                          GROUNDED ANSWER
```

That is the map.

Everything you learn later should attach somewhere to it.

---

# 42. The Three Questions That Explain Almost Everything

Whenever somebody introduces a new RAG technique, ask:

### Question 1

> **What problem is this solving?**

### Question 2

> **At which stage of the pipeline does it operate?**

### Question 3

> **What tradeoff does it introduce?**

Example:

```text
Semantic chunking

Problem:
fixed boundaries can cut across topics

Stage:
ingestion

Tradeoff:
more computation + more complexity
```

Example:

```text
Reranking

Problem:
initial retrieval ranking is imperfect

Stage:
retrieval

Tradeoff:
more latency / compute
```

Example:

```text
HyDE

Problem:
short query poorly resembles document language

Stage:
query transformation

Tradeoff:
extra LLM call + possible query drift
```

---

# 43. A RAG Technique Cheat Sheet

| Technique | Main problem | Stage |
|---|---|---|
| Structural chunking | preserve document organization | ingestion |
| Recursive chunking | split long text sensibly | ingestion |
| Semantic chunking | detect topic shifts | ingestion |
| Overlap | boundary continuity | ingestion |
| Parent-child retrieval | precision vs context | retrieval |
| Embeddings | semantic representation | ingestion/query |
| BM25 | exact lexical matching | retrieval |
| Hybrid search | combine lexical + semantic | retrieval |
| RRF | merge ranked lists | retrieval |
| Cross-encoder | improve relevance ranking | reranking |
| ColBERT / late interaction | finer-grained matching | retrieval/ranking |
| Query rewriting | improve search formulation | query transformation |
| Multi-query | increase recall via variants | query transformation |
| Step-back | retrieve broader context | query transformation |
| HyDE | bridge query-document language gap | query transformation |
| Metadata filtering | constrain search space | retrieval |
| Context compression | remove unnecessary content | post-retrieval |
| Deduplication | reduce repeated evidence | post-retrieval |
| Citation metadata | trace evidence | generation/output |

---

# 44. The RAG Decision Tree

When building a system:

```text
START
  │
  ▼
What is the source?
  │
  ├── structured records → maybe retrieve records directly
  │
  └── documents → continue
          │
          ▼
Does the document have useful structure?
          │
     ┌────┴────┐
     ▼         ▼
    YES        NO
     │         │
 structure   recursive / semantic
     │         │
     └────┬────┘
          ▼
What should one retrieval result represent?
          │
          ▼
Choose chunking strategy
          │
          ▼
Choose dense / sparse / hybrid retrieval
          │
          ▼
Need more recall?
    ┌─────┴─────┐
    ▼           ▼
   YES          NO
    │           │
query expand   baseline
multi-query
HyDE
    │
    ▼
Need better precision?
          │
          ▼
       rerank
          │
          ▼
Need more context?
          │
          ▼
 parent expansion / context selection
          │
          ▼
       GENERATE
          │
          ▼
      EVALUATE
```

---

# 45. The “Before Chunking” Checklist

Before writing `chunk(text)`, ask:

```text
□ What is the source type?
□ Does it already have meaningful structure?
□ What should one retrieval result represent?
□ What is the embedding model's input limit?
□ Is semantic coherence important?
□ Do tables/code/lists need special handling?
□ Do I need overlap?
□ What metadata must survive?
□ Will retrieval be dense, sparse, hybrid, or filtered?
□ How will I evaluate whether the chunks are good?
```

The most important one:

> **What should one retrieval result represent?**

---

# 46. The “RAG Interview” Explanation

When someone asks:

> “Explain RAG.”

A strong answer can be:

> RAG is an architecture where a system retrieves relevant external knowledge and supplies it to an LLM as context before generation. At ingestion time we parse documents, chunk them, create embeddings, and store the chunks with metadata. At query time we transform and embed the user query, retrieve candidate passages using dense, sparse, or hybrid search, optionally rerank them, select the best context, and pass that evidence to the LLM. The quality of the final answer therefore depends heavily on retrieval quality, context selection, and grounding, not just the model itself.

Then draw:

```text
Docs → chunks → embeddings → index

Query → retrieval → rerank → context → LLM → answer
```

That alone sounds much more senior than listing random buzzwords.

---

# 47. The “Why Isn't My RAG Working?” Checklist

```text
1. Did parsing destroy structure?
2. Are chunks too small?
3. Are chunks too large?
4. Are semantic boundaries reasonable?
5. Is the query poorly phrased?
6. Is the embedding model appropriate?
7. Is vector retrieval missing obvious chunks?
8. Would BM25 catch exact terms better?
9. Should I use hybrid retrieval?
10. Is reranking needed?
11. Am I retrieving too many chunks?
12. Are duplicates flooding the context?
13. Is metadata filtering correct?
14. Did the relevant evidence actually reach the prompt?
15. Is the prompt asking the model to use the evidence?
16. Is the final answer supported by the evidence?
```

---

# 48. The “Do Not Confuse These” Table

| Do not confuse | With |
|---|---|
| embedding | retrieval |
| retrieval | generation |
| chunk size | context-window size |
| vector similarity | relevance truth |
| candidate retrieval | final ranking |
| reranking | generation |
| generated HyDE text | source evidence |
| metadata | embedding |
| context window | database |
| high similarity | guaranteed relevance |
| more chunks | better answer |
| RAG | fine-tuning |

---

# 49. RAG vs Fine-Tuning

A useful mental distinction:

## RAG

Changes **what information the model sees** at runtime.

```text
knowledge → retrieve → prompt
```

## Fine-tuning

Changes **the model's learned behavior/parameters**.

```text
training data → parameter updates → model behavior
```

Very roughly:

```text
Need fresh / private / changing knowledge?
→ RAG is often appropriate.

Need consistent behavior / style / task specialization?
→ fine-tuning may help.
```

These are not mutually exclusive.

---

# 50. The Deepest Mental Model

RAG is really a problem of **information flow under constraints**.

You have:

```text
huge knowledge base
      ↓
limited retrieval budget
      ↓
limited context budget
      ↓
limited attention
      ↓
need accurate answer
```

Therefore the system constantly tries to maximize:

```text
RELEVANCE OF EVIDENCE
```

while minimizing:

```text
NOISE
LATENCY
COST
DUPLICATION
SECURITY RISK
```

This is why seemingly different techniques all fit together.

---

# 51. One Giant Mental Picture

Memorize this before memorizing algorithms:

```text
                       RAG
                        │
        ┌───────────────┴────────────────┐
        │                                │
   KNOWLEDGE SIDE                    QUESTION SIDE
        │                                │
     documents                         user
        │                                │
      parsing                          query
        │                                │
    structure                        rewriting
        │                                │
     chunking                        expansion
        │                                │
    embeddings                          │
        │                                │
      index                        query embedding
        │                                │
        └───────────────┬────────────────┘
                        │
                     RETRIEVE
                        │
              ┌─────────┴─────────┐
              │                   │
            dense               sparse
              │                   │
              └─────────┬─────────┘
                        │
                       RRF
                        │
                    reranking
                        │
                 context selection
                        │
                      PROMPT
                        │
                       LLM
                        │
                     ANSWER
                        │
                    evaluation
                        │
                  improve system
                        │
                        └──────► repeat
```

This is not a collection of disconnected tricks.

It is one pipeline.

---

# 52. The 60-Second Bus Read

When I only have one minute, read this:

```text
RAG = retrieve evidence + generate with evidence.

INGESTION:
parse → structure → chunk → embed → store

QUERY:
query → transform → retrieve → rerank → select context

GENERATION:
context + question → LLM → answer

Embeddings:
meaning → vector geometry

Vector DB:
find nearby vectors

BM25:
find matching words / terms

Hybrid:
use both

RRF:
merge rankings

Reranker:
judge candidate relevance more carefully

Chunking:
ask “what should one retrieval result represent?”

Semantic chunking:
use meaning shifts as boundaries

HyDE:
generate hypothetical document → embed it → retrieve real docs

Recall:
did I find the relevant evidence?
Precision:
did I mostly retrieve useful evidence?

Context:
is a scarce budget, not a dumping ground.

Debugging:
query → retrieval → rerank → context → prompt → answer

Most important idea:
THE LLM CAN ONLY REASON OVER THE EVIDENCE YOU ACTUALLY GIVE IT.
```

---

# 53. The Bedtime Recall Test

Close your eyes and answer these without looking:

```text
1. What is the difference between ingestion and query time?
2. What exactly is an embedding?
3. What does a vector database do?
4. Why do we chunk?
5. What should determine chunk boundaries?
6. Dense vs sparse retrieval?
7. Why hybrid retrieval?
8. What does RRF do?
9. Retriever vs reranker?
10. Bi-encoder vs cross-encoder?
11. What problem does query rewriting solve?
12. What problem does HyDE solve?
13. What is recall vs precision?
14. Why can more context make things worse?
15. How would you debug a bad RAG answer?
```

If I can explain those in plain English, I understand RAG.

If I can only recite definitions, I am memorizing RAG.

Those are very different species.

---

# 54. Final Rules to Burn Into Memory

## Rule 1

> **RAG is retrieval first, generation second.**

## Rule 2

> **Embeddings create a searchable representation; they do not create truth.**

## Rule 3

> **Chunking should serve retrieval.**

## Rule 4

> **Ask what one retrieval result should represent.**

## Rule 5

> **Use lexical search when exact terms matter.**

## Rule 6

> **Use semantic search when meaning matters.**

## Rule 7

> **Use reranking when initial retrieval is too noisy.**

## Rule 8

> **Context is a scarce resource.**

## Rule 9

> **Generated text is not evidence. Retrieved source material is evidence.**

## Rule 10

> **When RAG fails, inspect the evidence path before blaming the LLM.**

---

# 55. The Final Mental Model

If everything else disappears from my brain, keep this:

```text
            KNOWLEDGE
                │
              chunk
                │
              embed
                │
              index
                │
                │
QUESTION ──► RETRIEVE ──► RERANK ──► CONTEXT ──► LLM
                ▲                                      │
                │                                      ▼
          transform query                           ANSWER

```

And the question underneath every RAG technique is:

> **“How do I get the right evidence into the model's context, reliably and efficiently?”**

That is the game.

Everything else is an optimization, a tradeoff, or a way of fixing a particular failure mode.

---

## Personal Note

Do not try to memorize every algorithm in one sitting.

First make the pipeline feel like a physical machine in your head:

```text
DOCUMENTS
   ↓
prepare the warehouse
   ↓
CHUNKS
   ↓
turn meaning into searchable geometry
   ↓
INDEX
   ↓
QUESTION
   ↓
search the warehouse
   ↓
RERANK
   ↓
choose the evidence
   ↓
LLM
   ↓
write the answer
```

Once that picture is stable, new RAG techniques stop looking like random academic spells.

They become answers to very specific engineering problems.

And that is when RAG starts becoming something you **understand**, rather than something you can merely describe.
