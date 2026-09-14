# 🧭 Vector Indexing Bible
### A practical decision guide for choosing Cosine, L2, and Inner Product

> **Purpose:** Stop choosing a vector index because a tutorial happened to use it.
> This README is a field guide for answering: **"What does my vector mean, and what should my index optimize?"**

---

## 📌 The 30-Second Rule

Before choosing an index, ask one question:

> **Does vector magnitude carry meaning, or do I only care about direction?**

| What matters? | Usually choose |
|---|---|
| Semantic meaning / conceptual direction | **Cosine** |
| Absolute geometric distance | **L2 / Euclidean** |
| Direction **plus meaningful magnitude** | **Inner Product (IP / Dot Product)** |
| Unit-normalized vectors and you want dot-product search | **IP** |

For most ordinary **text embeddings, RAG, semantic search, document retrieval, and text classification**, start with **Cosine** unless you have a specific reason not to.

---

# 🧠 1. The Mental Model You Need First

A vector is not merely a list of floats.

Imagine:

```text
                 magnitude
                    ↑
                    |
                    |       B
                    |      /
                    |     /
                    |    /  θ
                    |   /
                    |  /
                    | / A
                    |/
                    +----------------→
                         direction
```

Every vector has two useful properties:

### Direction

"What concept does this vector point toward?"

For semantic embeddings, direction often captures things such as:

- topic
- meaning
- intent
- semantic relationships
- similarity

### Magnitude

"How large/strong is this vector?"

Magnitude may or may not be meaningful.

For ordinary sentence embeddings, **you usually should not assume that vector length represents importance, popularity, quality, or confidence**.

That distinction is critical.

---

# 🎯 2. The Three Main Metrics

## Cosine Similarity

Cosine compares the **angle** between vectors.

```text
cosine_similarity(u, v)
    = (u · v) / (||u|| ||v||)
```

A common distance representation is:

```text
cosine_distance = 1 - cosine_similarity
```

### Mental model

> **"Are these vectors pointing in the same semantic direction?"**

Magnitude is normalized away.

### Best default for

- semantic search
- RAG
- text embeddings
- document retrieval
- question → passage matching
- variable-length text
- cross-document semantic comparison

---

# 📏 3. L2 / Euclidean Distance

L2 measures straight-line distance.

```text
d(u,v) = sqrt( Σ (uᵢ - vᵢ)² )
```

Some vector indexes use **squared L2** because the square root does not change ranking.

```text
d²(u,v) = Σ (uᵢ - vᵢ)²
```

### Mental model

> **"How far apart are these points in absolute space?"**

Magnitude matters.

### Best when

The coordinates themselves have meaningful geometry and scale.

Examples:

- physical coordinates
- sensor measurements
- engineered feature vectors
- spatial representations
- some image/audio feature spaces

---

# ⚡ 4. Inner Product / Dot Product

```text
u · v = Σ uᵢvᵢ
```

Dot product combines:

- directional alignment
- vector magnitude

### Mental model

> **"How strongly do these vectors interact?"**

Large aligned vectors can produce a larger dot product than small aligned vectors.

This makes IP especially useful when magnitude is deliberately meaningful.

---

# 🌲 5. Master Decision Flow

Use this every time.

```text
START
  |
  v
What do the vector values represent?
  |
  +--> Semantic/text embedding?
  |       |
  |       +--> Do you have a special reason magnitude matters?
  |               |
  |               +--> NO --> COSINE
  |               |
  |               +--> YES --> IP
  |
  +--> Physical/geometric coordinates?
  |       |
  |       +--> YES --> L2
  |
  +--> Recommendation / learned user-item interaction?
  |       |
  |       +--> YES --> IP is a strong candidate
  |
  +--> Unit-normalized vectors?
          |
          +--> YES --> IP and cosine produce equivalent ranking
          |           under the usual mathematical setup.
          |
          +--> NO --> Continue reasoning about magnitude.
```

## The important version

```text
                 ┌──────────────────────┐
                 │ What does vector     │
                 │ magnitude mean?      │
                 └──────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
        "Nothing useful"             "It matters"
              │                           │
              ▼                           ▼
           COSINE                Does larger magnitude
                                 help the objective?
                                      │
                              ┌───────┴───────┐
                              │               │
                             YES              NO
                              │               │
                              ▼               ▼
                              IP              L2
```

---

# 🧪 6. Five Cosine Scenarios

## 1. RAG: "How do I reset my password?"

**Vector:** document chunks

**Query:** `"How do I reset my password?"`

**Choose:** `cosine`

### Why?

You care about semantic direction.

A 20-word chunk and a 300-word chunk can express the same concept. You generally do not want the longer vector to win simply because it has a different magnitude.

---

## 2. Product semantic search: "high protein breakfast"

**Vector:** product descriptions

**Query:** `"high protein breakfast"`

**Choose:** `cosine`

### Why?

You want products whose descriptions are semantically related to the query.

You are not trying to reward a product because its embedding happens to have a larger norm.

---

## 3. FAQ retrieval

**Query:**

> "Can I return something after 30 days?"

**Documents:**

- "Our return policy allows unopened products within 30 days."
- "Shipping normally takes 3-5 business days."
- "Warranty coverage lasts one year."

**Choose:** `cosine`

### Why?

The relevant relationship is semantic meaning, not absolute vector size.

---

## 4. Code search

**Query:**

> "remove duplicates from a pandas dataframe"

**Candidates:**

- code snippets
- Stack Overflow explanations
- internal engineering documentation

**Choose:** `cosine`

### Why?

You want intent similarity between natural language and code/document embeddings.

---

## 5. Multilingual semantic retrieval

**Query:**

> "Where can I find the refund policy?"

**Corpus:** English + French + German + Spanish

**Choose:** `cosine`

### Why?

For multilingual embedding models, semantic direction is usually the important signal.

---

# 📏 7. Five L2 Scenarios

## 1. Indoor positioning

```text
point A = [12.2, 8.4]
point B = [13.1, 9.0]
```

**Choose:** `L2`

### Why?

These coordinates represent actual physical location.

Distance has a direct geometric interpretation.

---

## 2. Sensor anomaly detection

Suppose a machine feature vector contains:

```text
[temperature, vibration, pressure, rpm]
```

You have a known normal operating region.

**Choose:** `L2` when those feature scales have been deliberately engineered and calibrated.

### Why?

Absolute deviation from the expected operating point matters.

**Important:** feature scaling/standardization can be just as important as the metric.

---

## 3. Geometric feature matching

Suppose vectors encode measurements such as:

```text
[x_position, y_position, width, height]
```

**Choose:** `L2`

### Why?

You care about actual coordinate deviation.

---

## 4. Engineered numerical feature vectors

Example:

```text
[age, income, account_balance, transaction_count]
```

After appropriate preprocessing/scaling, you may use L2 for nearest-neighbor analysis.

### Why?

You are working with numerical feature geometry rather than semantic language embeddings.

---

## 5. Some image/audio feature spaces

Suppose a model produces a feature vector where Euclidean distance has been validated as a meaningful measure of similarity.

**Choose:** `L2`

### Why?

The correct metric comes from the representation and the training objective, not from the word "embedding."

---

# ⚡ 8. Five Inner Product Scenarios

## 1. Recommendation systems

A user vector:

```text
u = user preferences
```

An item vector:

```text
v = item representation
```

The model scores:

```text
score = u · v
```

**Choose:** `IP`

### Why?

The dot product directly models user-item affinity.

This is one of the classic Maximum Inner Product Search (MIPS) problems.

---

## 2. Learned retrieval models

A query encoder creates:

```text
q = query embedding
```

A document encoder creates:

```text
d = document embedding
```

The training objective explicitly uses:

```text
q · d
```

**Choose:** `IP`

### Why?

Your model's scoring function is literally the dot product.

Do not randomly swap it for cosine because a blog post said cosine is popular.

---

## 3. Product recommendation with a deliberate popularity signal

Suppose you intentionally construct:

```text
v = normalized_semantic_vector × popularity_weight
```

Then:

```text
q · v
```

can make aligned products with larger weights score higher.

**Choose:** `IP`

### Why?

You deliberately encoded magnitude as part of the ranking signal.

### Warning

Do **not** assume the raw norm of a normal embedding means popularity.

You must intentionally design and validate this behavior.

---

## 4. Ad ranking

A learned user-interest vector interacts with an advertisement vector:

```text
score = user_vector · ad_vector
```

**Choose:** `IP`

### Why?

The dot product can represent learned interaction strength, including magnitude.

---

## 5. Unit-normalized embeddings

If:

```text
||u|| = 1
||v|| = 1
```

then:

```text
u · v = cosine_similarity(u,v)
```

**Choose:** `IP` if your system is designed around normalized vectors and dot-product search.

### Why?

You get the same ranking as cosine similarity under the same vectors.

This is a mathematical equivalence, not a magical performance guarantee. Benchmark your actual system.

---

# 🧮 9. The Relationship Nobody Should Memorize Incorrectly

For normalized vectors:

```text
||u|| = ||v|| = 1
```

we have:

```text
L2² = ||u-v||²
     = ||u||² + ||v||² - 2(u·v)
     = 2 - 2(u·v)
```

Since:

```text
u · v = cosine_similarity(u,v)
```

then:

```text
L2² = 2 - 2 cosine_similarity
```

Therefore:

```text
Cosine ranking
      ≡
IP ranking
      ≡
L2 ranking
```

for unit-normalized vectors.

### This means

If all vectors are normalized, the three metrics can produce the same nearest-neighbor ordering.

### This does NOT mean

> "All three indexes are always interchangeable."

They are not.

Once magnitude varies meaningfully, the equivalence disappears.

---

# 🚨 10. The Biggest Trap: "Magnitude Means Popularity"

This statement is dangerous:

> "My embedding has a larger magnitude, therefore the product is more popular."

No.

A standard embedding model does not automatically promise:

```text
||embedding|| = popularity
```

If you want popularity to influence ranking, make it explicit.

For example:

```text
semantic_vector = normalize(embedding)

popularity_weight = f(ratings, rating_count, sales)

final_vector = semantic_vector * popularity_weight
```

Then IP can incorporate the magnitude.

But this is now **your ranking design**, not some secret property of embeddings.

---

# 🛒 11. Your Grocery / SmartCart Example

Imagine the user asks:

> "Find chicken breast for meal prep."

You have:

```text
Product
├── name
├── description
├── price
├── rating
├── rating_count
├── store
└── embedding
```

### Bad mental model

```text
embedding → magically contains price + rating + popularity
```

It doesn't.

### Better architecture

Use separate signals:

```text
                 ┌──────────────┐
Query ──────────►│ Semantic     │
                 │ Retrieval    │
                 └──────┬───────┘
                        │
                        ▼
                 Top 50 products
                        │
                        ▼
              ┌──────────────────┐
              │ Metadata /       │
              │ Business Ranking │
              └────────┬─────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
          price      rating     distance
                       │
                       ▼
                  Final ranking
```

For example:

```text
semantic_score
+ rating_score
+ price_score
+ store_distance_score
```

This is often cleaner than forcing every business signal into an embedding.

---

# 🧭 12. Decision Examples: "What Am I Actually Searching?"

| User request | Primary retrieval idea |
|---|---|
| "documents about vector databases" | **Cosine** |
| "products similar to this product" | **Cosine** or model-specific metric |
| "find nearby coordinates" | **L2** |
| "find users/items with maximum learned affinity" | **IP** |
| "find the closest sensor state" | **L2** |
| "find FAQ answering this question" | **Cosine** |
| "rank ads by learned interaction score" | **IP** |
| "semantic search over grocery descriptions" | **Cosine** |
| "recommend items using a two-tower model" | **IP** |
| "nearest physical location" | **L2** |

---

# 🏗️ 13. Separate Retrieval From Ranking

This is one of the most useful production concepts.

## Stage 1: Retrieval

Use your vector index to cheaply find candidates.

```text
10,000,000 products
        ↓
vector search
        ↓
Top 100 candidates
```

## Stage 2: Ranking

Use richer business logic.

```text
Top 100
   ↓
price
rating
availability
store
distance
inventory
promotions
user preferences
   ↓
Top 10
```

### Why?

Vector search is excellent at answering:

> "Which things are semantically similar?"

It is not necessarily the right place to encode every business rule known to humanity.

---

# 🧩 14. Metadata Filtering Is Not the Same as Vector Similarity

Suppose the user asks:

> "Find vegan protein powder under $30 at Walmart."

Do not expect the embedding alone to enforce:

```text
vegan = true
price < 30
store = Walmart
```

Use metadata filtering where your database supports it.

Conceptually:

```text
FILTER:
    vegan = true
    price < 30
    store = "Walmart"

THEN

VECTOR SEARCH:
    "protein powder"
```

Or use the database's supported filtered-vector-search strategy.

The exact execution order depends on the vector database and index architecture.

---

# 📊 15. Similarity vs Distance

This causes an absurd amount of avoidable confusion.

### Similarity

```text
higher = better
```

Example:

```text
0.95
0.82
0.61
```

### Distance

```text
lower = better
```

Example:

```text
0.05
0.18
0.39
```

Always check what your database returns.

Never blindly write:

```python
if score > 0.8:
    accept()
```

until you know whether `score` means:

```text
similarity
```

or:

```text
distance
```

---

# 🗄️ 16. ChromaDB Quick Reference

```python
import chromadb

client = chromadb.Client()
```

## Cosine

```python
collection = client.create_collection(
    name="semantic_search",
    metadata={"hnsw:space": "cosine"}
)
```

Use for:

```text
RAG
semantic search
FAQ retrieval
document search
product description search
```

## L2

```python
collection = client.create_collection(
    name="geometric_search",
    metadata={"hnsw:space": "l2"}
)
```

Use when:

```text
absolute geometric distance matters
```

## Inner Product

```python
collection = client.create_collection(
    name="recommendations",
    metadata={"hnsw:space": "ip"}
)
```

Use when:

```text
dot product is your scoring function
or magnitude intentionally contributes
```

---

# 🔎 17. FAISS Quick Reference

```python
import faiss

dimension = 384
```

## L2

```python
index = faiss.IndexFlatL2(dimension)
```

## Inner Product

```python
index = faiss.IndexFlatIP(dimension)
```

## Cosine with FAISS

FAISS commonly implements cosine search by:

```text
normalize vectors
+
IndexFlatIP
```

Conceptually:

```python
faiss.normalize_L2(vectors)

index = faiss.IndexFlatIP(dimension)
index.add(vectors)
```

Because normalized dot product equals cosine similarity.

---

# 🧱 18. HNSW: Metric vs Index Algorithm

Do not confuse these two concepts.

### Metric

Defines:

> "What does close mean?"

Examples:

```text
cosine
L2
inner product
```

### HNSW

Defines:

> "How can I search a huge vector collection efficiently?"

HNSW is an ANN index/search structure.

So:

```text
Metric = definition of similarity/distance

HNSW = search strategy
```

You can think:

```text
                 Vector Search
                      │
             ┌────────┴────────┐
             │                 │
          Metric             Index
             │                 │
      ┌──────┼──────┐          │
   Cosine    L2     IP        HNSW
```

This distinction becomes extremely important once you start tuning production vector systems.

---

# ⚙️ 19. ANN Does Not Mean "Approximate Embeddings"

ANN means:

> **Approximate Nearest Neighbor**

The vectors remain the vectors.

The **search process** is approximate.

Instead of comparing your query against every vector:

```text
query
  ↓
compare against 10,000,000 vectors
```

an ANN index tries to find excellent candidates much faster:

```text
query
  ↓
ANN structure
  ↓
small candidate set
  ↓
nearest neighbors
```

This is why indexes such as HNSW exist.

---

# 🧪 20. How to Test Your Metric Instead of Guessing

Build a small evaluation set.

```text
Query
  ↓
known relevant documents
  ↓
run vector search
  ↓
measure Recall@K
```

Example:

```text
Recall@10 =
relevant items found in top 10
--------------------------------
total relevant items
```

Compare:

```text
Cosine
L2
IP
```

under the same dataset and query set.

Then measure:

- Recall@K
- latency
- memory
- throughput / QPS
- index build time

The correct metric is the one that works for **your representation and objective**, not the one with the most enthusiastic blog post.

---

# 🚦 21. Production Decision Checklist

Before creating your collection/index:

- [ ] What do my vectors represent?
- [ ] Are they text embeddings, learned recommendation vectors, or numerical features?
- [ ] Is vector magnitude meaningful?
- [ ] Is magnitude intentionally encoded?
- [ ] Are my vectors unit-normalized?
- [ ] What scoring function did my embedding/retrieval model train for?
- [ ] Does my database return similarity or distance?
- [ ] Am I accidentally treating distance as similarity?
- [ ] Do I need metadata filtering?
- [ ] Should business ranking happen after vector retrieval?
- [ ] Have I tested Recall@K?
- [ ] Have I tested latency/QPS?
- [ ] Have I measured memory usage?
- [ ] Have I validated the metric on real queries?

---

# 🧠 22. The "I Forgot Everything" Cheat Sheet

If your brain has temporarily become a cache miss:

```text
TEXT / SEMANTIC SEARCH
        ↓
     COSINE
```

```text
PHYSICAL / NUMERICAL GEOMETRY
        ↓
       L2
```

```text
LEARNED DOT-PRODUCT SCORE
        ↓
       IP
```

```text
UNIT-NORMALIZED VECTORS
        ↓
   COSINE ≈ IP ≈ L2 ranking
```

```text
RATING / PRICE / STORE / INVENTORY
        ↓
Don't blindly stuff them into embeddings.
Use metadata filtering and/or a reranker.
```

---

# 🏆 23. Final Mental Model

Remember this sentence:

> **Cosine asks "Which way?"**
>
> **L2 asks "How far?"**
>
> **Inner Product asks "How strongly do these vectors interact?"**

And remember the more important engineering rule:

> **The vector representation, training objective, and ranking objective should determine the metric.**

Not:

```text
"I saw cosine in a tutorial."
```

Not:

```text
"Someone on Reddit said IP is faster."
```

Not:

```text
"Embedding = vector, therefore all vectors are interchangeable."
```

---

# 📚 24. One-Page Reference Table

| Scenario | Metric | Reason |
|---|---|---|
| RAG semantic retrieval | **Cosine** | Direction/meaning |
| FAQ search | **Cosine** | Semantic alignment |
| Product text search | **Cosine** | Semantic alignment |
| Code search | **Cosine** | Intent similarity |
| Multilingual retrieval | **Cosine** | Semantic direction |
| Physical coordinates | **L2** | Absolute distance |
| Numerical nearest neighbors | **L2** | Feature-space geometry |
| Sensor-state comparison | **L2** | Absolute deviation |
| Geometric measurements | **L2** | Spatial distance |
| Validated Euclidean image/audio features | **L2** | Feature geometry |
| Two-tower recommendation | **IP** | Learned dot-product affinity |
| MIPS | **IP** | Maximum inner product |
| Learned retrieval using dot product | **IP** | Matches training objective |
| Intent × deliberate magnitude weighting | **IP** | Magnitude contributes |
| Normalized vectors + dot-product infrastructure | **IP** | Equivalent ranking to cosine |
| Price/rating/store constraints | **Metadata / reranking** | Not inherently embedding semantics |

---

# 🧭 The Ultimate Rule

When you are about to create a vector index, stop and ask:

```text
┌─────────────────────────────────────────────┐
│ What does "similar" actually mean here?     │
└──────────────────────┬──────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Direction     Distance     Interaction
          │            │            │
          ▼            ▼            ▼
       COSINE          L2           IP
```

That is the decision.

Everything else is benchmarking, engineering, and occasionally discovering that humanity invented three names for something it could have explained on one page.
