# RAG & Vector Search Vocabulary — Daily Review

A practical Q&A guide for learning tokenization, embeddings, vectors, similarity metrics, and retrieval-augmented generation (RAG).

---

## Table of Contents

1. [Vocabulary](#vocabulary)
2. [Conceptual Questions](#conceptual-questions)
3. [The RAG Mental Model](#the-rag-mental-model)
4. [Key Mathematical Relationships](#key-mathematical-relationships)
5. [Quick Daily Review](#quick-daily-review)

---

# Vocabulary

## 1. What is tokenization?

**Answer:** Tokenization is the process of splitting text into smaller units called **tokens** that a model can process.

Example:

```text
"I love cats"
       ↓
["I", " love", " cats"]
```

The exact tokens depend on the tokenizer.

### RAG connection

Tokenization matters for model context limits, but **chunking** usually determines the pieces of documents that are embedded and stored in a vector database.

**Remember:**  
> Tokenization splits text into tokens. Chunking splits documents into useful pieces for retrieval.

---

## 2. What is a token ID?

**Answer:** A token ID is an integer representing a token in a tokenizer's vocabulary.

Conceptually:

```text
"cat" → token ID 4821
"dog" → token ID 9182
```

A token ID is simply an **identifier**.

It does not inherently contain semantic meaning.

For example, token ID `4821` isn't mathematically "closer" in meaning to `4822` than to `9182`.

**Remember:**  
> Token ID = identifier, not meaning.

---

## 3. What is an embedding?

**Answer:** An embedding is a numerical representation of something, such as text, that captures semantic information.

Example:

```text
"dog"
  ↓
[0.21, -0.43, 0.77, ...]
```

The numbers allow us to compare meanings mathematically.

**Important:** In this context, an embedding is typically represented as a **vector**.

**Remember:**  
> Embedding = semantic numerical representation.

---

## 4. What is a vector?

**Answer:** A vector is an ordered collection of numbers that can represent things such as magnitude and direction.

Example:

```text
[3, 4]
```

can be visualized as an arrow in 2D space.

In machine learning:

> An embedding is typically represented as a vector.

So you will often hear:

> **Embedding = vector representation of data**

---

## 5. What is dimensionality?

**Answer:** Dimensionality is the number of values/components in a vector.

Examples:

```text
[0.2, 0.5]          → 2 dimensions
[0.2, 0.5, 0.1]     → 3 dimensions
[0.2, ..., 0.7]     → 1536 dimensions
```

A 1536-dimensional embedding can be thought of as a point in a 1536-dimensional mathematical space.

You don't need to physically visualize all 1536 dimensions.

**Remember:**  
> Dimensionality = number of numbers in the vector.

---

## 6. What is normalization?

**Answer:** Normalization means scaling a vector according to a rule, often so that its magnitude becomes 1.

For L2 normalization:

```text
normalized_vector = vector / ||vector||₂
```

Example:

```text
[3, 4]
```

Magnitude:

```text
√(3² + 4²) = 5
```

After L2 normalization:

```text
[0.6, 0.8]
```

The magnitude is now 1.

L1 and L2 are common norms. L3 exists mathematically, but L2 normalization is the one you will commonly encounter in vector-search discussions.

**Remember:**  
> Normalization can remove the effect of magnitude so comparisons focus more on direction.

---

## 7. What is a unit vector?

**Answer:** A unit vector is a vector whose magnitude is exactly 1.

Example:

```text
[0.6, 0.8]
```

because:

```text
√(0.6² + 0.8²) = 1
```

Normalization can turn a vector into a unit vector.

**Remember:**  
> Unit vector = vector with length 1.

---

## 8. What is cosine similarity?

**Answer:** Cosine similarity measures how aligned two vectors are by looking at the angle between them.

Formula:

```text
cosine similarity = (A · B) / (||A|| ||B||)
```

Interpretation:

```text
  1  → same direction
  0  → perpendicular
 -1  → opposite direction
```

### Intuition

> "I don't care how far you go. If we're going in the same direction, we're buddies."

For semantic search, vectors pointing in similar directions generally represent similar meanings.

**Remember:**  
> Cosine similarity cares about direction/alignment, not raw magnitude.

---

## 9. What is Euclidean/L2 distance?

**Answer:** Euclidean distance measures the straight-line distance between two vectors/points.

Example:

```text
A = [1, 1]
B = [4, 5]
```

Distance:

```text
√((4-1)² + (5-1)²) = 5
```

So:

> Euclidean distance asks, "How far apart are these two vectors?"

**Remember:**  
> Smaller Euclidean distance = closer vectors.

---

## 10. What is dot product?

**Answer:** The dot product multiplies corresponding components and adds the results.

```text
A · B = Σ(Aᵢ × Bᵢ)
```

It can also be expressed as:

```text
A · B = ||A|| ||B|| cos(θ)
```

Therefore, dot product considers:

1. Magnitude of A
2. Magnitude of B
3. Direction/alignment

**Remember:**  
> Dot product cares about both magnitude and direction.

---

## 11. What is magnitude?

**Answer:** Magnitude is the length of a vector.

For:

```text
[3, 4]
```

the magnitude is:

```text
√(3² + 4²) = 5
```

Think:

> Magnitude = how long the vector is.

---

## 12. What is distance vs similarity?

### Distance

Asks:

> "How far apart are these two vectors?"

Usually:

> Lower distance = better/closer match.

### Similarity

Asks:

> "How similar or aligned are these two vectors?"

Usually:

> Higher similarity = better match.

| Metric | Better match |
|---|---|
| Euclidean distance | Lower |
| Cosine distance | Lower |
| Cosine similarity | Higher |
| Dot product | Usually higher |

**Remember:**  
> Distance asks "how far?" Similarity asks "how similar?"

---

## 13. What is Top-K?

**Answer:** Top-K means returning the K highest-ranked results from a search.

For example:

```text
n_results = 3
```

could return:

```text
1. Chunk A → best match
2. Chunk B → second-best match
3. Chunk C → third-best match
```

In RAG, these chunks can then be provided to the LLM as context.

**Important:** Top-K means the K best matches according to the retrieval system. It does not guarantee that every result is actually relevant.

**Remember:**  
> Top-K = give me the K best candidates.

---

## 14. What is an embedding model?

**Answer:** An embedding model converts input such as text into numerical vectors that capture semantic relationships.

Typical RAG flow:

```text
Document
   ↓
Chunk
   ↓
Embedding model
   ↓
Vector / embedding
   ↓
Vector database
```

At query time:

```text
User query
   ↓
Same embedding model
   ↓
Query vector
   ↓
Vector search
   ↓
Similar document chunks
```

**Remember:**  
> The embedding model converts text into embeddings, which are represented as vectors.

---

## 15. What is a vector index?

**Answer:** A vector index is a data structure that helps a vector database efficiently find vectors that are close or similar to a query vector.

Imagine:

```text
10 million document embeddings
```

You don't want to naively compare the query with every vector if the index can make nearest-neighbor search much faster.

Common approaches include:

- HNSW
- IVF
- Product Quantization (PQ)

**Interview answer:**

> A vector index makes nearest-neighbor/vector similarity search efficient at scale.

**Remember:**  
> Vector database = stores/searches vectors.  
> Vector index = helps make that search efficient.

---

# Conceptual Questions

## 16. Why isn't a token ID an embedding?

**Answer:** A token ID is simply an integer identifier for a token.

For example:

```text
"cat" → 1234
"dog" → 5678
```

There is no meaningful semantic relationship between the numbers `1234` and `5678`.

An embedding contains learned numerical information that represents semantic relationships.

**Key distinction:**

```text
Token ID
→ "Which token is this?"

Embedding
→ "What does this represent semantically?"
```

---

## 17. Why does increasing vector magnitude affect dot product?

**Answer:** Because dot product depends on magnitude and direction.

```text
A · B = ||A|| ||B|| cos(θ)
```

If you increase the magnitude while keeping the direction the same, the dot product increases.

**Remember:**  
> Dot product = magnitude + direction.

---

## 18. Why doesn't increasing magnitude affect cosine similarity?

**Answer:** Cosine similarity divides by both vector magnitudes:

```text
cos(θ) = (A · B) / (||A|| ||B||)
```

So the magnitude is effectively canceled out.

For example:

```text
A = [1, 1]
B = [2, 2]
```

A and B have different magnitudes, but they point in exactly the same direction.

Therefore:

```text
cosine similarity = 1
```

**Remember:**  
> Cosine similarity focuses on direction/alignment.

---

## 19. Why normalize vectors?

**Answer:** We normalize vectors when we want comparisons to focus on direction/alignment rather than magnitude, or when the chosen retrieval setup benefits from normalized vectors.

Normalization does **not** remove dimensions.

It changes the scale/length of the vector.

Example:

```text
[3, 4]
   ↓ normalize
[0.6, 0.8]
```

The direction is preserved while the magnitude becomes 1.

**Remember:**  
> Normalize when magnitude shouldn't dominate the comparison.

---

## 20. Why must query and documents use the same embedding model?

**Answer:** Because the embedding model defines the vector space and how semantic concepts are represented.

For example:

```text
Document
   ↓
Model A
   ↓
768-dimensional vector
```

If your query uses a completely different model:

```text
Query
   ↓
Model B
   ↓
1536-dimensional vector
```

the representations may belong to different vector spaces and aren't necessarily comparable.

Therefore:

> Query and documents should normally use the same embedding model/version and compatible preprocessing.

**Remember:**  
> Same model → same semantic vector space.

---

## 21. Why does Chroma need a distance metric?

**Answer:** Chroma needs a distance/similarity metric to determine how close a query vector is to stored document vectors.

Examples include:

- Cosine
- Euclidean/L2
- Inner product

The metric is used to rank the retrieved vectors.

Conceptually:

```text
Query vector
     ↓
Compare against stored vectors
     ↓
Calculate distance/similarity
     ↓
Rank results
     ↓
Return Top-K
```

The metric itself doesn't decide whether a result is "worth sending to the LLM." Your retrieval/application logic can make that decision.

**Remember:**  
> The metric tells the database how to compare vectors.

---

## 22. Why does `n_results=2` return two documents?

**Answer:** `n_results=2` is essentially asking the retriever:

> "Give me the top 2 matching results."

Conceptually:

```text
Query
  ↓
Vector search
  ↓
1. Chunk A → best match
2. Chunk B → second-best match
```

One important nuance:

> It doesn't guarantee that both results are relevant. They are simply the two best matches according to the retrieval system.

---

## 23. Why might the top result still be irrelevant?

**Answer:** Embedding similarity is only an approximation of relevance.

Possible causes include:

### 1. Poor chunking

Important information may have been split across chunks, or chunks may be too large/small.

### 2. Weak embedding model

The embedding model may not represent the domain or query particularly well.

### 3. Ambiguous query

The user's question may have multiple possible meanings.

### 4. Vocabulary mismatch

The query and document may describe the same concept using very different terminology.

### 5. Retrieval configuration

The chosen metric, index, or search configuration may not work well for the setup.

### 6. Bad source data

The correct information may simply not exist in the knowledge base.

### 7. Metadata/filtering problems

The search may be retrieving documents from the wrong category, source, date, user, etc.

### 8. K is too small

The relevant chunk might be ranked #5 while you're only retrieving the top 2.

**Strong interview answer:**

> The top result can still be irrelevant because embedding similarity is only an approximation of relevance. Poor chunking, ambiguous queries, weak embeddings, bad data, metadata issues, or retrieval configuration can all cause incorrect results.

---

# The RAG Mental Model

Keep this pipeline in your head.

## Document ingestion

```text
DOCUMENT
   ↓
CHUNKING
   ↓
EMBEDDING MODEL
   ↓
EMBEDDING / VECTOR
   ↓
VECTOR DATABASE
   ↓
VECTOR INDEX
```

## Query time

```text
USER QUERY
   ↓
SAME EMBEDDING MODEL
   ↓
QUERY VECTOR
   ↓
VECTOR SEARCH
   ↓
DISTANCE / SIMILARITY
   ↓
TOP-K CHUNKS
   ↓
LLM
   ↓
ANSWER
```

---

# Tokenization vs RAG Retrieval

Don't mix these two concepts.

## Tokenization

```text
Text
 ↓
Tokenizer
 ↓
Tokens
 ↓
Token IDs
```

This is about how text is represented for a language model.

## RAG retrieval

```text
Document
 ↓
Chunk
 ↓
Embedding model
 ↓
Embedding/vector
 ↓
Vector DB
 ↓
Similarity search
```

This is about finding semantically relevant information.

---

# Key Mathematical Relationships

## Vector magnitude

For:

```text
A = [a₁, a₂, ..., aₙ]
```

L2 magnitude is:

```text
||A||₂ = √(a₁² + a₂² + ... + aₙ²)
```

---

## Dot product

```text
A · B = Σ(Aᵢ × Bᵢ)
```

And:

```text
A · B = ||A|| ||B|| cos(θ)
```

So dot product depends on:

```text
Magnitude + Direction
```

---

## Cosine similarity

```text
cosine similarity =
(A · B) / (||A|| ||B||)
```

So cosine similarity focuses on:

```text
Direction / Alignment
```

---

## Euclidean distance

```text
distance(A,B) =
√Σ(Aᵢ - Bᵢ)²
```

So Euclidean distance focuses on:

```text
How far apart the vectors are
```

---

## Normalization

L2 normalization:

```text
normalized A = A / ||A||₂
```

After normalization:

```text
||normalized A||₂ = 1
```

---

# Quick Daily Review

If you only have 2 minutes on your commute, review these:

### Token ID
> An integer identifier for a token. It has no inherent semantic meaning.

### Embedding
> A numerical representation that captures semantic information.

### Vector
> An ordered collection of numbers representing something in a mathematical space.

### Dimensionality
> The number of values in a vector.

### Magnitude
> The length of a vector.

### Unit vector
> A vector with magnitude 1.

### Normalization
> Scaling a vector, often to magnitude 1, so magnitude has less influence.

### Cosine similarity
> Measures how aligned two vectors are.

### Euclidean distance
> Measures how far apart two vectors are.

### Dot product
> Measures alignment while also being affected by magnitude.

### Top-K
> Return the K highest-ranked retrieval results.

### Embedding model
> Converts input such as text into semantic vectors.

### Vector index
> Makes nearest-neighbor vector search efficient.

### RAG
> Retrieve relevant information first, then give it to the LLM to help generate an answer.

---

# The One Mental Model to Remember

```text
                    RAG
                     │
          ┌──────────┴──────────┐
          │                     │
      Documents              Query
          │                     │
       Chunking                 │
          │                     │
    Embedding Model ◄───────────┘
          │
          ↓
       Vectors
          │
          ↓
    Vector Database
          │
      Vector Index
          │
          ↓
   Distance/Similarity
          │
          ↓
        Top-K
          │
          ↓
     Relevant Chunks
          │
          ↓
          LLM
          │
          ↓
        Answer
```

### Final rule of thumb

> **Token IDs identify tokens. Embeddings represent meaning. Vectors store those representations. Similarity/distance compares them. Top-K retrieves the best candidates. The LLM uses those candidates to answer the user.**
