import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

# -------------------------------------------------------------------
# STEP 1: PREPARE CORPUS & BUILD DUAL INDICES (SPARSE + DENSE)
# -------------------------------------------------------------------
documents = [
    "The Hurontario LRT project (MLS zone L5A) connects Cooksville GO to Mississauga Valleys.",
    "Trillium Health Partners hospital expansion phase active site with final structural beams rising for 2027.",
    "The Dundas BRT corridor features dedicated bus lanes to bypass traffic into Cooksville.",
    "The Bloor Street redesign transforms the streetscape with new cycle tracks and trees.",
    "Property tax assessment for 384 Lolita Gardens is recorded at $6,391 per year."
]

doc_ids = [f"doc_{i}" for i in range(len(documents))]

# --- A. SPARSE INDEX (BM25) ---
# Tokenize documents by splitting words
tokenized_corpus = [doc.lower().split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# BM25 = TF [Term Frequency] + IDF [Inverse Document Frequency] + length normalization

#Does this document contain my terms, 
# how important are those terms, 
# how often do they occur,
#  and how much should document length matter?

# --- B. DENSE INDEX (ChromaDB) ---
client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = client.create_collection(
    name="episode_007_hybrid",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    documents=documents,
    ids=doc_ids,
    metadatas=[{"source": "local_rag_lab"} for _ in documents]
)

print(f"✅ Indexed {len(documents)} documents across both BM25 and ChromaDB.\n")

# -------------------------------------------------------------------
# STEP 2: IMPLEMENT RECIPROCAL RANK FUSION (RRF)
# -------------------------------------------------------------------
def reciprocal_rank_fusion(sparse_results: list[str], dense_results: list[str], k: int = 60, top_n: int = 3) -> list[tuple[str, float]]:
    """
    Fuses two ranked lists using Reciprocal Rank Fusion (RRF).
    Scores docs based on 1 / (k + rank).
    """
    rrf_scores = {}
    
    # Process sparse ranking list
    for rank, doc in enumerate(sparse_results):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
        rrf_scores[doc] += 1.0 / (k + (rank + 1))
        
    # Process dense ranking list
    for rank, doc in enumerate(dense_results):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
        rrf_scores[doc] += 1.0 / (k + (rank + 1))
        
    # Sort by combined RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]

# -------------------------------------------------------------------
# STEP 3: EXECUTE HYBRID QUERY
# -------------------------------------------------------------------
query_text = "What is the tax amount for 384 Lolita Gardens?"

print(f"Query: '{query_text}'\n")

# 1. Sparse Search (BM25)
tokenized_query = query_text.lower().split(" ")
bm25_scores = bm25.get_scores(tokenized_query)
bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
sparse_ranked_docs = [documents[i] for i in bm25_top_indices]

# 2. Dense Search (Vector Cosine Distance)
dense_results = collection.query(query_texts=[query_text], n_results=len(documents))
dense_ranked_docs = dense_results['documents'][0]

# 3. Fuse via RRF
fused_results = reciprocal_rank_fusion(sparse_ranked_docs, dense_ranked_docs, k=60, top_n=3)

print("=== 🏆 HYBRID RRF FUSED RESULTS ===")
for rank, (doc, score) in enumerate(fused_results):
    print(f"Rank {rank+1} [RRF Score: {score:.4f}]: {doc}")