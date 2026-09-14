import time
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

# -------------------------------------------------------------------
# STEP 1: INITIALIZE STAGE 1 (BI-ENCODER) & STAGE 2 (CROSS-ENCODER)
# -------------------------------------------------------------------
print("Loading models...")
# Stage 1: Fast vector embedding function
bi_encoder_fn = embedding_functions.DefaultEmbeddingFunction()

# Stage 2: Deep Cross-Encoder model (ms-marco-MiniLM-L-6-v2)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Initialize ChromaDB for Stage 1 candidate generation
client = chromadb.Client()
collection = client.create_collection(
    name="episode_008_reranking",
    embedding_function=bi_encoder_fn,
    metadata={"hnsw:space": "cosine"}
)

documents = [
    "The Hurontario LRT project connects Cooksville GO to Mississauga Valleys along an 18km corridor.",
    "Trillium Health Partners hospital expansion phase is under active construction targeting 2027 completion.",
    "Dundas BRT introduces dedicated bus lanes to bypass traffic into Cooksville station.",
    "Bloor Street complete street redesign features cycle tracks, new trees, and pedestrian lighting.",
    "Property tax assessment for 384 Lolita Gardens is recorded at $6,391 per year for 2023.",
    "Cooksville transit hub provides direct GO train connections to downtown Toronto Union Station.",
    "Mary Fix Creek flood mitigation infrastructure was 100% completed in June 2025."
]

doc_ids = [f"doc_{i}" for i in range(len(documents))]
collection.add(documents=documents, ids=doc_ids)

print(f"✅ Indexed {len(documents)} documents in vector store.\n")

# -------------------------------------------------------------------
# STEP 2: STAGE 1 RETRIEVAL (FETCH CANDIDATES)
# -------------------------------------------------------------------
query_text = "What is happening with the new hospital development timeline?"

# Fetch Top-5 candidates (Bi-Encoder search)
stage1_start = time.time()
stage1_results = collection.query(query_texts=[query_text], n_results=5)
stage1_time = (time.time() - stage1_start) * 1000

candidate_docs = stage1_results['documents'][0]

print("=== 1. STAGE 1: VECTOR CANDIDATES (Top-5) ===")
print(f"Retrieval Time: {stage1_time:.2f} ms")
for idx, doc in enumerate(candidate_docs):
    print(f"Rank {idx+1}: {doc}")
print("=" * 50 + "\n")

# -------------------------------------------------------------------
# STEP 3: STAGE 2 RERANKING (CROSS-ENCODER)
# -------------------------------------------------------------------
# Construct (Query, Document) pairs for full cross-attention evaluation
query_doc_pairs = [[query_text, doc] for doc in candidate_docs]

stage2_start = time.time()
raw_scores = cross_encoder.predict(query_doc_pairs)
stage2_time = (time.time() - stage2_start) * 1000

# Pair document strings with raw Cross-Encoder scores and sort descending
reranked_results = sorted(zip(candidate_docs, raw_scores), key=lambda x: x[1], reverse=True)

print("=== 2. STAGE 2: CROSS-ENCODER RERANKED (Top-3) ===")
print(f"Reranking Time: {stage2_time:.2f} ms")
for rank, (doc, score) in enumerate(reranked_results[:3]):
    print(f"Rank {rank+1} [Score: {score:.4f}]: {doc}")