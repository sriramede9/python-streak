import time
import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# STEP 1: TOKENIZATION INSPECTION
# -------------------------------------------------------------------
sample_text = "The Hurontario LRT connects Cooksville GO to Mississauga Valleys."

# OpenAI-style BPE tokenizer inspection
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode(sample_text)

print("=== 1. TOKENIZATION ANALYSIS ===")
print(f"Raw Text ({len(sample_text)} chars): '{sample_text}'")
print(f"Token Count: {len(tokens)} tokens")
print(f"Token IDs: {tokens[:8]}...")
print("Decoded Tokens:", [enc.decode([t]) for t in tokens[:8]])
print("=" * 40 + "\n")

# -------------------------------------------------------------------
# STEP 2: COMPARE LOCAL EMBEDDING MODELS
# -------------------------------------------------------------------
# We compare two popular local open-source models:
# 1. all-MiniLM-L6-v2 (Lightweight, 384-dim)
# 2. all-mpnet-base-v2 (Higher accuracy, 768-dim)

print("=== 2. LOADING LOCAL MODELS ===")
model_small = SentenceTransformer("all-MiniLM-L6-v2")
model_large = SentenceTransformer("all-mpnet-base-v2")

documents = [
    "Trillium Health Partners hospital site active construction target 2027.",
    "Dundas BRT dedicated lanes bypass traffic to Cooksville GO station.",
    "Bloor Street complete street redesign features cycle tracks and trees.",
    "Mary Fix Creek flood mitigation project completed in June 2025."
]

# Benchmark Model 1: Small (384D)
start_time = time.time()
embeddings_small = model_small.encode(documents)
small_time = (time.time() - start_time) * 1000

# Benchmark Model 2: Large (768D)
start_time = time.time()
embeddings_large = model_large.encode(documents)
large_time = (time.time() - start_time) * 1000

print(f"MiniLM-L6-v2  -> Dimensions: {embeddings_small.shape[1]} | Encoding Time: {small_time:.2f} ms")
print(f"mpnet-base-v2 -> Dimensions: {embeddings_large.shape[1]} | Encoding Time: {large_time:.2f} ms")
print("=" * 40 + "\n")

# -------------------------------------------------------------------
# STEP 3: CHROMADB INDEX WITH CUSTOM EMBEDDING MODEL
# -------------------------------------------------------------------
client = chromadb.Client()

# Chroma collection using custom embeddings explicitly
collection = client.create_collection(
    name="episode_003_custom_embeds",
    metadata={"hnsw:space": "cosine"}
)

# Add pre-computed 768-dim vectors directly
collection.add(
    documents=documents,
    embeddings=embeddings_large.tolist(),
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"model": "all-mpnet-base-v2"} for _ in documents]
)

# Query using the exact same embedding model
query_str = "When will the hospital construction be finished?"
query_vector = model_large.encode([query_str]).tolist()

results = collection.query(
    query_embeddings=query_vector,
    n_results=1
)

print("=== 3. RETRIEVAL WITH 768-DIM VECTOR ===")
print(f"Query: '{query_str}'")
print(f"Top Result: {results['documents'][0][0]}")
print(f"Cosine Distance Score: {results['distances'][0][0]:.4f}")