import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------------------------
# STEP 1: INITIALIZE VECTOR STORE WITH METADATA PAYLOADS
# -------------------------------------------------------------------
client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

collection = client.create_collection(
    name="episode_005_metadata",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# Multi-domain documents with structured metadata
documents = [
    "The Hurontario LRT connects Cooksville GO to Mississauga Valleys.",
    "Trillium Health Partners hospital expansion target completion is 2027.",
    "Dundas BRT introduces dedicated bus lanes to bypass traffic.",
    "Bloor Street complete street redesign features cycle tracks and trees.",
    "Mary Fix Creek flood mitigation project completed in June 2025."
]

metadatas = [
    {"category": "transit", "year": 2026, "status": "active", "doc_type": "public_plan"},
    {"category": "healthcare", "year": 2027, "status": "active", "doc_type": "internal_report"},
    {"category": "transit", "year": 2026, "status": "active", "doc_type": "public_plan"},
    {"category": "urban_design", "year": 2026, "status": "complete", "doc_type": "public_plan"},
    {"category": "infrastructure", "year": 2025, "status": "complete", "doc_type": "internal_report"}
]

ids = [f"doc_{i}" for i in range(len(documents))]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"✅ Indexed {collection.count()} documents with metadata payloads.\n")

# -------------------------------------------------------------------
# STEP 2: METADATA PRE-FILTERED QUERIES
# -------------------------------------------------------------------

query_text = "What infrastructure or transit project updates are there?"

# Scenario A: Filter strictly for transit documents
print("=== 1. FILTERED SEARCH: category == 'transit' ===")
results_transit = collection.query(
    query_texts=[query_text],
    n_results=5,
    where={"category": "transit"}  # Pre-filter condition
)

for doc, meta in zip(results_transit['documents'][0], results_transit['metadatas'][0]):
    print(f"[{meta['category'].upper()}] {doc}")

# Scenario B: Compound logical filter ($and operator)
print("\n=== 2. COMPOUND FILTER: status == 'active' AND year >= 2026 ===")
results_compound = collection.query(
    query_texts=[query_text],
    n_results=5,
    where={
        "$and": [
            {"status": {"$eq": "active"}},
            {"year": {"$gte": 2026}}
        ]
    }
)

for doc, meta in zip(results_compound['documents'][0], results_compound['metadatas'][0]):
    print(f"[{meta['category']} | {meta['year']}] {doc}")