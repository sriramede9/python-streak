import time
from typing import List, Dict, Tuple
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

# -------------------------------------------------------------------
# STEP 1: DEFINE UNCONTEXTUALIZED RAW CHUNKS (THE AMBIGUITY PROBLEM)
# -------------------------------------------------------------------
full_document = """
TRANSIT & INFRASTRUCTURE REPORT 2026: PEEL REGION & MISSISSAUGA
Section 1: Hurontario Hazel McCallion LRT
The light rail system spans an 18-kilometer dedicated right-of-way connecting Brampton to Port Credit. Active vehicle testing is ongoing with full passenger revenue service targeted for late 2026.

Section 2: Dundas Bus Rapid Transit Corridor
The project introduces dedicated median bus lanes along Dundas Street. The 3-Bloor route will utilize these lanes to bypass Cooksville vehicle congestion, allowing transit riders to reach the mobility hub in under 7 minutes.

Section 3: Regional Hospital Infrastructure Expansion
Trillium Health Partners is currently completing structural development of the new inpatient surgical care tower. The facility is scheduled for clinical opening in late 2027 or early 2028, adding substantial capacity to the Cooksville area.
"""

# Chunks split without document-level context (Notice pronoun & missing entity ambiguities)
raw_chunks = [
    "Active vehicle testing is ongoing with full passenger revenue service targeted for late 2026.",
    "The 3-Bloor route will utilize these lanes to bypass vehicle congestion, allowing transit riders to reach the mobility hub in under 7 minutes.",
    "The facility is scheduled for clinical opening in late 2027 or early 2028, adding substantial capacity to the local area."
]

# -------------------------------------------------------------------
# STEP 2: CONTEXTUAL PREFIX GENERATOR
# (In production, run this via local Ollama or OpenAI at ingestion)
# -------------------------------------------------------------------
def generate_situating_context(full_doc: str, chunk: str) -> str:
    """
    Simulates prompt:
    'Given the full document: <doc>{full_doc}</doc>
     Provide a short 1-2 sentence context to situate this chunk within the overall document: <chunk>{chunk}</chunk>'
    """
    situating_prompts = {
        0: "This chunk refers to the Hurontario Hazel McCallion LRT project testing timeline in Mississauga.",
        1: "This chunk describes the Dundas Bus Rapid Transit (BRT) corridor travel time improvements for the 3-Bloor route connecting to Cooksville GO station.",
        2: "This chunk details the opening timeline and capacity expansion of the Trillium Health Partners hospital facility in Cooksville."
    }
    # Return mock or deterministic situating prefix
    for idx, raw in enumerate(raw_chunks):
        if raw == chunk:
            return situating_prompts[idx]
    return "This chunk is an excerpt from the Peel Region Transit and Infrastructure Report 2026."

contextualized_chunks = [
    f"{generate_situating_context(full_document, c)}\n\n{c}"
    for c in raw_chunks
]

print("=== 1. CHUNK COMPARISON ===")
print("[RAW CHUNK 3]:")
print(f"\"{raw_chunks[2]}\"\n")
print("[CONTEXTUALIZED CHUNK 3]:")
print(f"\"{contextualized_chunks[2]}\"\n")
print("=" * 60)

# -------------------------------------------------------------------
# STEP 3: DUAL INDEXING BENCHMARK (RAW vs CONTEXTUALIZED)
# -------------------------------------------------------------------
client = chromadb.Client()
embed_fn = embedding_functions.DefaultEmbeddingFunction()

# Collection A: Raw Chunks
col_raw = client.create_collection("ep013_raw", embedding_function=embed_fn)
col_raw.add(
    documents=raw_chunks,
    ids=[f"raw_{i}" for i in range(len(raw_chunks))]
)

# Collection B: Contextualized Chunks
col_context = client.create_collection("ep013_contextual", embedding_function=embed_fn)
col_context.add(
    documents=contextualized_chunks,
    ids=[f"ctx_{i}" for i in range(len(contextualized_chunks))]
)

# BM25 Sparse Indices
bm25_raw = BM25Okapi([c.lower().split(" ") for c in raw_chunks])
bm25_context = BM25Okapi([c.lower().split(" ") for c in contextualized_chunks])

# -------------------------------------------------------------------
# STEP 4: RETRIEVAL ACCURACY TEST
# -------------------------------------------------------------------
# Ambiguous query containing entity names NOT in the raw chunk text
user_query = "What is the timeline for the Trillium hospital expansion?"

print(f"\n🔎 USER QUERY: '{user_query}'\n")

# A. Raw Chunk Retrieval
print("--- A. RAW CHUNK RETRIEVAL ---")
raw_vec_res = col_raw.query(query_texts=[user_query], n_results=1)
raw_bm25_scores = bm25_raw.get_scores(user_query.lower().split(" "))

print(f"Dense Vector Top Hit (Distance: {raw_vec_res['distances'][0][0]:.4f}):")
print(f"  \"{raw_vec_res['documents'][0][0]}\"")
print(f"BM25 Scores for Target Chunk (Chunk 3): {raw_bm25_scores[2]:.4f} (Tokens missing!)")

# B. Contextualized Chunk Retrieval
print("\n--- B. CONTEXTUALIZED CHUNK RETRIEVAL ---")
ctx_vec_res = col_context.query(query_texts=[user_query], n_results=1)
ctx_bm25_scores = bm25_context.get_scores(user_query.lower().split(" "))

print(f"Dense Vector Top Hit (Distance: {ctx_vec_res['distances'][0][0]:.4f}):")
print(f"  \"{ctx_vec_res['documents'][0][0]}\"")
print(f"BM25 Scores for Target Chunk (Chunk 3): {ctx_bm25_scores[2]:.4f} (Keywords matched!)")