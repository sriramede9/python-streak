import time
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------------------------
# STEP 1: INITIALIZE VECTOR STORE WITH DECLARATIVE CORPUS
# -------------------------------------------------------------------
client = chromadb.Client()
embed_fn = embedding_functions.DefaultEmbeddingFunction()

collection = client.create_collection(
    name="episode_011_hyde",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"},
)

documents = [
    "The Dundas Bus Rapid Transit (BRT) creates dedicated transit priority lanes along Dundas Street, allowing the 3-Bloor bus to bypass automobile congestion and connect directly to Cooksville GO station in under 7 minutes.",
    "Trillium Health Partners hospital redevelopment features a new multi-storey patient tower with modern surgical suites scheduled for clinical occupancy by late 2027.",
    "The Hurontario Hazel McCallion LRT operates 18km of dedicated surface rail connecting Brampton Gateway Terminal down to Port Credit GO.",
    "The City of Mississauga Complete Streets program on Bloor Street installs separated bi-directional cycle tracks and permeable soil tree trenches.",
    "Municipal property tax assessments in Mississauga Valleys (MLS zone L5A) reflect annual municipal and regional infrastructure levies.",
]

doc_ids = [f"chunk_{i}" for i in range(len(documents))]
collection.add(documents=documents, ids=doc_ids)
print(f"✅ Ingested {len(documents)} declarative documents into vector store.\n")

# -------------------------------------------------------------------
# STEP 2: SIMULATED / LOCAL LLM QUERY TRANSFORMATION GENERATORS
# (In production, replace these functions with an Ollama or API call)
# -------------------------------------------------------------------


def generate_hypothetical_document(query: str) -> str:
    """
    Simulates a zero-shot LLM generating a hypothetical passage (HyDE).
    Notice how it mirrors the tone, length, and style of real documents.
    """
    # Prompt template used in production LLMs:
    # "Write a concise, factual paragraph answering the following question as if it were an encyclopedia excerpt: {query}"
    hyde_passages = {
        "travel time 3-Bloor bus to Cooksville GO": (
            "The 3-Bloor bus route utilizes dedicated rapid bus lanes along the Dundas BRT corridor "
            "to bypass local traffic bottlenecks, providing rapid direct transit access to Cooksville GO "
            "station within several minutes during peak travel hours."
        ),
        "hospital opening date": (
            "The newly expanded hospital medical facility and inpatient surgical care tower is slated "
            "to complete structural development and open for patient admission in late 2027 or 2028."
        ),
    }
    # Fallback to query if mock not matched
    return hyde_passages.get(
        query,
        f"A detailed technical report regarding {query} with operational specifications.",
    )


def generate_multi_query_expansions(query: str) -> List[str]:
    """
    Generates 3 variations/paraphrases of a query to overcome lexical phrasing bias.
    """
    # In production: Prompt LLM: "Generate 3 different search variations of: {query}"
    if "3-Bloor" in query or "Cooksville" in query:
        return [
            query,
            "Dundas BRT dedicated bus lanes travel time to Cooksville",
            "Mississauga transit 3-Bloor congestion bypass speed",
        ]
    return [
        query,
        f"Detailed specifications of {query}",
        f"Overview and timeline of {query}",
    ]


# -------------------------------------------------------------------
# STEP 3: BENCHMARK RAW QUERY vs. HyDE RETRIEVAL
# -------------------------------------------------------------------

user_raw_query = "travel time 3-Bloor bus to Cooksville GO"

print("=" * 60)
print(f"🔎 RAW USER QUERY: '{user_raw_query}'")
print("=" * 60)

# A. Standard Raw Query Search (Query-to-Doc)
raw_results = collection.query(query_texts=[user_raw_query], n_results=2)
print("\n--- 1. DIRECT RAW RETRIEVAL (Query-to-Doc) ---")
for idx, (doc, dist) in enumerate(
    zip(raw_results["documents"][0], raw_results["distances"][0])
):
    print(f"Rank {idx + 1} [Cosine Distance: {dist:.4f}]: {doc}")

# B. HyDE Retrieval (Doc-to-Doc)
hypo_doc = generate_hypothetical_document(user_raw_query)
print(f"\n--- 2. GENERATED HYPOTHETICAL DOCUMENT (HyDE) ---")
print(f"'{hypo_doc}'")

hyde_results = collection.query(query_texts=[hypo_doc], n_results=2)
print(f"\n--- 3. HyDE RETRIEVAL (Doc-to-Doc) ---")
for idx, (doc, dist) in enumerate(
    zip(hyde_results["documents"][0], hyde_results["distances"][0])
):
    print(f"Rank {idx + 1} [Cosine Distance: {dist:.4f}]: {doc}")

# -------------------------------------------------------------------
# STEP 4: MULTI-QUERY EXPANSION WITH RECIPROCAL RANK FUSION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("🔀 MULTI-QUERY EXPANSION + RRF FUSION")
print("=" * 60)

expanded_queries = generate_multi_query_expansions(user_raw_query)
print(f"Expanded Queries: {expanded_queries}\n")

rrf_scores: Dict[str, float] = {}
k_rrf = 60

for q in expanded_queries:
    res = collection.query(query_texts=[q], n_results=3)
    retrieved = res["documents"][0]
    for rank, doc in enumerate(retrieved):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + (1.0 / (k_rrf + (rank + 1)))

fused_rankings = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

for rank, (doc, score) in enumerate(fused_rankings[:3]):
    print(f"Fused Rank {rank + 1} [RRF Score: {score:.5f}]: {doc}")
