import time
from typing import Dict, List

import chromadb
import ollama
from chromadb.utils import embedding_functions

# -------------------------------------------------------------------
# STEP 1: INITIALIZE LOCAL VECTOR STORE
# -------------------------------------------------------------------
client = chromadb.Client()
embed_fn = embedding_functions.DefaultEmbeddingFunction()

collection = client.create_collection(
    name="episode_011_live_hyde",
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
print(f"✅ Ingested {len(documents)} documents into vector store.\n")


# -------------------------------------------------------------------
# STEP 2: DYNAMIC HyDE GENERATOR VIA LOCAL OLLAMA
# -------------------------------------------------------------------
def generate_dynamic_hyde(query: str, model_name: str = "llama3.2") -> str:
    """
    Calls local Ollama to write a zero-shot hypothetical document passage.
    """
    system_prompt = (
        "You are an expert technical writer. Write a short, declarative, factual-sounding "
        "excerpt from an official report or encyclopedia that directly answers the user's question. "
        "Do NOT write introductory conversational filler like 'Sure!' or 'Here is...'. "
        "Output ONLY the raw informational paragraph."
    )

    start_time = time.time()
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        options={"temperature": 0.3},  # Low temperature for focused factual style
    )
    elapsed = (time.time() - start_time) * 1000

    passage = response["message"]["content"].strip()
    print(f"⚡ Ollama Generation Time: {elapsed:.2f} ms ({model_name})")
    return passage


# -------------------------------------------------------------------
# STEP 3: COMPARE RAW RETRIEVAL VS. DYNAMIC HyDE
# -------------------------------------------------------------------
user_query = (
    "How long does the 3-Bloor bus take to reach the GO station via dedicated lanes?"
)

print("=" * 65)
print(f"🔎 USER QUERY: '{user_query}'")
print("=" * 65)

# A. Raw Query Search (Query-to-Doc)
raw_results = collection.query(query_texts=[user_query], n_results=2)
print("\n--- 1. DIRECT RAW RETRIEVAL (Query-to-Doc) ---")
for idx, (doc, dist) in enumerate(
    zip(raw_results["documents"][0], raw_results["distances"][0])
):
    print(f"Rank {idx + 1} [Cosine Distance: {dist:.4f}]: {doc}")

# B. Dynamic HyDE (Doc-to-Doc)
print("\n--- 2. GENERATING DYNAMIC HYPOTHETICAL PASSAGE ---")
hypo_passage = generate_dynamic_hyde(user_query, model_name="llama3.2")
print(f'\nGenerated Passage:\n"{hypo_passage}"')

hyde_results = collection.query(query_texts=[hypo_passage], n_results=2)
print("\n--- 3. DYNAMIC HyDE RETRIEVAL (Doc-to-Doc) ---")
for idx, (doc, dist) in enumerate(
    zip(hyde_results["documents"][0], hyde_results["distances"][0])
):
    print(f"Rank {idx + 1} [Cosine Distance: {dist:.4f}]: {doc}")
