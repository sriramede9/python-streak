import os

import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------------------------
# STEP 1: INITIALIZE LOCAL VECTOR STORE & EMBEDDING MODEL
# -------------------------------------------------------------------
# We use ChromaDB in-memory mode for rapid local iteration.
client = chromadb.Client()

# Default embedding model: all-MiniLM-L6-v2 (runs 100% locally on CPU)
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

collection = client.create_collection(
    name="rag_foundation", embedding_function=embedding_fn
)


# -------------------------------------------------------------------
# STEP 2: DOCUMENT INGESTION (RAW TEXT TO VECTORS)
# -------------------------------------------------------------------
raw_documents = [
    "The Hurontario LRT is an 18km light rail line connecting Mississauga and Brampton.",
    "Trillium Health Partners is building a major hospital expansion scheduled for completion around 2027.",
    "The Dundas BRT project introduces dedicated bus lanes to bypass traffic into Cooksville GO station.",
    "The Bloor Street redesign transforms the corridor with cycle tracks, upgraded trees, and new lighting.",
]

doc_ids = [f"doc_{i}" for i in range(len(raw_documents))]
metadatas = [{"source": "transit_plan_2026"} for _ in raw_documents]

# Chroma converts these texts to 384-dimensional vectors under the hood
collection.add(documents=raw_documents, ids=doc_ids, metadatas=metadatas)

print(f"✅ Ingested {collection.count()} documents into local vector store.\n")

# -------------------------------------------------------------------
# STEP 3: RETRIEVAL (VECTOR SEARCH)
# -------------------------------------------------------------------
user_query = "When is the new hospital expected to finish construction?"
top_k = 2

results = collection.query(query_texts=[user_query], n_results=top_k)

retrieved_docs = results["documents"][0]
distances = results["distances"][0]

print("=== 🔍 RETRIEVAL RESULTS ===")
for idx, (doc, dist) in enumerate(zip(retrieved_docs, distances)):
    print(f"Rank {idx + 1} [Distance: {dist:.4f}]: {doc}")
print("=" * 30 + "\n")

user_query2="Tell me about healthcare infrastructure timelines"
results2=collection.query(query_texts=[user_query2],n_results=top_k)

for index,(doc,dist) in enumerate(zip(results2["documents"][0],results2["distances"][0])):
    print(f"Rank {index+1} [Distance :{dist:.4f}]: {doc}")

user_query3="why does it rain in winter in Australia?"    

results3=collection.query(query_texts=[user_query3],n_results=top_k)

for index , (doc,distance) in enumerate(zip(results3["documents"][0],results3["distances"][0])):
    print(f" Rank : {index+1} Distance : {distance:.4f} and {doc}")

# -------------------------------------------------------------------
# STEP 4: PROMPT SYNTHESIS
# -------------------------------------------------------------------
context_block = "\n".join([f"- {d}" for d in retrieved_docs])

system_prompt = f"""You are a precise AI research assistant.
Answer the user question using ONLY the provided context snippets below.
If the context does not contain the answer, say "I cannot answer based on the context provided."

--- CONTEXT ---
{context_block}

--- QUESTION ---
{user_query}

--- ANSWER ---
"""

print("=== 🤖 GROUNDED PROMPT READY FOR LLM ===")
print(system_prompt)
