import time
from dataclasses import dataclass
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------------------------
# STEP 1: INITIALIZE VECTOR STORE WITH DIVERSE MULTI-DOMAIN CORPUS
# -------------------------------------------------------------------
client = chromadb.Client()
embed_fn = embedding_functions.DefaultEmbeddingFunction()

collection = client.create_collection(
    name="episode_017_decomposition",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

documents = [
    "The Hurontario Hazel McCallion LRT line spans 18km connecting Port Credit GO to Brampton Gateway Terminal. Testing is underway with phased commissioning continuing along the central corridor.",
    "Trillium Health Partners hospital redevelopment features a new patient care tower at the Peter Gilgan anchor site, with clinical operations planned for late 2027 to 2028.",
    "The Dundas Bus Rapid Transit (BRT) creates dedicated median lanes enabling the 3-Bloor route to bypass Cooksville vehicle traffic.",
    "Under Mississauga R3 zoning regulations, detached residential properties allow secondary legal basement suites subject to parking and fire code standards.",
    "The Mary Fix Creek flood mitigation project completed in June 2025 de-risks regional infrastructure and establishes restored greenway trails."
]

doc_ids = [f"doc_{i}" for i in range(len(documents))]
collection.add(
    documents=documents,
    ids=doc_ids,
    metadatas=[{"doc_id": d} for d in doc_ids]
)

print(f"✅ Indexed {len(documents)} multi-domain documents.\n")

# -------------------------------------------------------------------
# STEP 2: DEFINE SUB-QUESTION DECOMPOSITION DATA STRUCTURES
# -------------------------------------------------------------------

@dataclass
class SubQuery:
    sub_id: str
    target_axis: str
    query_text: str

@dataclass
class SubQueryResult:
    sub_query: SubQuery
    retrieved_chunks: List[str]
    distances: List[float]

# -------------------------------------------------------------------
# STEP 3: DECOMPOSITION ENGINE
# (In production, driven by a structured-output LLM call)
# -------------------------------------------------------------------

def decompose_complex_query(compound_query: str) -> List[SubQuery]:
    """
    Deconstructs compound questions into targeted, independent sub-questions.
    Production Prompt:
    'You are a query planner. Break down the user prompt into independent,
     atomic search queries. Return a structured list of sub-queries.'
    """
    # Deterministic mapping for our lab run; easily replaced with an Ollama call
    if "hospital" in compound_query.lower() and "lrt" in compound_query.lower():
        return [
            SubQuery(
                sub_id="sq_1",
                target_axis="Healthcare Infrastructure",
                query_text="What is the opening timeline for the Trillium hospital expansion?"
            ),
            SubQuery(
                sub_id="sq_2",
                target_axis="Rapid Transit",
                query_text="What is the operational status and route of the Hurontario LRT?"
            )
        ]
    elif "zoning" in compound_query.lower() and "transit" in compound_query.lower():
        return [
            SubQuery(
                sub_id="sq_1",
                target_axis="Municipal Zoning",
                query_text="What are the regulations for R3 residential zoning and secondary suites?"
            ),
            SubQuery(
                sub_id="sq_2",
                target_axis="Transit Access",
                query_text="How does Dundas BRT and LRT improve mobility in Cooksville?"
            )
        ]
    # Fallback: Single query pass-through
    return [SubQuery(sub_id="sq_root", target_axis="General", query_text=compound_query)]

# -------------------------------------------------------------------
# STEP 4: SCATTER-GATHER RETRIEVAL & SYNTHESIS
# -------------------------------------------------------------------

def execute_decomposed_rag(compound_query: str, top_k_per_sub: int = 1):
    print("=" * 65)
    print(f"🧩 COMPOUND USER QUERY:\n\"{compound_query}\"")
    print("=" * 65)

    start_time = time.time()

    # 1. Decompose
    sub_queries = decompose_complex_query(compound_query)
    print(f"\nDecomposed into {len(sub_queries)} targeted sub-queries:")
    for sq in sub_queries:
        print(f"  [{sq.sub_id} | {sq.target_axis}]: \"{sq.query_text}\"")

    # 2. Scatter: Execute parallel/independent retrievals
    sub_results: List[SubQueryResult] = []
    for sq in sub_queries:
        res = collection.query(query_texts=[sq.query_text], n_results=top_k_per_sub)
        sub_results.append(
            SubQueryResult(
                sub_query=sq,
                retrieved_chunks=res['documents'][0],
                distances=res['distances'][0]
            )
        )

    retrieval_time = (time.time() - start_time) * 1000
    print(f"\n⏱️ Scatter-Gather Retrieval Completed in {retrieval_time:.2f} ms")

    # 3. Gather: Build partitioned context block
    partitioned_context = ""
    for item in sub_results:
        partitioned_context += f"\n### Context Axis: {item.sub_query.target_axis}\n"
        for doc, dist in zip(item.retrieved_chunks, item.distances):
            partitioned_context += f"- (Dist: {dist:.4f}) {doc}\n"

    # 4. Generate structured synthesis prompt
    synthesis_prompt = f"""You are a multi-disciplinary research analyst.
Answer the user's multi-part inquiry using the partitioned evidence below.
Ensure each aspect of the question is addressed under its corresponding section.

--- PARTITIONED EVIDENCE ---
{partitioned_context}

--- USER QUESTION ---
{compound_query}

--- STRUCTURED RESPONSE ---"""

    print("\n" + "=" * 65)
    print("🤖 STRUCTURED PROMPT ASSEMBLED FOR LLM")
    print("=" * 65)
    print(synthesis_prompt)

if __name__ == "__main__":
    test_compound_query = "What is the expected opening timeline for the hospital redevelopment, and what is the current progress on the Hurontario LRT?"
    execute_decomposed_rag(test_compound_query, top_k_per_sub=1)