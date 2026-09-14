import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------------------------
# STEP 1: DATA STRUCTURES FOR PROPOSITIONS
# -------------------------------------------------------------------


@dataclass
class Proposition:
    prop_id: str
    parent_id: str
    statement: str


@dataclass
class Passage:
    passage_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------------
# STEP 2: PROPOSITION EXTRACTION & DE-CONTEXTUALIZATION
# (In production, this is executed by an LLM like Llama-3.2 or Mistral)
# -------------------------------------------------------------------

# Raw complex passage containing multiple distinct facts
sample_passage = (
    "The Hurontario Hazel McCallion LRT is an 18-kilometer dedicated rapid transit line "
    "connecting Port Credit GO with Brampton Gateway Terminal. Active vehicle track testing is "
    "progressing toward revenue service in late 2026. Along the route, property values near "
    "completed stations historically see a 5% to 10% appreciation once trains run visibly."
)


def extract_propositions(passage_text: str, parent_id: str) -> List[Proposition]:
    """
    Simulates decomposing a complex paragraph into atomic, self-contained statements:
    1. Each proposition contains a single fact.
    2. All pronouns and references are resolved.
    """
    # Production prompt: "Extract atomic, self-contained declarative facts from the following text..."
    atomic_statements = [
        "The Hurontario Hazel McCallion LRT spans an 18-kilometer dedicated rapid transit corridor.",
        "The Hurontario Hazel McCallion LRT connects Port Credit GO station with Brampton Gateway Terminal.",
        "Active vehicle testing on the Hurontario LRT targets complete revenue service in late 2026.",
        "Property values near completed transit stations historically appreciate between 5% and 10% once trains are operational.",
    ]

    return [
        Proposition(
            prop_id=f"prop_{uuid.uuid4().hex[:8]}", parent_id=parent_id, statement=stmt
        )
        for stmt in atomic_statements
    ]


# -------------------------------------------------------------------
# STEP 3: MULTI-VECTOR RETRIEVER WITH PARENT RESOLUTION
# -------------------------------------------------------------------


class PropositionalRetriever:
    def __init__(self):
        self.client = chromadb.Client()
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.create_collection(
            name="episode_015_propositions",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.passage_store: Dict[str, Passage] = {}

    def index_passage(self, text: str, metadata: Dict[str, Any] = None):
        p_id = f"parent_{uuid.uuid4().hex[:8]}"
        self.passage_store[p_id] = Passage(
            passage_id=p_id, text=text, metadata=metadata or {}
        )

        # Extract atomic propositions
        propositions = extract_propositions(text, p_id)

        # Index each proposition as an individual vector
        self.collection.add(
            documents=[p.statement for p in propositions],
            ids=[p.prop_id for p in propositions],
            metadatas=[{"parent_id": p.parent_id} for p in propositions],
        )
        print(
            f"✅ Ingested passage '{p_id}' with {len(propositions)} indexed proposition vectors."
        )

    def search(
        self, query: str, top_k_props: int = 2
    ) -> List[tuple[Passage, Proposition, float]]:
        results = self.collection.query(query_texts=[query], n_results=top_k_props)

        retrieved_statements = results["documents"][0]
        retrieved_metas = results["metadatas"][0]
        distances = results["distances"][0]
        prop_ids = results["ids"][0]

        matches = []
        for pid, stmt, meta, dist in zip(
            prop_ids, retrieved_statements, retrieved_metas, distances
        ):
            parent_doc = self.passage_store[meta["parent_id"]]
            matched_prop = Proposition(
                prop_id=pid, parent_id=meta["parent_id"], statement=stmt
            )
            matches.append((parent_doc, matched_prop, dist))
        return matches


# -------------------------------------------------------------------
# STEP 4: RUN RETRIEVAL TEST
# -------------------------------------------------------------------

if __name__ == "__main__":
    retriever = PropositionalRetriever()
    retriever.index_passage(
        sample_passage, metadata={"domain": "transit_infrastructure"}
    )

    # Specific targeted query
    query = "How much do properties appreciate when the trains start operating?"
    print(f"\n🔎 QUERY: '{query}'\n")

    hits = retriever.search(query, top_k_props=2)

    print("=== 🎯 PROPOSITIONAL VECTOR HITS ===")
    for rank, (parent, prop, dist) in enumerate(hits):
        print(f"\n[Rank {rank + 1}] Cosine Distance: {dist:.4f}")
        print(f'Matched Proposition: "{prop.statement}"')
        print(f"Resolved Parent ID : {parent.passage_id}")
        print(f'Parent Context Text: "{parent.text[:120]}..."')
