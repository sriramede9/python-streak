import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------------------------
# STEP 1: DEFINE HIERARCHICAL DATA CONTRACTS
# -------------------------------------------------------------------


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    text: str


@dataclass
class ParentDocument:
    parent_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------------
# STEP 2: BUILD HIERARCHICAL SPLITTER & INGESTION
# -------------------------------------------------------------------


class HierarchicalIngestionEngine:
    def __init__(self, parent_chunk_size: int = 600, child_chunk_size: int = 120):
        self.parent_size = parent_chunk_size
        self.child_size = child_chunk_size
        self.doc_store: Dict[str, ParentDocument] = {}  # In-memory Key-Value store

    def _split_into_sentences(self, text: str) -> List[str]:
        # Simple sentence splitter for local experimentation
        raw_sentences = text.replace("\n", " ").split(". ")
        return [s.strip() + "." for s in raw_sentences if s.strip()]

    def process_document(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[ChildChunk]:
        """
        1. Breaks document into large Parent blocks.
        2. Breaks each Parent block into small Child chunks.
        3. Stores Parents in doc_store and returns Child chunks for vector indexing.
        """
        meta = metadata or {}
        # Simulate creating a parent section
        parent_id = f"parent_{uuid.uuid4().hex[:8]}"
        self.doc_store[parent_id] = ParentDocument(
            parent_id=parent_id, text=text.strip(), metadata=meta
        )

        sentences = self._split_into_sentences(text)
        child_chunks = []
        current_child = ""

        for sent in sentences:
            if len(current_child) + len(sent) <= self.child_size:
                current_child += " " + sent
            else:
                if current_child:
                    child_chunks.append(
                        ChildChunk(
                            child_id=f"child_{uuid.uuid4().hex[:8]}",
                            parent_id=parent_id,
                            text=current_child.strip(),
                        )
                    )
                current_child = sent

        if current_child:
            child_chunks.append(
                ChildChunk(
                    child_id=f"child_{uuid.uuid4().hex[:8]}",
                    parent_id=parent_id,
                    text=current_child.strip(),
                )
            )

        return child_chunks


# -------------------------------------------------------------------
# STEP 3: PARENT-DOCUMENT RETRIEVER PIPELINE
# -------------------------------------------------------------------


class ParentDocumentRetriever:
    def __init__(self, ingestion_engine: HierarchicalIngestionEngine):
        self.engine = ingestion_engine
        self.client = chromadb.Client()
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.create_collection(
            name="episode_012_hierarchical",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_corpus(self, sections: List[str]):
        all_children: List[ChildChunk] = []
        for sec in sections:
            children = self.engine.process_document(
                sec, metadata={"domain": "infrastructure"}
            )
            all_children.extend(children)

        # Index ONLY the fine-grained children in the vector database
        self.collection.add(
            documents=[c.text for c in all_children],
            ids=[c.child_id for c in all_children],
            metadatas=[{"parent_id": c.parent_id} for c in all_children],
        )
        print(
            f"✅ Indexed {len(all_children)} small child chunks pointing to {len(self.engine.doc_store)} parent documents.\n"
        )

    def retrieve(self, query: str, top_k_children: int = 3) -> List[ParentDocument]:
        # 1. Vector search over child vectors
        results = self.collection.query(query_texts=[query], n_results=top_k_children)

        retrieved_child_texts = results["documents"][0]
        retrieved_metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        print("--- 🔍 RETRIEVED FINE-GRAINED CHILD CHUNKS ---")
        for i, (txt, meta, dist) in enumerate(
            zip(retrieved_child_texts, retrieved_metadatas, distances)
        ):
            print(
                f'Child {i + 1} [Dist: {dist:.4f} | Parent: {meta["parent_id"]}]:\n  "{txt}"'
            )

        # 2. Resolve and Deduplicate Parent IDs
        parent_ids_seen = set()
        resolved_parents: List[ParentDocument] = []

        for meta in retrieved_metadatas:
            p_id = meta["parent_id"]
            if p_id not in parent_ids_seen:
                parent_ids_seen.add(p_id)
                parent_doc = self.engine.doc_store.get(p_id)
                if parent_doc:
                    resolved_parents.append(parent_doc)

        return resolved_parents


# -------------------------------------------------------------------
# STEP 4: EXECUTION AND CONTEXT ASSEMBLY
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Real-world long-form multi-topic sections
    section_transit = (
        "The Hurontario Hazel McCallion LRT project is an 18-kilometer dedicated rapid transit line "
        "designed to link Port Credit GO station with Brampton Gateway Terminal. Active vehicle track "
        "testing is progressing rapidly, aiming for complete revenue operations by late 2026. "
        "The Dundas Bus Rapid Transit (BRT) works as a complementary corridor featuring dedicated "
        "transit priority lanes. This enables the 3-Bloor bus line to bypass heavy surface vehicle congestion "
        "and arrive at the Cooksville GO mobility hub in under 7 minutes during peak commuter rush."
    )

    section_health = (
        "The Trillium Health Partners hospital expansion represents one of Ontario's largest medical infrastructure initiatives. "
        "The final structural steel beams were hoisted into place for the massive new inpatient care tower. "
        "Clinical operations and specialized patient transfer are slated for late 2027 or early 2028. "
        "This facility will add hundreds of beds and specialized surgical suites, creating extensive housing "
        "demand for physicians and clinical nursing staff across the surrounding Mississauga Valleys neighborhood."
    )

    engine = HierarchicalIngestionEngine(child_chunk_size=110)
    retriever = ParentDocumentRetriever(ingestion_engine=engine)
    retriever.index_corpus([section_transit, section_health])

    user_query = (
        "What is the travel time for the 3-Bloor bus to reach Cooksville station?"
    )
    print(f"Query: '{user_query}'\n")

    parent_results = retriever.retrieve(user_query, top_k_children=3)

    print("\n" + "=" * 65)
    print("🏆 RESOLVED PARENT CONTEXT FOR GENERATION (DEDUPLICATED)")
    print("=" * 65)
    for idx, parent in enumerate(parent_results):
        print(f"\n[Parent {idx + 1} ID: {parent.parent_id}]")
        print(f'Full Text:\n"{parent.text}"')

    # Build prompt with parent context
    context_str = "\n\n".join([f"[{p.parent_id}] {p.text}" for p in parent_results])
    print("\n" + "=" * 65)
    print("🤖 GROUNDED PROMPT DELIVERED TO LLM")
    print("=" * 65)
    print(f"Context:\n{context_str}\n\nQuestion: {user_query}\nAnswer:")
