import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# -------------------------------------------------------------------
# DATA CONTRACTS
# -------------------------------------------------------------------

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    score: float
    retrieval_stage: str

# -------------------------------------------------------------------
# MODULE 1: CHUNKER
# -------------------------------------------------------------------

class RecursiveChunker:
    def __init__(self, chunk_size: int = 250, overlap: int = 40):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str, doc_id_prefix: str = "chunk", metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        separators = ["\n\n", "\n", ". ", " "]
        meta = metadata or {}
        
        def _split(doc: str, seps: List[str]) -> List[str]:
            if not seps or len(doc) <= self.chunk_size:
                return [doc]
            sep = seps[0]
            splits = doc.split(sep)
            chunks, current = [], ""
            for s in splits:
                item = s + (sep if sep != " " else "")
                if len(current) + len(item) <= self.chunk_size:
                    current += item
                else:
                    if current:
                        chunks.append(current.strip())
                    if len(item) > self.chunk_size:
                        chunks.extend(_split(item, seps[1:]))
                        current = ""
                    else:
                        current = item
            if current:
                chunks.append(current.strip())
            return chunks

        raw_chunks = _split(text, separators)
        final_chunks = []
        for i, c in enumerate(raw_chunks):
            chunk_content = f"...{raw_chunks[i-1][-self.overlap:]} {c}" if i > 0 and self.overlap > 0 else c
            final_chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id_prefix}_{i}",
                    text=chunk_content.strip(),
                    metadata=meta
                )
            )
        return final_chunks

# -------------------------------------------------------------------
# MODULE 2: HYBRID RETRIEVER (BM25 + CHROMADB)
# -------------------------------------------------------------------

class HybridRetriever:
    def __init__(self, collection_name: str = "modular_rag"):
        self.client = chromadb.Client()
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self.chunks_map: Dict[str, DocumentChunk] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_chunk_ids: List[str] = []

    def index_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return
        
        # 1. Populate Vector DB
        docs = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metas = [c.metadata for c in chunks]
        
        self.collection.add(documents=docs, ids=ids, metadatas=metas)
        
        # 2. Build local map & BM25 sparse index
        for c in chunks:
            self.chunks_map[c.chunk_id] = c
            
        tokenized_corpus = [c.text.lower().split(" ") for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_chunk_ids = ids
        print(f"📦 Successfully indexed {len(chunks)} chunks across Sparse & Dense indices.")

    def search_hybrid(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[ScoredChunk]:
        # A. Dense Vector Search
        dense_res = self.collection.query(query_texts=[query], n_results=min(top_k, len(self.chunks_map)))
        dense_ids = dense_res['ids'][0]

        # B. Sparse BM25 Search
        tokenized_q = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_q)
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        sparse_ids = [self.bm25_chunk_ids[i] for i in bm25_ranked_indices]

        # C. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        for rank, cid in enumerate(sparse_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + (rank + 1)))
        for rank, cid in enumerate(dense_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        sorted_cids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            ScoredChunk(chunk=self.chunks_map[cid], score=score, retrieval_stage="hybrid_rrf")
            for cid, score in sorted_cids
        ]

# -------------------------------------------------------------------
# MODULE 3: RERANKER & PIPELINE COORDINATOR
# -------------------------------------------------------------------

class ProductionRAGPipeline:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        print("Loading Cross-Encoder model (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def run(self, query: str, candidate_k: int = 8, final_n: int = 2) -> Dict[str, Any]:
        start_time = time.time()
        
        # Stage 1: Hybrid candidate generation
        candidates = self.retriever.search_hybrid(query, top_k=candidate_k)
        
        # Stage 2: Deep Cross-Encoder Reranking
        pairs = [[query, item.chunk.text] for item in candidates]
        raw_scores = self.reranker.predict(pairs)
        
        reranked = [
            ScoredChunk(chunk=item.chunk, score=float(score), retrieval_stage="cross_encoder")
            for item, score in zip(candidates, raw_scores)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)
        top_results = reranked[:final_n]
        
        total_time = (time.time() - start_time) * 1000

        # Stage 3: Build Grounded Prompt
        context_block = "\n".join([f"[{res.chunk.chunk_id}] {res.chunk.text}" for res in top_results])
        grounded_prompt = f"""You are a verified enterprise assistant.
Use ONLY the factual excerpts below to answer the user query.

--- CONTEXT ---
{context_block}

--- QUERY ---
{query}

--- ANSWER ---"""

        return {
            "query": query,
            "latency_ms": round(total_time, 2),
            "top_chunks": top_results,
            "grounded_prompt": grounded_prompt
        }

# -------------------------------------------------------------------
# STEP 4: EXECUTE END-TO-END PIPELINE DEMO
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Sample corpus
    transit_corpus = """
    The Hurontario LRT line is an 18km dedicated rapid transit corridor connecting Cooksville GO with Mississauga Valleys.
    Full testing vehicle operations are active, with complete revenue service targeted for late 2026.
    
    The Trillium Health Partners hospital redevelopment is a multi-phase construction project.
    Final structural steel beams were placed recently, preparing the facility for full clinical opening in 2027/2028.
    
    The Dundas Bus Rapid Transit (BRT) project creates dedicated transit priority lanes along the Dundas corridor.
    This enables the 3-Bloor route to bypass Cooksville surface vehicle congestion directly into transit hubs.
    
    The Bloor Street redesign transforms the road layout into a Complete Street, featuring integrated cycle tracks, upgraded soil cells for trees, and enhanced LED lighting.
    """

    # 1. Ingest & Chunk
    chunker = RecursiveChunker(chunk_size=200, overlap=30)
    document_chunks = chunker.split_text(transit_corpus, doc_id_prefix="infrastructure_2026", metadata={"source": "regional_plan"})

    # 2. Index into Hybrid Engine
    retriever = HybridRetriever(collection_name="capstone_phase_1")
    retriever.index_chunks(document_chunks)

    # 3. Instantiate and Execute Pipeline
    pipeline = ProductionRAGPipeline(retriever=retriever)
    
    query = "What transit updates are planned for the Cooksville and Dundas corridors?"
    output = pipeline.run(query=query, candidate_k=6, final_n=2)

    print("\n" + "=" * 60)
    print(f"🔍 QUERY: {output['query']}")
    print(f"⏱️ PIPELINE LATENCY: {output['latency_ms']} ms")
    print("=" * 60)
    print("\n🏆 TOP RERANKED CHUNKS:")
    for idx, item in enumerate(output["top_chunks"]):
        print(f"\n[{idx+1}] Score: {item.score:.4f} | ID: {item.chunk.chunk_id}")
        print(f"    Text: {item.chunk.text}")
        print(f"    Meta: {item.chunk.metadata}")

    print("\n" + "=" * 60)
    print("🤖 FINAL GROUNDED PROMPT:")
    print("=" * 60)
    print(output["grounded_prompt"])
