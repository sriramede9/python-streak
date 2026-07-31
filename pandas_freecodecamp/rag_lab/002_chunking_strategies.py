import os
import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------------------------
# STEP 1: LOAD RAW UNSTRUCTURED TEXT
# -------------------------------------------------------------------
def load_document(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------------------------------------------
# STEP 2: IMPLEMENT CHUNKING ALGORITHMS FROM SCRATCH
# -------------------------------------------------------------------

def fixed_size_chunking(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    """Splits text into fixed character counts with sliding window overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def recursive_character_chunking(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """
    Recursively splits on semantic boundaries:
    Paragraphs (\n\n) -> Lines (\n) -> Sentences (. ) -> Words ( )
    """
    separators = ["\n\n", "\n", ". ", " "]
    
    def split_text(doc: str, separators: list[str]) -> list[str]:
        if not separators or len(doc) <= chunk_size:
            return [doc]
        
        sep = separators[0]
        splits = doc.split(sep)
        
        final_chunks = []
        current_chunk = ""
        
        for s in splits:
            item = s + (sep if sep != " " else "")
            if len(current_chunk) + len(item) <= chunk_size:
                current_chunk += item
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                # If a single item is bigger than chunk_size, recurse on next separator
                if len(item) > chunk_size:
                    sub_splits = split_text(item, separators[1:])
                    final_chunks.extend(sub_splits)
                    current_chunk = ""
                else:
                    current_chunk = item
                    
        if current_chunk:
            final_chunks.append(current_chunk.strip())
            
        return final_chunks

    raw_chunks = split_text(text, separators)
    
    # Add overlap across recursive chunks
    overlapped_chunks = []
    for i, c in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prefix = raw_chunks[i-1][-overlap:]
            overlapped_chunks.append(f"...{prefix} {c}")
        else:
            overlapped_chunks.append(c)
            
    return overlapped_chunks

# -------------------------------------------------------------------
# STEP 3: RUN COMPARATIVE RAG EXPERIMENT
# -------------------------------------------------------------------

# Load sample document or fallback text
sample_doc = """
The Hurontario LRT project connects Mississauga and Brampton across 18km of dedicated track. 
Active testing is underway, with full revenue service scheduled for late 2026. Once visible trains begin running, property values near transit hubs typically experience a 5-10% lift.

Simultaneously, the Trillium Health Partners hospital expansion represents one of Ontario's largest medical infrastructure projects. 
The final structural beams were raised on the new facility, preparing for an opening in 2027/2028. High-income staff including doctors and specialized nurses will drive massive local housing demand in the Cooksville corridor.

Further mobility improvements include the Dundas Bus Rapid Transit (BRT) line. 
Dedicated bus lanes allow 3-Bloor transit to bypass surface traffic to reach the Cooksville GO station or LRT line in minutes.
"""

print("=== 1. FIXED-SIZE CHUNKS ===")
fixed_chunks = fixed_size_chunking(sample_doc, chunk_size=120, overlap=20)
for i, c in enumerate(fixed_chunks[:3]):
    print(f"Chunk {i+1} ({len(c)} chars): {repr(c)}")

print("\n=== 2. RECURSIVE CHUNKS ===")
rec_chunks = recursive_character_chunking(sample_doc, chunk_size=200, overlap=30)
for i, c in enumerate(rec_chunks[:3]):
    print(f"Chunk {i+1} ({len(c)} chars): {repr(c)}")

# -------------------------------------------------------------------
# STEP 4: INDEX RECURSIVE CHUNKS IN CHROMADB
# -------------------------------------------------------------------
client = chromadb.Client()
fn = embedding_functions.DefaultEmbeddingFunction()
collection = client.create_collection("episode_002_chunks", embedding_function=fn)

collection.add(
    documents=rec_chunks,
    ids=[f"rec_{i}" for i in range(len(rec_chunks))],
    metadatas=[{"chunk_strategy": "recursive", "source": "transit_plan.txt"} for _ in rec_chunks]
)

query = "When is the Trillium hospital expansion expected to open?"
res = collection.query(query_texts=[query], n_results=1)

print("\n=== 3. RETRIEVAL WITH RECURSIVE CHUNKS ===")
print(f"Query: {query}")
print(f"Top Retrieved Chunk: {res['documents'][0][0]}")
print(f"Distance Score: {res['distances'][0][0]:.4f}")