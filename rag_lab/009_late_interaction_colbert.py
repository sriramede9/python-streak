import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# -------------------------------------------------------------------
# STEP 1: LOAD TRANSFORMER MODEL FOR TOKEN-LEVEL EMBEDDINGS
# -------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading transformer backbone ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_token_embeddings(text: str) -> tuple[list[str], torch.Tensor]:
    """
    Passes text through transformer to extract token-level vector matrix.
    Returns (token_strings, matrix of shape [num_tokens, hidden_dim]).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract last_hidden_state: Shape [1, seq_len, 384]
    embeddings = outputs.last_hidden_state.squeeze(0)
    
    # Normalize vectors along dimension 1 (L2 normalization)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Convert token IDs to human-readable strings
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, embeddings

# -------------------------------------------------------------------
# STEP 2: IMPLEMENT THE MAXSIM OPERATOR FROM FIRST PRINCIPLES
# -------------------------------------------------------------------

def colbert_maxsim(query_embeds: torch.Tensor, doc_embeds: torch.Tensor) -> float:
    """
    Calculates ColBERT MaxSim score between Query and Document token matrices.
    
    Query Matrix: [Q_len, D]
    Doc Matrix  : [D_len, D]
    
    1. Similarity Matrix [Q_len, D_len] = Query @ Doc.T
    2. Max per Query token = max along dim 1
    3. Sum across Query tokens = sum()
    """
    # 1. Pairwise cosine similarity matrix between all query tokens and doc tokens
    sim_matrix = torch.matmul(query_embeds, doc_embeds.T)  # Shape: [Q_len, D_len]
    
    # 2. For each query token, take maximum score against any document token
    max_sim_per_query_token, _ = torch.max(sim_matrix, dim=1)  # Shape: [Q_len]
    
    # 3. Sum max scores to produce total Late Interaction score
    total_score = torch.sum(max_sim_per_query_token).item()
    return total_score

# -------------------------------------------------------------------
# STEP 3: RUN LATE INTERACTION RETRIEVAL DEMO
# -------------------------------------------------------------------

documents = [
    "The Hurontario LRT connects Cooksville GO to Mississauga Valleys.",
    "Trillium Health Partners hospital expansion completes around 2027.",
    "Dundas BRT dedicated bus lanes bypass traffic into Cooksville.",
    "Property tax assessment for 384 Lolita Gardens is $6,391 per year."
]

print("\nIngesting and generating token-level embeddings for corpus...")
doc_matrices = []
for doc in documents:
    _, doc_emb = get_token_embeddings(doc)
    doc_matrices.append(doc_emb)

print(f"✅ Pre-computed token matrices for {len(documents)} documents.")

query = "When is the hospital project finishing?"
q_tokens, q_emb = get_token_embeddings(query)

print(f"\nQuery: '{query}'")
print(f"Query Tokens ({len(q_tokens)}): {q_tokens}")

print("\n=== 🏆 COLBERT MAXSIM SCORES ===")
scores = []
start_time = time.time()
for idx, doc_emb in enumerate(doc_matrices):
    score = colbert_maxsim(q_emb, doc_emb)
    scores.append((documents[idx], score))
elapsed = (time.time() - start_time) * 1000

# Sort results by MaxSim score descending
scores.sort(key=lambda x: x[1], reverse=True)

for rank, (doc, score) in enumerate(scores):
    print(f"Rank {rank+1} [MaxSim Score: {score:.4f}]: {doc}")

print(f"\nMaxSim Execution Time across corpus: {elapsed:.2f} ms")