import re
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# STEP 1: LOAD EMBEDDING MODEL & DEFINE MULTI-TOPIC DOCUMENT
# -------------------------------------------------------------------
print("Loading embedding backbone for semantic drift detection...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Check model input limit
print(model.get_sentence_features)
print("Model Max Seq Length:", model.max_seq_length)
# Underlying tokenizer physical capacity
print("Tokenizer Max Length:", model.tokenizer.model_max_length)
print("Truncation Side:     ", model.tokenizer.truncation_side)


# Inspect the Pooling module directly (index 1)
pooling_layer = model[1]
print("Word Embedding Dim:  ", pooling_layer.word_embedding_dimension)
print("Mean Pooling active: ", pooling_layer.pooling_mode_mean_tokens)
print("CLS Pooling active:  ", pooling_layer.pooling_mode_cls_token)
print("Max Pooling active:  ", pooling_layer.pooling_mode_max_tokens)

# Verify silent truncation behavior:
long_text = "transit " * 400
encoded_inputs = model.tokenizer(long_text, truncation=True)
print("Input tokens generated:", len(encoded_inputs["input_ids"]))
# For all-MiniLM-L6-v2, this caps strictly at 256.

# A continuous text body with clear topical shifts
raw_text = (
    "The Hurontario Hazel McCallion LRT project spans an 18-kilometer dedicated rail corridor. "
    "Active vehicle track testing is progressing rapidly with passenger service scheduled for late 2026. "
    "Dedicated light rail vehicles operate along surface rights-of-way to avoid road congestion. "
    "Trillium Health Partners hospital expansion is undergoing extensive structural development. "
    "Final structural steel beams were hoisted into place for the new inpatient care facility. "
    "The hospital expansion is scheduled for patient admissions starting around late 2027. "
    "Cooksville property tax assessments for detached residential homes were adjusted in the recent tax roll. "
    "Annual municipal levies reflect funding formulas for regional infrastructure and school board operations. "
    "Property tax accounts can be managed online through municipal self-service payment portals."
)

# -------------------------------------------------------------------
# STEP 2: IMPLEMENT SEMANTIC CHUNKER FROM FIRST PRINCIPLES
# -------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """Splits raw text into individual sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def compute_cosine_distances(embeddings: np.ndarray) -> list[float]:
    """
    Computes cosine distance (1 - cosine_similarity) between 
    consecutive sentence embeddings (i and i+1).
    """
    distances = []
    # Normalize vectors to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / norms

    for i in range(len(norm_embeddings) - 1):
        cos_sim = np.dot(norm_embeddings[i], norm_embeddings[i + 1])
        cos_dist = 1.0 - float(cos_sim)
        distances.append(cos_dist)
    return distances

def semantic_chunking(
    text: str, 
    threshold_percentile: float = 75.0
) -> tuple[list[str], list[float], float]:
    """
    Groups sentences into cohesive chunks based on semantic drift peaks.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return sentences, [], 0.0

    # 1. Embed every sentence
    embeddings = model.encode(sentences)

    # 2. Compute distance between consecutive sentences
    distances = compute_cosine_distances(embeddings)

    # 3. Calculate dynamic boundary threshold using percentile cutoff
    threshold = float(np.percentile(distances, threshold_percentile))

    # 4. Partition sentences wherever distance exceeds threshold
    chunks = []
    current_chunk = [sentences[0]]

    for i, dist in enumerate(distances):
        if dist > threshold:
            # Semantic boundary detected: seal current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks, distances, threshold

# -------------------------------------------------------------------
# STEP 3: RUN CHUNKING EXPERIMENT & INSPECT BOUNDARIES
# -------------------------------------------------------------------

sentences = split_into_sentences(raw_text)
chunks, distances, threshold = semantic_chunking(raw_text, threshold_percentile=70.0)

print(f"\nDocument parsed into {len(sentences)} total sentences.")
print(f"Calculated Semantic Distance Threshold: {threshold:.4f}\n")

print("=== 1. CONSECUTIVE SENTENCE DISTANCE DRIFT ===")
for i, dist in enumerate(distances):
    marker = "🚨 [SPLIT BOUNDARY]" if dist > threshold else "  "
    print(f"Between Sentences {i+1} & {i+2}: Cosine Distance = {dist:.4f} {marker}")

print("\n=== 2. RESULTING SEMANTIC CHUNKS ===")
for idx, chunk in enumerate(chunks):
    print(f"\n--- Chunk {idx + 1} ({len(chunk.split())} words) ---")
    print(f"\"{chunk}\"")