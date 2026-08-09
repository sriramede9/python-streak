import numpy as np
import time

# -------------------------------------------------------------------
# STEP 1: IMPLEMENT DISTANCE METRICS FROM FIRST PRINCIPLES
# -------------------------------------------------------------------

def l2_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Euclidean (L2) distance formula."""
    return float(np.sqrt(np.sum((u - v) ** 2)))

def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    """Raw Dot Product (Inner Product)."""
    return float(np.sum(u * v))

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine Similarity (Angle comparison independent of magnitude)."""
    norm_u = np.sqrt(np.sum(u ** 2))
    norm_v = np.sqrt(np.sum(v ** 2))
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(dot_product(u, v) / (norm_u * norm_v))

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Scales vector to unit length (|v| = 1.0)."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# -------------------------------------------------------------------
# STEP 2: VERIFY VECTOR BEHAVIORS & MAGNITUDE DISTORTION
# -------------------------------------------------------------------

# Document A: Short text embedding
vec_doc_a = np.array([0.5, 0.8, 0.1, 0.3], dtype=np.float32)

# Document B: Exact same directional topic, but artificially inflated magnitude (e.g. repeated words)
vec_doc_b = vec_doc_a * 3.5

# User Query vector
vec_query = np.array([0.4, 0.9, 0.2, 0.1], dtype=np.float32)

print("=== 1. DISTANCE METRIC COMPARISON ===")
print(f"L2 Distance (Doc A vs Query): {l2_distance(vec_doc_a, vec_query):.4f}")
print(f"L2 Distance (Doc B vs Query): {l2_distance(vec_doc_b, vec_query):.4f}  <-- Penalized heavily for magnitude!")

print(f"\nDot Product (Doc A vs Query): {dot_product(vec_doc_a, vec_query):.4f}")
print(f"Dot Product (Doc B vs Query): {dot_product(vec_doc_b, vec_query):.4f}  <-- Skewed by high magnitude!")

print(f"\nCosine Similarity (Doc A vs Query): {cosine_similarity(vec_doc_a, vec_query):.4f}")
print(f"Cosine Similarity (Doc B vs Query): {cosine_similarity(vec_doc_b, vec_query):.4f}  <-- IDENTICAL directional alignment!")
print("=" * 50 + "\n")

# -------------------------------------------------------------------
# STEP 3: PROOF OF UNIT-NORMALIZED SPEED TRICK
# -------------------------------------------------------------------

norm_query = l2_normalize(vec_query)
norm_doc_a = l2_normalize(vec_doc_a)

cos_sim = cosine_similarity(vec_query, vec_doc_a)
dot_sim = dot_product(norm_query, norm_doc_a)

print("=== 2. NORMALIZATION MATHEMATICAL PROOF ===")
print(f"Cosine Similarity (Unnormalized): {cos_sim:.6f}")
print(f"Dot Product (Normalized)       : {dot_sim:.6f}")
print("Notice: On L2-normalized vectors, Dot Product IS Cosine Similarity!")
print("Production systems normalize vectors up-front to swap expensive division for fast Dot Product.")