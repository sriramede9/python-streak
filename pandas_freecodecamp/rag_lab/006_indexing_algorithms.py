import time
import numpy as np
from sklearn.cluster import KMeans

# Fix random seed for reproducibility
np.random.seed(42)

# -------------------------------------------------------------------
# STEP 1: GENERATE SYNTHETIC VECTOR DATASET (10,000 vectors, 128-dim)
# -------------------------------------------------------------------
NUM_VECTORS = 10000
DIM = 128

# Create random L2-normalized synthetic dataset
dataset = np.random.randn(NUM_VECTORS, DIM).astype(np.float32)
dataset /= np.linalg.norm(dataset, axis=1, keepdims=True) # this is the step doing normalization for all our data pre saving

query = np.random.randn(1, DIM).astype(np.float32)
query /= np.linalg.norm(query, axis=1, keepdims=True)

print(f"✅ Generated dataset of {NUM_VECTORS:,} vectors across {DIM} dimensions.\n")

# -------------------------------------------------------------------
# STEP 2: FLAT INDEX (EXACT BRUTE-FORCE SEARCH)
# -------------------------------------------------------------------
def flat_search(query_vec: np.ndarray, docs: np.ndarray, k: int = 5):
    """Calculates dot product against ALL vectors (O(N) search)."""
    start = time.time()
    scores = np.dot(docs, query_vec.T).squeeze()
    top_k_indices = np.argsort(scores)[::-1][:k]
    search_time = (time.time() - start) * 1000
    return top_k_indices, search_time

flat_indices, flat_time = flat_search(query, dataset, k=5)

print("=== 1. FLAT INDEX (EXACT NEAREST NEIGHBOR) ===")
print(f"Top 5 Indices: {flat_indices}")
print(f"Search Time:   {flat_time:.4f} ms")
print("=" * 45 + "\n")

# -------------------------------------------------------------------
# STEP 3: INVERTED FILE INDEX (IVF) FROM SCRATCH
# -------------------------------------------------------------------
NUM_CLUSTERS = 32  # Number of Voronoi cells (nlist)

print("Building IVF Index (Clustering vector space into 32 cells)...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=5)
cluster_labels = kmeans.fit_predict(dataset)
centroids = kmeans.cluster_centers_

# Build Inverted Index map: cluster_id -> list of document indices
inverted_index = {i: np.where(cluster_labels == i)[0] for i in range(NUM_CLUSTERS)}

def ivf_search(query_vec: np.ndarray, nprobe: int = 2, k: int = 5):
    """
    IVF Search:
    1. Find the 'nprobe' nearest cluster centroids to query.
    2. Search ONLY vectors inside those clusters.
    """
    start = time.time()
    # Find closest centroids
    centroid_scores = np.dot(centroids, query_vec.T).squeeze()
    closest_clusters = np.argsort(centroid_scores)[::-1][:nprobe]
    
    # Gather candidate document indices
    candidate_indices = np.concatenate([inverted_index[c] for c in closest_clusters])
    candidate_docs = dataset[candidate_indices]
    
    # Calculate scores on candidates only
    candidate_scores = np.dot(candidate_docs, query_vec.T).squeeze()
    top_candidate_ranks = np.argsort(candidate_scores)[::-1][:k]
    
    top_k_indices = candidate_indices[top_candidate_ranks]
    search_time = (time.time() - start) * 1000
    
    return top_k_indices, search_time, len(candidate_indices)

# -------------------------------------------------------------------
# STEP 4: EVALUATE RECALL vs LATENCY TRADE-OFF
# -------------------------------------------------------------------
print("=== 2. IVF INDEX SEARCH (ANN) ===")
for nprobe in [1, 2, 8]:
    ivf_indices, ivf_time, candidates_checked = ivf_search(query, nprobe=nprobe, k=5)
    
    # Recall @ 5 = How many of the ground truth Flat top-5 are found in IVF top-5
    recall = len(set(flat_indices).intersection(set(ivf_indices))) / 5.0 * 100
    
    print(f"nprobe={nprobe:2d} | Time: {ivf_time:.4f} ms | Scanned: {candidates_checked:4d}/{NUM_VECTORS} vectors | Recall@5: {recall:.0f}%")