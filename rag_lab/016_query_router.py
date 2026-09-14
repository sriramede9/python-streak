import time
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# STEP 1: DEFINE ROUTING DESTINATIONS & REFERENCE DATA
# -------------------------------------------------------------------

class RouteDestination(str, Enum):
    DIRECT_LLM = "direct_llm"         # Greetings, creative tasks, general chit-chat
    VECTOR_SEARCH = "vector_search"   # Unstructured knowledge retrieval (LRT, hospital, transit)
    SQL_DATABASE = "sql_database"     # Exact structured lookup (property taxes, unit counts, MLS records)
    WEB_SEARCH = "web_search"         # External or real-time live events

# Reference utterance examples per route
ROUTE_EXAMPLES: Dict[RouteDestination, List[str]] = {
    RouteDestination.DIRECT_LLM: [
        "Hello",
        "Hi there, how are you?",
        "Good morning!",
        "Thanks for your help",
        "Write a short poem about trains",
        "Who are you and what do you do?"
    ],
    RouteDestination.VECTOR_SEARCH: [
        "What are the updates on the Hurontario LRT timeline?",
        "Tell me about the hospital expansion plans and medical tower",
        "How will the Dundas BRT bypass Cooksville congestion?",
        "Details on the Bloor Street redesign cycle tracks",
        "Explain how transit projects impact residential property values"
    ],
    RouteDestination.SQL_DATABASE: [
        "What is the property tax assessment for 384 Lolita Gardens?",
        "Show me the lot dimensions and zoning code for the property",
        "How many total bedrooms and bathrooms are in the detached house?",
        "What was the MLS sale price and closing date in October 2024?",
        "How many total parking spaces are registered for the address?"
    ],
    RouteDestination.WEB_SEARCH: [
        "What is the current traffic right now on Hurontario street?",
        "Live weather forecast for Mississauga this afternoon",
        "Breaking news transit delays today",
        "Current stock price of Metrolinx contractors"
    ]
}

# -------------------------------------------------------------------
# STEP 2: BUILD FAST EMBEDDING-BASED SEMANTIC ROUTER (<5ms)
# -------------------------------------------------------------------

class SemanticRouter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.40):
        print(f"Loading embedding model ({model_name}) for semantic router...")
        self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        self.route_centroids: Dict[RouteDestination, np.ndarray] = {}
        self._build_centroids()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)

    def _build_centroids(self):
        """Precomputes normalized centroid vector for each route category."""
        for route, examples in ROUTE_EXAMPLES.items():
            embeddings = self.model.encode(examples)
            normalized = self._normalize(embeddings)
            # Centroid is the average vector across all example prompts for this route
            centroid = np.mean(normalized, axis=0)
            centroid = self._normalize(centroid)
            self.route_centroids[route] = centroid
        print("✅ Precomputed route centroids for all destinations.\n")

    def route_query(self, query: str) -> Tuple[RouteDestination, float]:
        """Calculates cosine similarity to all centroids and returns best match."""
        q_vec = self.model.encode([query])
        q_norm = self._normalize(q_vec)[0]

        best_route = RouteDestination.VECTOR_SEARCH  # Default fallback
        max_similarity = -1.0

        for route, centroid in self.route_centroids.items():
            # Dot product on unit-normalized vectors equals cosine similarity
            similarity = float(np.dot(q_norm, centroid))
            if similarity > max_similarity:
                max_similarity = similarity
                best_route = route

        # Fallback if confidence is below cutoff threshold
        if max_similarity < self.threshold:
            return RouteDestination.VECTOR_SEARCH, max_similarity

        return best_route, max_similarity

# -------------------------------------------------------------------
# STEP 3: DISPATCHER SIMULATION (EXECUTION HANDLERS)
# -------------------------------------------------------------------

def execute_pipeline(query: str, router: SemanticRouter):
    start = time.time()
    route, confidence = router.route_query(query)
    latency_ms = (time.time() - start) * 1000

    print(f"Query: \"{query}\"")
    print(f"⏱️ Router Decision: {latency_ms:.2f} ms | Route: [{route.value.upper()}] (Score: {confidence:.3f})")

    # Action based on route decision
    if route == RouteDestination.DIRECT_LLM:
        print("  -> Handler: Skipping RAG retrieval. Sending directly to LLM conversational stream.\n")
    elif route == RouteDestination.SQL_DATABASE:
        print("  -> Handler: Extracting entity ID and querying structured SQL metadata table.\n")
    elif route == RouteDestination.VECTOR_SEARCH:
        print("  -> Handler: Triggering Hybrid Search (BM25 + Dense) + Cross-Encoder Reranker.\n")
    elif route == RouteDestination.WEB_SEARCH:
        print("  -> Handler: Dispatching query to real-time external search API.\n")

# -------------------------------------------------------------------
# STEP 4: TEST TESTBED ON VARIED QUERIES
# -------------------------------------------------------------------

if __name__ == "__main__":
    router = SemanticRouter(similarity_threshold=0.35)

    test_queries = [
        "Hey good afternoon!",
        "When is the Trillium health facility scheduled to open?",
        "What are the annual property taxes on 384 Lolita Gardens?",
        "Is there a bus delay right now on Dundas?",
        "Can you explain the transit-oriented community concept for Cooksville?"
    ]

    print("=== 🚦 RUNNING SEMANTIC QUERY ROUTER ===")
    for q in test_queries:
        execute_pipeline(q, router)