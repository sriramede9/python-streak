import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Dataset: Synthetic AI Vector Search Ingestion & Reranking Telemetry
np.random.seed(2026)
n_docs = 1000

df = pd.DataFrame({
    'doc_id': [f"doc_{i:05d}" for i in range(n_docs)],
    'raw_content': np.random.choice([
        "  User guide for API Authentication using OAuth2.0 and JWT tokens.  ",
        "Error log: NullPointerException in ChunkingPipeline.java at line 42.",
        "   ",  # Intentional empty whitespace document
        "Vector database benchmark: Comparing HNSW vs IVF-PQ indexing performance.",
        "Retrieval-Augmented Generation (RAG) architecture patterns for enterprise LLMs.",
        None  # Intentional missing document
    ], size=n_docs),
    'source_type': np.random.choice(['pdf', 'html', 'markdown', 'json_log'], size=n_docs),
    'chunk_size_tokens': np.random.randint(10, 1500, size=n_docs),
    'reranker_score': np.random.uniform(-5.0, 12.0, size=n_docs),
    'is_relevant': np.random.choice([0, 1], p=[0.85, 0.15], size=n_docs)
})

# ==============================================================================
# AI ENGINEERING PRACTICE: TEXT CORPUS CLEANING & RAG VECTOR PREPROCESSING
# ==============================================================================

# Q1 [Corpus Validation & Empty Document Purging]:
# Context: Raw scraped corpora for LLM fine-tuning or RAG ingestion frequently contain blank strings or NaN records.
# Business/ML Purpose: Remove non-informative documents to prevent zero-vector generation and API embedding errors.
# Expected Skill: `.dropna()` on text columns combined with string stripping `.str.strip()` and boolean filtering.
# Task: Create a cleaned DataFrame `clean_corpus_df` by dropping rows where 'raw_content' is NaN or reduces to an empty string after stripping whitespace.
# Your solution:

df['raw_content'] = df['raw_content'].str.strip().replace('', np.nan)
clean_corpus_df = df.dropna(subset=['raw_content'])
clean_corpus_df.info()

# Q2 [Text Normalization & Character Scrubbing]:
# Context: Irregular casing and special characters pollute token vocabularies and degrade vector search precision.
# Business/ML Purpose: Standardize text features prior to BM25 sparse indexing or dense embedding generation.
# Expected Skill: Vectorized string lowercasing `.str.lower()` and regex cleaning `.str.replace()`.
# Task: Create 'cleaned_content' in `clean_corpus_df` where all text is lowercased and any character that is NOT alphanumeric or a space is removed.
# Your solution:

clean_corpus_df['cleaned_content'] = clean_corpus_df['raw_content'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)

# Q3 [RAG Chunking - Token Window Thresholding & Filtering]:
# Context: Embedding models (e.g., text-embedding-3-small) have strict token context limits (e.g., max 512 or 8192 tokens).
# Business/ML Purpose: Filter out oversized chunks that would cause truncation loss or API payload rejection.
# Expected Skill: Boolean condition querying on numerical columns (`df[df['col'] <= threshold]`).
# Task: Filter `clean_corpus_df` to retain only documents where 'chunk_size_tokens' is between 30 and 512 tokens.
# Your solution:
clean_corpus_df['chunk_size_tokens'].between(left=30,right=512).sum()
clean_corpus_df = clean_corpus_df[clean_corpus_df['chunk_size_tokens'].between(left=30,right=512)]

# Q4 [Synthesizing Structured RAG Document Payloads]:
# Context: Vector databases (e.g., Qdrant, Pinecone) require structured metadata alongside raw text chunks.
# Business/ML Purpose: Combine metadata attributes into standardized string representations for embedding models.
# Expected Skill: Vectorized string concatenation (`df['col1'] + ...`).
# Task: Create a column 'document_payload' with the exact format: "Source: " + source_type + " | Content: " + cleaned_content.
# Your solution:

clean_corpus_df['document_payload'] = "Source: " + clean_corpus_df['source_type'] + " | Content: " + clean_corpus_df['cleaned_content']+"."

# Q5 [Log Scale Transformation for Skewed Token Counts]:
# Context: Token length distributions often exhibit extreme right skewness, impacting neural reranker feature normalization.
# Business/ML Purpose: Compress feature variance to ensure stable gradient calculations in multi-modal scoring models.
# Expected Skill: Vectorized logarithmic transformation using `np.log1p()`.
# Task: Create a new column 'log_token_count' calculated as the `np.log1p` transform of 'chunk_size_tokens'.
# Your solution:

clean_corpus_df['log_token_count'] = np.log1p(clean_corpus_df['chunk_size_tokens'])
# Q6 [Reranker Score Standard Normalization (Z-Score)]:
# Context: Raw cross-encoder reranker scores vary across different query loads and temperature settings.
# Business/ML Purpose: Standardize reranker logit distributions to zero mean and unit variance for downstream thresholding.
# Expected Skill: Vectorized Z-score calculation `(x - mean) / std`.
# Task: Calculate the Z-score of 'reranker_score' across the corpus and store it in a new column 'z_reranker_score'.
# Your solution:

mean_reranker_score = clean_corpus_df['reranker_score'].mean()
std_reranker_score = clean_corpus_df['reranker_score'].std()
clean_corpus_df['z_reranker_score'] = (clean_corpus_df['reranker_score'] - mean_reranker_score)/std_reranker_score

# Q7 [High-Cardinality One-Hot Encoding for File Source Types]:
# Context: Categorical metadata ('source_type') improves dense-sparse hybrid retrieval when encoded as tabular features.
# Business/ML Purpose: Generate One-Hot binary vectors without dropping reference categories for complete neural network inputs.
# Expected Skill: `pd.get_dummies()` with `dtype=int` and custom prefixes.
# Task: Generate one-hot encoded columns for 'source_type' using the prefix 'src' and join them back to `clean_corpus_df`.
# Your solution:

clean_corpus_df=pd.get_dummies(clean_corpus_df,columns=['source_type'],prefix='src',dtype=int)
# Q8 [Min-Max Feature Scaling on Continuous Predictors]:
# Context: Merging disparate continuous metrics (token length, normalized logits) requires uniform feature bounds [0.0, 1.0].
# Business/ML Purpose: Prevent high-magnitude features from dominating vector distance or loss function calculations.
# Expected Skill: Min-Max scaling vector implementation `(x - min) / (max - min)`.
# Task: Transform 'z_reranker_score' into a new column 'scaled_reranker_score' bounded between 0.0 and 1.0.
# Your solution:

clean_corpus_df['scaled_reranker_score'] = (clean_corpus_df['z_reranker_score'] - clean_corpus_df['z_reranker_score'].min())/(clean_corpus_df['z_reranker_score'].max() - clean_corpus_df['z_reranker_score'].min())

# Q9 [Stratified Train/Validation Split for Relevance Classifiers]:
# Context: Evaluation sets for retrieval models must maintain identical proportions of relevant vs non-relevant chunks.
# Business/ML Purpose: Prevent class distribution drift between training and validation benchmarks.
# Expected Skill: Stratified sampling via index manipulation or scikit-learn helper integration.
# Task: Partition `clean_corpus_df` into an 80% training set (`train_df`) and 20% validation set (`val_df`), preserving the ratio of 'is_relevant' targets.
# Your solution:
train_df,val_df = train_test_split(clean_corpus_df,test_size=0.20,random_state=42,stratify=clean_corpus_df['is_relevant'])

# Q10 [Dense Float32 Feature Matrix Export for Reranker Models]:
# Context: Exporting cleaned feature data structures into zero-NaN NumPy arrays for PyTorch or XGBoost model pipelines.
# Business/ML Purpose: Ensure structural matrix integrity prior to tensor allocation.
# Expected Skill: Column selection, verification with `.isna().sum()`, and `.to_numpy(dtype=np.float32)` conversion.
# Task: Select numerical features ['log_token_count', 'scaled_reranker_score', 'is_relevant'], verify zero nulls, and extract 2D feature array `X` and 1D target array `y` ('is_relevant').
# Your solution:

features = ['log_token_count', 'scaled_reranker_score']

assert clean_corpus_df[
    features + ['is_relevant']
].isna().sum().sum() == 0

X = clean_corpus_df[features].to_numpy(dtype=np.float32)

y = clean_corpus_df['is_relevant'].to_numpy(dtype=np.float32)

X.shape, y.shape