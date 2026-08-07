import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset: Synthetic AI Assistant Interaction & Prompt Engineering Logs
# Simulating production telemetry from an LLM serving pipeline
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'session_id': [f"sess_{i:05d}" for i in range(n_samples)],
    'user_prompt': np.random.choice([
        "How do I optimize my SQL query? Contact me at user@example.com",
        "Summarize this document: https://docs.internal.net/v1",
        "Generate a Python script for RAG retrieval using ChromaDB",
        "My credit card is 4111-2222-3333-4444, check my billing balance",
        "What is the capital of France?",
        "   "  # Intentional empty whitespace prompt
    ], size=n_samples),
    'system_response': np.random.choice([
        "Here is the optimized query plan using indexes...",
        "Here is a concise 3-bullet summary of your document...",
        "import chromadb\nclient = chromadb.Client()...",
        "I cannot handle sensitive payment information.",
        "The capital of France is Paris.",
        None  # Intentional model response failure (null)
    ], size=n_samples),
    'model_name': np.random.choice(['gpt-4o', 'claude-3-5-sonnet', 'llama-3-70b', 'mistral-large'], size=n_samples),
    'latency_ms': np.random.exponential(scale=450, size=n_samples) + 50,
    'user_rating': np.random.choice([1, 2, 3, 4, 5, np.nan], p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1], size=n_samples)
})

# ==============================================================================
# AI ENGINEERING PRACTICE: LLM TELEMETRY & NLP PIPELINE PREPROCESSING
# ==============================================================================

# Q1 [PII Scrubbing & Text Anonymization]:
# Context: Raw user prompts stored in production logs often contain sensitive PII (emails, credit card numbers).
# Business/ML Purpose: Scrub sensitive user information prior to saving logs into public fine-tuning datasets.
# Expected Skill: Vectorized regex replacement (.str.replace) with regular expressions.
# Task: Create 'clean_prompt' by replacing all email addresses (`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`) with '[EMAIL_MASKED]' and card patterns (`\d{4}-\d{4}-\d{4}-\d{4}`) with '[CARD_MASKED]'.
# Your solution:
df['clean_prompt'] = df['user_prompt'].str.replace(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}','[EMAIL_MASKED]',regex=True).str.replace(r'\d{4}-\d{4}-\d{4}-\d{4}','[CARD_MASKED]',regex=True)

# Q2 [Heuristic Token Estimation Feature Engineering]:
# Context: Serving LLMs costs money per token. Estimating token counts locally before sending API calls reduces latency/cost.
# Business/ML Purpose: Engineer input and output token count features using a standard heuristic (~4 characters per token).
# Expected Skill: Vectorized string length calculation (.str.len()) and integer division.
# Task: Create two columns: 'est_input_tokens' (`clean_prompt` char length // 4) and 'est_output_tokens' (`system_response` char length // 4).
# Your solution:
df['est_input_tokens'] = df['clean_prompt'].str.len()/4
df['est_output_tokens'] = df['clean_prompt'].str.len()/4

# Q3 [Handling Null Model Responses & Failed Invocations]:
# Context: LLM API timeouts or guardrail blocks lead to missing (NaN) model responses in telemetry.
# Business/ML Purpose: Filter out incomplete or corrupt prompt-response pairs before fine-tuning dataset generation.
# Expected Skill: .dropna() targeting specific text columns and filtering blank whitespace strings.
# Task: Drop rows where 'system_response' is null OR 'clean_prompt' consists only of empty whitespace.
# Your solution:

df.dropna(subset='system_response',inplace=True)
condition= df['clean_prompt'].str.contains(r'^\s*$',regex=True)
df.loc[condition, 'clean_prompt'] = np.nan
df.dropna(subset='clean_prompt',inplace=True)
df.info()


# Q4 [Latency Outlier Truncation / Capping]:
# Context: Extreme serving latencies (e.g., cold starts, timeout retries) skew performance benchmarks.
# Business/ML Purpose: Cap extreme latency values at the 99th percentile (winsorization) to stabilize training features.
# Expected Skill: .quantile() calculation and np.where vectorized threshold capping.
# Task: Calculate the 99th percentile of 'latency_ms'. Create 'latency_ms_capped' where any value exceeding the 99th percentile is set to the 99th percentile value.
# Your solution:

latency_ms_99_percentile = float(df['latency_ms'].quantile(0.99))
df['latency_ms_capped'] =df['latency_ms']
condition_latency = df['latency_ms_capped'] > latency_ms_99_percentile
df.loc[condition_latency,'latency_ms_capped'] = latency_ms_99_percentile
df.head()
# Q5 [Binary Quality Label Construction]:
# Context: Fine-tuning alignment models (DPO / RLHF) requires clear binary preference signals.
# Business/ML Purpose: Convert noisy 1-5 star user feedback into a clean binary target (1 = High Quality, 0 = Low Quality).
# Expected Skill: Vectorized conditional assignment using np.select or np.where.
# Task: Create a target column 'is_high_quality' where 'user_rating' >= 4 is 1, 'user_rating' <= 2 is 0, and unrated (NaN) rows are assigned 0.
# Your solution:

condition_is_high_quality = df['user_rating'] >=4

df['is_high_quality']=np.where(condition_is_high_quality,1,0)
df['is_high_quality'].value_counts()
df.info()
# Q6 [Context Window Formatting for Vector Database Chunking]:
# Context: RAG embedding pipelines require structured prompt-response blocks formatted into standardized text blocks.
# Business/ML Purpose: Combine user input and model output into a single context document for embedding ingestion.
# Expected Skill: Vectorized string concatenation (`df['col1'] + ...`).
# Task: Create a new column 'rag_chunk' with the format: "User: " + clean_prompt + " | Assistant: " + system_response.
# Your solution:

df['rag_chunk'] = "User: " + df['clean_prompt'] + " | Assistant: " + df['system_response'] +"."

# Q7 [Categorical Encoding for Model Routing]:
# Context: Different LLM backends ('model_name') have distinct performance profiles requiring categorical encoding.
# Business/ML Purpose: Convert categorical model identifiers into one-hot dummy variables for routing models.
# Expected Skill: pd.get_dummies() with `dtype=int` and prefix configuration.
# Task: Generate one-hot encoded columns for 'model_name' with prefix 'model', keeping all dummy columns (no drop_first).
# Your solution:

df=pd.get_dummies(data=df,prefix='model',columns=['model_name'],dtype=int)
df.info()

# Q8 [Min-Max Scaling on Token & Latency Features]:
# Context: Combining token counts and latency metrics into tabular classifiers requires normalized input scales [0, 1].
# Business/ML Purpose: Normalize numerical features to prevent large-scale metrics from dominating gradient descent.
# Expected Skill: Vectorized Min-Max scaling formula `(x - min) / (max - min)`.
# Task: Scale 'est_input_tokens' and 'latency_ms_capped' to a [0, 1] range in new columns 'scaled_tokens' and 'scaled_latency'.
# Your solution:

min_est_input = df['est_input_tokens'].min()
max_est_input = df['est_input_tokens'].max()
diff=max_est_input-min_est_input
df['scaled_tokens'] = round((df['est_input_tokens']- min_est_input)/(diff))
df['scaled_tokens'].value_counts()

min_latency_ms_capped = df['latency_ms_capped'].min()
max_latency_ms_capped = df['latency_ms_capped'].max()
diff_ = max_latency_ms_capped - min_latency_ms_capped
df['scaled_latency'] = round((df['latency_ms_capped']-min_latency_ms_capped)/(diff_))
df['scaled_latency'].value_counts()
# Q9 [Stratified Train/Validation Split for RLHF Datasets]:
# Context: Preference datasets must maintain identical distributions of high/low quality samples across train and validation splits.
# Business/ML Purpose: Partition data into 80% train and 20% validation sets while preserving 'is_high_quality' class ratios.
# Expected Skill: Grouped/Stratified splitting via Pandas index shuffling or scikit-learn integration.
# Task: Deterministically shuffle `df` (random_state=42) and create `train_df` (80%) and `val_df` (20%) stratified by 'is_high_quality'.
# Your solution:

train_Df,value_df=train_test_split(df,random_state=42,stratify=df['is_high_quality'],test_size=0.2)

# Q10 [Dense Float32 NumPy Feature Matrix Export]:
# Context: Exporting tabular features for deep learning or GBDT (LightGBM/XGBoost) model consumption.
# Business/ML Purpose: Extract numerical predictors and target labels into clean, non-null float32 arrays.
# Expected Skill: Selecting numerical columns, checking for zero nulls with .isna().sum(), and .to_numpy(dtype=np.float32).
# Task: Select features ['est_input_tokens', 'est_output_tokens', 'scaled_latency', 'is_high_quality'], verify 0 nulls, and extract 2D array `X` (features) and 1D array `y` ('is_high_quality').
# Your solution:

assert df[['est_input_tokens', 'est_output_tokens', 'scaled_latency', 'is_high_quality']].isna().sum().sum() == 0

X_features=df[['est_input_tokens', 'est_output_tokens', 'scaled_latency', 'is_high_quality']].to_numpy(dtype=np.float32)
Y_features = df['is_high_quality']