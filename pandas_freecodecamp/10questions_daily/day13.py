import pandas as pd
import numpy as np

# Dataset: Synthetic LLM Alignment & Fine-Tuning Evaluation Corpus
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'eval_id': [f"eval_{i:05d}" for i in range(n_samples)],
    'prompt_text': np.random.choice([
        "Explain quantum computing in simple terms for a high school student.",
        "Refactor this Python script to use asyncio: import time...",
        "   ",  # Intentional whitespace prompt
        "Write a SQL query to join orders and customers with a GROUP BY clause.",
        "Summarize the following legal clause: The party of the first part...",
        None  # Intentional missing prompt
    ], size=n_samples),
    'response_text': np.random.choice([
        "Quantum computing uses qubits that can exist in superposition...",
        "Here is the refactored code using asyncio.gather()...",
        "SELECT c.id, COUNT(o.id) FROM customers c JOIN orders o...",
        "This clause establishes liability caps for third-party vendors...",
        "I cannot assist with that request due to safety policies."
    ], size=n_samples),
    'model_family': np.random.choice(['llama-3', 'mistral-v2', 'qwen-2.5', 'deepseek-r1'], size=n_samples),
    'prompt_tokens': np.random.randint(15, 2048, size=n_samples),
    'completion_tokens': np.random.randint(5, 1024, size=n_samples),
    'generation_time_sec': np.random.exponential(scale=1.5, size=n_samples) + 0.1,
    'reward_score': np.random.uniform(-3.5, 4.5, size=n_samples)
})

# ==============================================================================
# AI ENGINEERING PRACTICE: LLM EVALUATION & PREPROCESSING PIPELINE
# ==============================================================================

# Q1 [Corpus Validation & Empty Prompt Scrubbing]:
# Context: Alignment datasets (DPO/RLHF) require non-empty prompt strings to avoid training models on blank inputs.
# Business/ML Purpose: Ensure evaluation samples contain valid prompt text prior to embedding or tokenization.
# Expected Skill: .dropna() combined with .str.strip() and boolean masking.
# Task: Create a cleaned DataFrame `valid_eval_df` by dropping rows where 'prompt_text' is NaN or contains only whitespace.
# Your solution:

df['prompt_text'].isna().sum()
df['prompt_text']=df['prompt_text'].str.strip().replace('',np.nan)
valid_eval_df = df.dropna(subset='prompt_text')
df.info()

# Q2 [Feature Engineering - Token Compression / Efficiency Ratio]:
# Context: Model response brevity and efficiency are critical for low-latency assistant applications.
# Business/ML Purpose: Engineer a token expansion ratio feature (completion tokens / prompt tokens) to detect verbose or truncated responses.
# Expected Skill: Vectorized division (`df['col1'] / df['col2']`).
# Task: Create a new column 'token_ratio' calculated as 'completion_tokens' divided by 'prompt_tokens'.
# Your solution:

valid_eval_df['token_ratio']= valid_eval_df['completion_tokens']/valid_eval_df['prompt_tokens']

# Q3 [Text Pattern Extraction - Code Detection Indicator]:
# Context: Multi-modal instruction tuning requires separating code-generation prompts from general prose queries.
# Business/ML Purpose: Build a binary heuristic feature indicating whether a prompt involves programming tasks.
# Expected Skill: Regex string searching with `.str.contains(case=False)`.
# Task: Create a binary column 'is_code_prompt' (1 or 0) indicating whether 'prompt_text' contains keywords ('python', 'sql', 'import', 'select', 'script').
# Your solution:

valid_eval_df['is_code_prompt']= valid_eval_df['prompt_text'].str.contains(
    r'python|sql|import|select|script', 
    case=False, 
    regex=True).astype(dtype=int)

# Q4 [Throughput Calculation - Tokens Per Second Metric]:
# Context: Inference pipeline performance tracking requires evaluating generation speed across model architectures.
# Business/ML Purpose: Calculate tokens-per-second throughput features for latency-aware routing models.
# Expected Skill: Vectorized numerical computation.
# Task: Create a column 'tokens_per_sec' calculated as 'completion_tokens' / 'generation_time_sec'.
# Your solution:

valid_eval_df['tokens_per_sec'] = valid_eval_df['completion_tokens']/valid_eval_df['generation_time_sec']


# Q5 [Outlier Detection - Generation Time Capping]:
# Context: Inference timeouts or queue spikes create extreme latency outliers that distort regression training metrics.
# Business/ML Purpose: Cap extreme generation times at the 95th percentile threshold (winsorization).
# Expected Skill: Quantile computation (`.quantile()`) and `np.where()` vectorized capping.
# Task: Compute the 95th percentile of 'generation_time_sec'. Create 'generation_time_capped' where any value exceeding the 95th percentile is set to that threshold.
# Your solution:

nintey_five_percentile=valid_eval_df['generation_time_sec'].quantile(q=0.95)

exceeding_95_percentile = valid_eval_df['generation_time_sec'] > nintey_five_percentile

valid_eval_df['generation_time_capped'] = valid_eval_df['generation_time_sec']

valid_eval_df.loc[exceeding_95_percentile,'generation_time_capped'] = nintey_five_percentile

# Q6 [Binary Reward Target Labeling]:
# Context: Preference models (DPO) require binary targets indicating whether a response meets high-quality reward thresholds.
# Business/ML Purpose: Convert continuous reward scores into a binary alignment label (1 = Accepted, 0 = Rejected).
# Expected Skill: Vectorized conditional assignment using `np.where()`.
# Task: Create a binary column 'is_accepted' where 'reward_score' >= 1.0 is labeled 1, and < 1.0 is labeled 0.
# Your solution:

valid_eval_df['is_accepted'] = 0

reward_score_condition = valid_eval_df['reward_score'] >=1

valid_eval_df.loc[reward_score_condition,'is_accepted'] = 1

# Q7 [Categorical Encoding - Model Identifier Dummies]:
# Context: Tabular meta-evaluators require model architecture tags in encoded numeric format.
# Business/ML Purpose: One-hot encode model families without dropping reference levels for full-rank neural net inputs.
# Expected Skill: `pd.get_dummies()` with `dtype=int` and custom prefixes.
# Task: Generate one-hot encoded columns for 'model_family' using prefix 'model' and concatenate them back to `valid_eval_df`.
# Your solution:


# Q8 [Min-Max Feature Scaling on Performance Metrics]:
# Context: Combining continuous throughput and length metrics into tabular classifiers requires uniform scale normalization [0.0, 1.0].
# Business/ML Purpose: Standardize features to prevent larger scale metrics (e.g., token counts) from dominating gradient descent.
# Expected Skill: Vectorized Min-Max scaling formula `(x - min) / (max - min)`.
# Task: Scale 'tokens_per_sec' into a new column 'scaled_throughput' bounded between 0.0 and 1.0.
# Your solution:


# Q9 [Stratified Train/Validation Partitioning for Preference Models]:
# Context: Evaluation sets for DPO/RLHF models must maintain identical proportions of accepted vs rejected responses.
# Business/ML Purpose: Prevent target distribution drift between training and validation benchmark splits.
# Expected Skill: Stratified sampling via index manipulation or scikit-learn integration.
# Task: Partition `valid_eval_df` into an 80% training set (`train_df`) and 20% validation set (`val_df`), preserving the ratio of 'is_accepted' targets.
# Your solution:


# Q10 [Dense Float32 Feature Matrix Export]:
# Context: Exporting tabular features into zero-NaN NumPy float32 matrices for PyTorch/XGBoost model consumption.
# Business/ML Purpose: Ensure structural matrix completeness prior to tensor allocation.
# Expected Skill: Column selection, verification with `.isna().sum()`, and `.to_numpy(dtype=np.float32)` conversion.
# Task: From `valid_eval_df`, select features ['token_ratio', 'tokens_per_sec', 'scaled_throughput', 'is_code_prompt'], verify zero nulls, and extract 2D feature matrix `X` and 1D target array `y` ('is_accepted').
# Your solution: