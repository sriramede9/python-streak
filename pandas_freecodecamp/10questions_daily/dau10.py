import pandas as pd
import numpy as np

# Dataset: Synthetic Financial Fraud & Transaction Telemetry
np.random.seed(101)
n_tx = 1200

df = pd.DataFrame(
    {
        "transaction_id": [f"tx_{i:06d}" for i in range(n_tx)],
        "timestamp": pd.date_range(start="2026-08-01", periods=n_tx, freq="min"),
        "user_id": np.random.choice([f"user_{i:03d}" for i in range(1, 50)], size=n_tx),
        "amount": np.random.exponential(scale=150.0, size=n_tx) + 1.0,
        "merchant_category": np.random.choice(
            ["grocery", "electronics", "travel", "crypto_exchange", "luxury"], size=n_tx
        ),
        "ip_country": np.random.choice(
            ["US", "CA", "UK", "DE", "RU", "CN", "NG", np.nan],
            p=[0.6, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01],
            size=n_tx,
        ),
        "is_fraud": np.random.choice([0, 1], p=[0.96, 0.04], size=n_tx),
    }
)

# ==============================================================================
# AI ENGINEERING PRACTICE: FRAUD FEATURE PIPELINES & TABULAR PREPROCESSING
# ==============================================================================

# Q1 [Imputation & Feature Defaulting for Categorical Pipelines]:
# Context: Missing geolocation codes in transaction streams cause feature encoding pipelines to drop rows or throw errors.
# Business/ML Purpose: Preserve transaction volume while capturing 'missingness' as an explicit risk signal for fraud models.
# Expected Skill: .fillna() on string categorical columns.
# Task: Fill missing values in 'ip_country' with the explicit string category 'UNKNOWN_COUNTRY'.
# Your solution:

df["ip_country"].isna().sum()
df["ip_country"].value_counts()
# Count how many rows are literally the string 'nan'
(df["ip_country"] == "nan").sum()
df["ip_country"] = df["ip_country"].replace("nan", "UNKNOWN_COUNTRY")
# Q2 [Feature Engineering - Transaction Velocity / Rolling Window Aggregation]:
# Context: High transaction frequency within short timeframes (velocity spike) is a primary indicator of card testing fraud.
# Business/ML Purpose: Engineer historical context features for real-time risk scoring models.
# Expected Skill: .groupby() combined with .rolling() window calculations on time series data.
# Task: Calculate a 10-minute rolling count of transactions per 'user_id' based on 'timestamp' and store in 'tx_count_10m'.
# Your solution:


# df['tx_count_10m']=
# 1. Sort to guarantee row alignment
df = df.sort_values(["user_id", "timestamp"])

# 2. Calculate and assign the rolling count safely
df["tx_count_10m"] = (
    df.groupby("user_id").rolling("10min", on="timestamp")["timestamp"].count().values
)


# Q3 [Logarithmic Power Transformation for Heavily Skewed Distibutions]:
# Context: Transaction amounts follow a heavy right-skewed power-law distribution ($1 vs $10,000+), causing gradient instability in neural networks.
# Business/ML Purpose: Compress range variance to improve convergence rate during model training.
# Expected Skill: Vectorized log transformation with `np.log1p()`.
# Task: Create a new column 'log_amount' containing the log1p transform of the 'amount' column.
# Your solution:
df["log_amount"] = np.log1p(df["amount"])

# Q4 [Frequency / Count Encoding for High-Cardinality Categoricals]:
# Context: User IDs carry strong historical behavioral weight, but standard One-Hot Encoding creates an overly sparse matrix.
# Business/ML Purpose: Encode high-cardinality identifiers into dense numerical features representing user activity levels.
# Expected Skill: Value mapping using `.value_counts()` and `.map()`.
# Task: Create a column 'user_activity_freq' where each 'user_id' is mapped to its relative occurrence frequency in the dataset.
# Your solution:

ref_rel_freq = df["user_id"].value_counts(normalize=True).to_dict()

df["user_activity_freq"] = df["user_id"].map(ref_rel_freq)

# Q5 [Target-Based Risk Probability Feature Mapping]:
# Context: Certain merchant categories (e.g., 'crypto_exchange', 'luxury') carry significantly higher intrinsic fraud risk.
# Business/ML Purpose: Encode categorical features as empirical fraud risk ratios to boost tree-based model split efficiency.
# Expected Skill: Grouped mean calculation (.groupby().transform('mean')).
# Task: Calculate the historical fraud rate per 'merchant_category' and assign it to a new column 'merchant_risk_score' (avoiding target leakage by taking the global category mean).
# Your solution:

df["merchant_category"].value_counts()
df.head()
df["merchant_risk_score"] = df.groupby("merchant_category")["is_fraud"].transform(
    lambda x: x.mean()
)
# Q6 [Statistical Outlier Capping / Winsorization via IQR]:
# Context: Outlier transaction amounts disrupt numerical scale normalization and linear decision boundaries.
# Business/ML Purpose: Bound extreme values at statistical upper limits rather than discarding valuable fraud records.
# Expected Skill: Quantile computation (`.quantile()`) and `np.where()` threshold replacement.
# Task: Compute the 75th percentile (Q3) and IQR of 'amount'. Cap any values exceeding `Q3 + 3.0 * IQR` at that upper bound in a column 'amount_capped'.
# Your solution:

q1, q3 = df["amount"].quantile([0.25, 0.75])
iqr = q3 - q1
df["amount_capped"] = df["amount"]
condition = df["amount"] > q3 + (3.0 * iqr)
df.loc[condition, "amount_capped"] = iqr

# Q7 [Binary Risk Feature Extraction - Domain Heuristics]:
# Context: Combining foreign IP origin and high-risk merchant types represents a high-probability fraud heuristic.
# Business/ML Purpose: Create explicit interaction signals for tabular GBDT / XGBoost classifiers.
# Expected Skill: Multi-condition boolean masking with `np.where()`.
# Task: Create a binary feature 'is_high_risk_tx' (1 or 0) that triggers if 'merchant_category' is in ['crypto_exchange', 'luxury'] AND 'ip_country' is NOT in ['US', 'CA'].
# Your solution:

condition_fraud = df["merchant_category"].isin(["crypto_exchange", "luxury"]) & ~(
    df["ip_country"].isin(["US", "CA"])
)

df["is_high_risk_tx"] = np.where(condition_fraud, 1, 0)

# Q8 [Min-Max Scaled Matrix Preprocessing]:
# Context: Distance-based models (k-NN, SVMs) and Autoencoders require feature scales bounded strictly between 0.0 and 1.0.
# Business/ML Purpose: Standardize numerical inputs to uniform numeric ranges.
# Expected Skill: Vectorized Min-Max transformation `(x - min) / (max - min)`.
# Task: Scale 'log_amount' and 'tx_count_10m' into new columns 'scaled_amount' and 'scaled_velocity' ranging from 0.0 to 1.0.
# Your solution:
min_log_amount = df["log_amount"].min()
max_log_amount = df["log_amount"].max()
diff_log = max_log_amount - min_log_amount

df["scaled_amount"] = (df["log_amount"] - min_log_amount) / diff_log

min_tx_amount = df["tx_count_10m"].min()
max_tx_amount = df["tx_count_10m"].max()
diff_tx = max_tx_amount - min_tx_amount

df["scaled_velocity"] = (df["tx_count_10m"] - min_tx_amount) / diff_tx

# Q9 [Handling Severe Class Imbalance - Downsampling Majority Class]:
# Context: Fraud datasets are heavily imbalanced (~96% non-fraud, ~4% fraud), leading to biased classification bounds.
# Business/ML Purpose: Balance training distributions to ensure equal class weight during gradient updates.
# Expected Skill: Boolean filtering, length evaluation, and `.sample()` matching.
# Task: Create a balanced DataFrame `balanced_df` by keeping all fraud rows (`is_fraud == 1`) and randomly sampling an equal number of non-fraud rows (`is_fraud == 0`) with `random_state=42`.
# Your solution:

# protect the fraud cases as we care about them
fraud_rows = df[df["is_fraud"] == 1]
len_fraud_rows = len(fraud_rows)

# randomly sampling an equal number of non-fraud rows
non_fraud_rows = df[df["is_fraud"] == 0].sample(n=len_fraud_rows, random_state=42)
balanced_df = (
    pd.concat([fraud_rows, non_fraud_rows])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
# Q10 [Dense Float32 Feature Matrix and Label Vector Export]:
# Context: Production inference pipelines require clean 2D NumPy array inputs for Tensor/GBDT models.
# Business/ML Purpose: Verify complete feature integrity (zero NaNs) and export numeric arrays.
# Expected Skill: Column selection, verification with `.isna().sum()`, and `.to_numpy(dtype=np.float32)`.
# Task: From `balanced_df`, select features ['scaled_amount', 'scaled_velocity', 'user_activity_freq', 'merchant_risk_score', 'is_high_risk_tx'], verify 0 NaNs, and export 2D matrix `X` and 1D target array `y` ('is_fraud').
# Your solution:
assert (
    balanced_df[
        [
            "scaled_amount",
            "scaled_velocity",
            "user_activity_freq",
            "merchant_risk_score",
            "is_high_risk_tx",
        ]
    ]
    .isna()
    .sum()
    .sum()
    == 0
)

x=   balanced_df[
        [
            "scaled_amount",
            "scaled_velocity",
            "user_activity_freq",
            "merchant_risk_score",
            "is_high_risk_tx",
        ]
    ].to_numpy(dtype=np.float32)
y = df['is_fraud'].to_numpy()