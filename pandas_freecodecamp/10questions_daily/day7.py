import re as re

import numpy as np
import pandas as pd
from scipy import stats

# Dataset: Simulated Web Server Security Logs
url = "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_json_logs/nginx_json_logs"

# Load JSON lines log data
df = pd.read_json(url, lines=True)

df.info()

# ==============================================================================
# AI ENGINEERING PRACTICE: LOG PREPROCESSING & ANOMALY FEATURE PIPELINE
# ==============================================================================

# Q1 [Datetime Parsing & Time-Series Validation]:
# Context: Security log anomaly detection models require accurate datetime indexes.
# Business/ML Purpose: Ensure raw timestamp strings are properly parsed to timezone-aware UTC datetimes.
# Expected Skill: pd.to_datetime() conversion and setting datetime indexes.
# Task: Convert the 'time' column to UTC datetime and drop any log rows with invalid/unparseable timestamps.
# Your solution:
df["time"]
df["time"] = pd.to_datetime(
    df["time"], utc=True, format="%d/%b/%Y:%H:%M:%S %z", errors="coerce"
)
df = df.set_index(df["time"])
df = df.dropna(subset="time")

# Q2 [Text Extraction - Regex Feature Parsing]:
# Context: User-Agent strings contain rich device and OS metadata for fraud detection.
# Business/ML Purpose: Extract structured categorical features from unstructured string logs.
# Expected Skill: String extraction with regex (.str.extract).
# Task: Extract the operating system ('Windows', 'Macintosh', 'Linux', 'Android', 'iPhone') from the 'agent' column into a new column 'os'.
# Your solution:

# Uses double quotes for the Python string, allowing single quotes inside freely
df["os"] = df["agent"].str.extract(
    r"(Windows|Macintosh|Linux|Android|iPhone)", flags=re.IGNORECASE
)

df["os"] = df["os"].str.capitalize()
df["os"].value_counts()

# Q3 [Imputation & Feature Defaulting]:
# Context: Missing categorical features can break downstream One-Hot encoders.
# Business/ML Purpose: Handle missing values in log data to maintain feature alignment.
# Expected Skill: .fillna() and string categorical handling.
# Task: Fill missing values in the 'os' column (from Q2) with 'Unknown'.
# Your solution:

df["os"] = df["os"].fillna(value="Unknown")
df["os"].value_counts()
# Q4 [IP Address Preprocessing / Subnet Extraction]:
# Context: High-cardinality IP addresses cause extreme feature sparsity in ML models.
# Business/ML Purpose: Reduce cardinality by extracting the /24 subnet (first 3 octets).
# Expected Skill: String splitting and vector joining (.str.split / .str.join).
# Task: Extract the first 3 octets of the 'remote_ip' column (e.g., '192.168.1.50' -> '192.168.1') into a column 'ip_subnet'.
# Your solution:

df["ip_subnet"] = df["remote_ip"].str.split(".", regex=False).str[:3].str.join(".")
df["ip_subnet"].value_counts()
# Q5 [Numerical Transformation & Skew Correction]:
# Context: Bytes transferred ('bytes') varies across several orders of magnitude.
# Business/ML Purpose: Apply log transformation to stabilize variance for neural network training.
# Expected Skill: Vectorized log transformation with np.log1p.
# Task: Create a column 'log_bytes' containing the log1p transformation of 'bytes'.
# Your solution:

df["bytes"] = np.log1p(df["bytes"])
# Q6 [Outlier Detection - Statistical Z-Score Thresholding]:
# Context: Unusually high response times ('request_time') often indicate DDoS attacks or server failures.
# Business/ML Purpose: Identify statistical outliers in server latency for anomaly labeling.
# Expected Skill: Computing Z-scores using mean/std and boolean filtering.
# Task: Identify log entries where 'request_time' has a Z-score > 3.0.
# Your solution:

z_scores = stats.zscore(a=df["bytes"])
outliers = df[abs(z_scores > 3.0)]
outliers

# Q7 [Time-Based Feature Extraction]:
# Context: Temporal patterns (hour of day, day of week) correlate heavily with malicious activity.
# Business/ML Purpose: Engineer cyclic/cyclical temporal features for ML classifiers.
# Expected Skill: Datetime properties (.dt.hour, .dt.dayofweek).
# Task: Extract 'hour_of_day' (0-23) and 'is_weekend' (1 if Saturday/Sunday, else 0) from the 'time' column.
# Your solution:

df["hour_of_day"] = df["time"].dt.hour

weekend_condition = df["time"].dt.weekday.isin([5, 6])

df["is_weekend"] = np.where(weekend_condition, 1, 0)


# Q8 [Categorical Encoding - Frequency Encoding]:
# Context: High-cardinality URI paths break traditional One-Hot Encoders.
# Business/ML Purpose: Encode URL endpoints as frequency counts to capture rare path targeting.
# Expected Skill: Mapping value counts to Series (.map / .value_counts).
# Task: Replace each URL path in the 'request' column with its relative frequency count across the dataset into 'request_freq'.
# Your solution:

relative_freq_counts_map = df["request"].value_counts(normalize=True).to_dict()

df["request_freq"] = df["request"].map(relative_freq_counts_map)
df["request_freq"]
# Q9 [Window Aggregation - Rolling Window Feature Engine]:
# Context: Sequential models need historical context (e.g., request frequency in the last 5 minutes).
# Business/ML Purpose: Calculate rolling statistics to detect rapid request bursts from single IP subnets.
# Expected Skill: .groupby() with rolling window aggregations (.rolling()).
# Task: Calculate a 5-minute rolling count of requests per 'ip_subnet'.
# Your solution:

# i remembered to sort before we do this
# Drops the current index and replaces it with a default integer index
# 1. Sort the ENTIRE DataFrame by time (Critical for rolling windows)
# Your index is already DatetimeIndex, so no need to set_index again.
# Just ensure it's sorted (it looks sorted by your info output range).

# Calculate rolling count per remote_ip
df.info()
# 1. Ensure the DataFrame is sorted by the DatetimeIndex (Critical for rolling)
df.set_index("time", inplace=True)
df.index
df = df.sort_index()

# 2. Perform groupby and rolling directly on the DatetimeIndex
# The 'on' parameter is NOT needed because the index is already datetime
rolling_result = (
    df.groupby("ip_subnet")
    .rolling("5min")["request"]
    .count()
    .reset_index(
        level=0, drop=True
    )  # This leaves a Series with a non-unique DatetimeIndex
)
# f.groupby('ip_subnet').rolling('5min')['request'].count().reset_index(level=0,drop=True)
# )
df["request_count_5min"] = rolling_result.values
df.head()


# Optional: Reset index if you need 'time' back as a regular column
# df.reset_index(inplace=True)

# Q10 [ML Feature Matrix Export]:
# Context: Preparing final cleaned arrays for XGBoost or PyTorch model ingestion.
# Business/ML Purpose: Construct a numeric NumPy feature matrix free of NaN values or string types.
# Expected Skill: Selecting numerical columns, checking nulls, and .to_numpy() conversion.
# Task: Select ['status', 'log_bytes', 'hour_of_day', 'is_weekend', 'request_freq'], drop any remaining nulls, and convert to a 2D float32 NumPy array X.
# Your solution:
df.info()
selected_columns = ["status", "log_bytes", "hour_of_day", "is_weekend", "request_freq"]


df[["bytes", "hour_of_day", "is_weekend", "request_freq"]].to_numpy(dtype="float32")
