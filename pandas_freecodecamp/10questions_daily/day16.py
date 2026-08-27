import pandas as pd
import numpy as np

# Dataset: Synthetic Network Intrusion Telemetry & Threat Intelligence Logs
np.random.seed(2026)
n_flows = 1200

df = pd.DataFrame({
    'flow_id': [f"flow_{i:06d}" for i in range(n_flows)],
    'timestamp': pd.date_range(start='2026-08-11 08:00:00', periods=n_flows, freq='50ms'),
    'src_ip': np.random.choice([
        '192.168.1.105', '10.0.4.12', '172.16.0.5', '45.33.32.156', '185.220.101.5', np.nan
    ], size=n_flows, p=[0.35, 0.25, 0.20, 0.10, 0.08, 0.02]),
    'dest_port': np.random.choice([80, 443, 22, 53, 8080, 3389, 9999], size=n_flows, p=[0.4, 0.35, 0.08, 0.1, 0.04, 0.02, 0.01]),
    'payload_bytes': np.random.exponential(scale=1200.0, size=n_flows) + 1.0,
    'flags_str': np.random.choice(['SYN', 'ACK', 'SYN-ACK', 'FIN-ACK', 'RST', 'CORRUPT_FLAG', None], size=n_flows),
    'anomaly_score': np.random.uniform(-2.5, 8.5, size=n_flows),
    'is_malicious': np.random.choice([0, 1], p=[0.94, 0.06], size=n_flows)
})

# ==============================================================================
# AI ENGINEERING PRACTICE: THREAT INTEL & NETWORK FEATURE PIPELINES
# ==============================================================================

# Q1 [IP Subnet Extraction & High-Cardinality Parsing]:
# Context: Raw IP addresses cause extreme feature sparsity in machine learning models.
# Business/ML Purpose: Extract the /24 subnet (first 3 octets) to aggregate local network behavior without overfitting to specific host IPs.
# Expected Skill: Vectorized string splitting `.str.split()` and `.str.join()`.
# Task: Extract the first 3 octets of 'src_ip' (e.g., '192.168.1.105' -> '192.168.1') into 'ip_subnet'. Fill any missing IP rows with '0.0.0'.
# Your solution:


# Q2 [Logarithmic Transformation for Skewed Packet Payload Sizes]:
# Context: Network packet byte sizes follow a heavy right-skewed distribution (from 1 byte to mega-bytes).
# Business/ML Purpose: Stabilize feature variance using log-scaling to improve gradient descent in neural threat detection models.
# Expected Skill: Vectorized logarithmic transformation using `np.log1p()`.
# Task: Create a new column 'log_payload_bytes' calculated as `np.log1p(df['payload_bytes'])`.
# Your solution:


# Q3 [Categorical Flag Cleaning & Pattern Validation]:
# Context: Malformed TCP flag strings in packet captures can cause one-hot encoders to crash or generate garbage categories.
# Business/ML Purpose: Filter invalid TCP flag strings before encoding features for intrusion detection classifiers.
# Expected Skill: `.isin()` filtering combined with `.fillna()`.
# Task: Replace any 'flags_str' that is NOT in ['SYN', 'ACK', 'SYN-ACK', 'FIN-ACK', 'RST'] with 'UNKNOWN_FLAG'.
# Your solution:


# Q4 [Domain Categorization & Port Risk Mapping]:
# Context: Specific network ports (e.g., SSH: 22, RDP: 3389) carry higher baseline risk for brute-force cyber attacks.
# Business/ML Purpose: Construct a binary indicator feature signaling high-risk administrative port traffic.
# Expected Skill: `.isin()` conditional mapping with `np.where()`.
# Task: Create a binary column 'is_admin_port' (1 or 0) indicating whether 'dest_port' is in `[22, 3389, 8080]`.
# Your solution:


# Q5 [Sliding Window Traffic Burst Velocity Calculation]:
# Context: Distributed Denial of Service (DDoS) attacks are characterized by rapid bursts of requests in short time windows.
# Business/ML Purpose: Calculate rolling flow counts per IP subnet to capture instantaneous network traffic spikes.
# Expected Skill: `.groupby()` with `.rolling()` window frequency calculations on time-indexed data.
# Task: Calculate a 1-second rolling count of flows per 'ip_subnet' based on 'timestamp' and store in 'subnet_flow_count_1s'.
# Your solution:


# Q6 [Anomaly Score Z-Standardization]:
# Context: Raw anomaly scores output by statistical detectors have varying means and standard deviations across sensor nodes.
# Business/ML Purpose: Standardize anomaly scores to zero mean and unit variance for downstream GBDT and neural rankers.
# Expected Skill: Vectorized Z-score calculation `(x - mean) / std`.
# Task: Compute the Z-score of 'anomaly_score' across the dataset and store it in a new column 'z_anomaly_score'.
# Your solution:


# Q7 [Outlier Capping / Winsorization on Packet Rates]:
# Context: Extreme traffic spikes (e.g., network test spikes) create statistical outliers that distort classification boundaries.
# Business/ML Purpose: Bound packet payload sizes at the 99th percentile threshold to stabilize loss function gradients.
# Expected Skill: Quantile computation (`.quantile()`) and `np.where()` vectorized capping.
# Task: Calculate the 99th percentile of 'payload_bytes'. Create 'payload_capped' where any value exceeding the 99th percentile is set to that threshold.
# Your solution:


# Q8 [One-Hot Encoding for Validated TCP Flags]:
# Context: Tabular intrusion detection models require categorical TCP state indicators in binary numerical format.
# Business/ML Purpose: Convert TCP flags into one-hot dummy variables while retaining all valid categories.
# Expected Skill: `pd.get_dummies()` with custom prefixes and `dtype=int`.
# Task: Generate one-hot encoded columns for 'flags_str' using the prefix 'flag' and join them back to `df`.
# Your solution:


# Q9 [Chronological Train/Validation Partitioning]:
# Context: Network traffic exhibits strong temporal dependencies; random splits introduce temporal data leakage from future traffic patterns.
# Business/ML Purpose: Partition data into historical training (first 80%) and future validation (final 20%) sets based strictly on timestamp order.
# Expected Skill: Index slicing `.iloc` on chronologically sorted DataFrames.
# Task: Ensure `df` is sorted by 'timestamp' and split it into an 80% training set (`train_df`) and 20% validation set (`val_df`) without shuffling.
# Your solution:


# Q10 [Dense Float32 Intrusion Feature Matrix Export]:
# Context: Exporting tabular network metrics into zero-NaN NumPy float32 matrices for PyTorch or XGBoost threat models.
# Business/ML Purpose: Verify structural matrix integrity (0 NaNs) prior to tensor allocation.
# Expected Skill: Column selection, verification with `.isna().sum()`, and `.to_numpy(dtype=np.float32)` conversion.
# Task: From `train_df`, select features ['log_payload_bytes', 'is_admin_port', 'z_anomaly_score', 'subnet_flow_count_1s'], verify zero nulls, and extract 2D feature matrix `X` and 1D target array `y` ('is_malicious').
# Your solution: