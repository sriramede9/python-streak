import pandas as pd
import numpy as np

# Dataset: Synthetic GPU Cluster Telemetry & Training Job Health Logs
np.random.seed(2026)
n_records = 1200

df = pd.DataFrame({
    'job_id': [f"job_{i:06d}" for i in range(n_records)],
    'timestamp': pd.date_range(start='2026-08-13 12:00:00', periods=n_records, freq='500ms'),
    'node_id': np.random.choice([f"node_a100_{i:02d}" for i in range(1, 10)], size=n_records),
    'gpu_memory_used_mb': np.random.uniform(4000.0, 81920.0, size=n_records),
    'gpu_temp_celsius': np.random.normal(loc=72.0, scale=8.0, size=n_records),
    'error_log_str': np.random.choice([
        "CUDA_OUT_OF_MEMORY: Tried to allocate 12.50 GiB",
        "STATUS_OK: Training iteration step completed in 120ms",
        "WARNING: Thermal throttling activated on GPU-3",
        "SYSTEM_HEALTH: All checks passed",
        "CORRUPTED_LOG_PAYLOAD",
        None
    ], size=n_records, p=[0.08, 0.60, 0.12, 0.15, 0.03, 0.02]),
    'node_health_score': np.random.uniform(-1.0, 10.0, size=n_records),
    'is_node_failure': np.random.choice([0, 1], p=[0.93, 0.07], size=n_records)
})

# ==============================================================================
# AI ENGINEERING PRACTICE: GPU CLUSTER & INFRASTRUCTURE PIPELINES
# ==============================================================================

# Q1 [Log Regex Parsing & Feature Extraction - OOM Memory Size]:
# Context: Out-Of-Memory (OOM) errors disrupt large-scale model training clusters.
# Business/ML Purpose: Extract requested allocation sizes (in GiB) from unformatted error logs to predict job failure memory thresholds.
# Expected Skill: Regex string extraction `.str.extract()` and numeric type conversion.
# Task: Extract the numeric float value following 'allocate' (e.g., '12.50' from 'allocate 12.50 GiB') into 'oom_requested_gib'. Fill NaNs with 0.0.
# Your solution:


# Q2 [Categorical Cleaning & Health Status Classification]:
# Context: Mixed log verbosity creates messy string categories that crash one-hot encoders.
# Business/ML Purpose: Classify raw log strings into standardized operational states before training cluster health classifiers.
# Expected Skill: String matching `.str.contains()` with `np.select()` multi-condition assignment.
# Task: Create 'log_status' categorized as 'OOM_ERROR' (if log contains 'OUT_OF_MEMORY'), 'THERMAL_WARN' (if contains 'Thermal'), 'OK' (if contains 'STATUS_OK' or 'SYSTEM_HEALTH'), and 'OTHER' for all else.
# Your solution:


# Q3 [Sliding Window GPU Temperature Spikes]:
# Context: Thermal throttling triggers sudden GPU frequency drops and model iteration latency spikes.
# Business/ML Purpose: Compute rolling maximum temperatures per node over short time windows to detect rapid thermal accumulation.
# Expected Skill: `.groupby()` with `.rolling()` window max calculations on time-indexed data.
# Task: Calculate a 5-second rolling maximum of 'gpu_temp_celsius' per 'node_id' based on 'timestamp' and store in 'rolling_max_temp_5s'.
# Your solution:


# Q4 [Physical Constraints Filtering & Thermal Sensor Validation]:
# Context: Sensor degradation or bus corruptions occasionally report invalid GPU temperatures (e.g., negative or > 120°C).
# Business/ML Purpose: Purge corrupted physical readings prior to training infrastructure failure predictors.
# Expected Skill: Multi-condition boolean masking (`0.0 <= temp <= 110.0`).
# Task: Filter `df` to retain only rows where 'gpu_temp_celsius' falls strictly within the valid range of 0.0°C to 110.0°C.
# Your solution:


# Q5 [Power Transformation for Skewed Memory Allocation Features]:
# Context: GPU memory consumption exhibits heavy bi-modal distribution (idle vs full 80GB VRAM allocation).
# Business/ML Purpose: Scale memory utilization linearly relative to max capacity (81,920 MB) to generate standardized load ratios.
# Expected Skill: Vectorized division (`df['col'] / max_val`).
# Task: Normalize 'gpu_memory_used_mb' by dividing by 81920.0 (A100 VRAM capacity in MB) to create a continuous feature 'vram_utilization_ratio' bounded between 0.0 and 1.0.
# Your solution:


# Q6 [Outlier Capping / Winsorization on Health Metrics]:
# Context: Unbounded node health scores contain extreme positive or negative outliers from synthetic telemetry probes.
# Business/ML Purpose: Cap health score extremes at 1st and 99th percentiles to stabilize gradient updates in neural monitoring models.
# Expected Skill: Quantile computation (`.quantile()`) and `np.where()` threshold replacement.
# Task: Compute the 1st and 99th percentiles of 'node_health_score'. Cap values below the 1st percentile or above the 99th percentile in a new column 'health_score_capped'.
# Your solution:


# Q7 [Target Risk Profiling - Empirical Node Failure Rates]:
# Context: Different GPU node architectures or batches ('node_id') exhibit varying baseline hardware failure probabilities.
# Business/ML Purpose: Map high-cardinality node IDs to historical failure rates to boost decision-tree splitting efficiency.
# Expected Skill: Grouped probability calculation using `.groupby().transform('mean')`.
# Task: Calculate the historical 'is_node_failure' rate for each 'node_id' and assign it to a new column 'node_failure_probability'.
# Your solution:


# Q8 [One-Hot Encoding for Operational Log Statuses]:
# Context: Tabular failure prediction models require categorical status flags in one-hot numeric format.
# Business/ML Purpose: Encode 'log_status' (from Q2) into binary vector features without dropping reference levels.
# Expected Skill: `pd.get_dummies()` with custom prefixes and `dtype=int`.
# Task: Generate one-hot encoded columns for 'log_status' using the prefix 'status' and concatenate them back to `df`.
# Your solution:


# Q9 [Chronological Train/Validation Split for Telemetry Time Series]:
# Context: GPU cluster telemetry is temporally correlated; random splits cause target leakage from adjacent time steps.
# Business/ML Purpose: Partition data into historical train (first 80%) and future validation (final 20%) sets strictly ordered by timestamp.
# Expected Skill: Sequential index slicing `.iloc` on chronologically sorted DataFrames.
# Task: Sort `df` by 'timestamp' and split it into an 80% training set (`train_df`) and 20% validation set (`val_df`) without shuffling.
# Your solution:


# Q10 [Dense Float32 Cluster Feature Matrix Export]:
# Context: Production inference pipelines require clean 2D NumPy float32 array inputs for GBDT or PyTorch models.
# Business/ML Purpose: Verify complete feature integrity (0 NaNs) and export numeric feature arrays.
# Expected Skill: Column selection, verification with `.isna().sum()`, and `.to_numpy(dtype=np.float32)`.
# Task: From `train_df`, select features ['vram_utilization_ratio', 'rolling_max_temp_5s', 'oom_requested_gib', 'node_failure_probability', 'health_score_capped'], verify 0 NaNs, and export 2D matrix `X` and 1D target array `y` ('is_node_failure').
# Your solution: