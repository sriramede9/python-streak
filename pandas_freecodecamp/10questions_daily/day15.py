import pandas as pd
import numpy as np

# Dataset: Synthetic Autonomous Vehicle Sensor & Perception Telemetry
np.random.seed(2026)
n_frames = 1200

df = pd.DataFrame({
    'frame_id': [f"frame_{i:06d}" for i in range(n_frames)],
    'timestamp': pd.date_range(start='2026-08-10 12:00:00', periods=n_frames, freq='100ms'),
    'vehicle_speed_mps': np.random.normal(loc=15.0, scale=4.0, size=n_frames),
    'steering_angle_rad': np.random.uniform(-0.5, 0.5, size=n_frames),
    'radar_distance_m': np.random.choice([np.nan, 5.2, 12.8, 45.0, 80.1, 120.5], size=n_frames, p=[0.12, 0.2, 0.3, 0.2, 0.1, 0.08]),
    'object_detected_class': np.random.choice(['pedestrian', 'vehicle', 'cyclist', 'static_obstacle', 'unknown'], size=n_frames),
    'bounding_box_str': np.random.choice([
        "bbox:[100,200,150,300]",
        "bbox:[50,80,200,400]",
        "INVALID_BBOX",
        "bbox:[0,10,50,50]",
        None
    ], size=n_frames),
    'is_critical_event': np.random.choice([0, 1], p=[0.92, 0.08], size=n_frames)
})

# ==============================================================================
# AI ENGINEERING PRACTICE: SENSOR TELEMETRY & PERCEPTION PIPELINES
# ==============================================================================

# Q1 [Sensor Signal Imputation & Forward Fill]:
# Context: High-frequency radar sensors occasionally drop frames due to occlusion or multi-path interference.
# Business/ML Purpose: Propagate the last known valid distance reading (forward-fill) to prevent NaN drops in real-time trajectory models.
# Expected Skill: `.ffill()` combined with `.bfill()` for residual edge cases.
# Task: Impute missing values in 'radar_distance_m' by forward-filling the most recent valid measurement, filling any initial leading NaNs with 0.0.
# Your solution:
df.info()
df.head()
if pd.isna(df['radar_distance_m'][0]):
    df['radar_distance_m'][0] = 0.0

df['radar_distance_ma'].ffill()    
# Q2 [Sliding Window Velocity Acceleration Feature Engineering]:
# Context: Raw vehicle speed is insufficient for predicting sudden braking or collision risk; rate of change (acceleration) is required.
# Business/ML Purpose: Engineer continuous acceleration features ($\Delta v / \Delta t$) across sequential telemetry frames.
# Expected Skill: `.diff()` on numeric columns or `.shift()` subtraction.
# Task: Calculate frame-to-frame vehicle acceleration in $m/s^2$ assuming a constant 0.1s timestep ($100ms$), storing it in 'vehicle_accel_mps2'.
# Your solution:


# Q3 [Text/Regex Feature Extraction - Bounding Box Coordinates]:
# Context: Object detection heads output string-formatted bounding box coordinates `bbox:[x,y,w,h]` that must be unpacked into numerical features.
# Business/ML Purpose: Extract bounding box area ($width \times height$) as an indicator of object proximity/size for collision risk models.
# Expected Skill: String extraction `.str.extract()` using regex and mathematical column creation.
# Task: Extract the 3rd and 4th integers (width and height) from 'bounding_box_str', convert to float, and calculate 'bbox_area' ($w \times h$). Assign 0.0 for invalid/missing boxes.
# Your solution:


# Q4 [Outlier Filtering & Physical Constraints Thresholding]:
# Context: Corrupted CAN-bus messages occasionally record physically impossible vehicle speeds (e.g., negative speed or > 100 m/s).
# Business/ML Purpose: Purge invalid physical sensor readings prior to training state-estimation neural networks.
# Expected Skill: Boolean filtering across physical range bounds (`0.0 <= speed <= 80.0`).
# Task: Filter `df` to keep only rows where 'vehicle_speed_mps' falls strictly within the physically plausible range of 0.0 to 80.0 m/s.
# Your solution:


# Q5 [Coordinate Transformation - Polar to Cartesian Feature Generation]:
# Context: Distance and steering angle represent polar coordinates relative to the ego-vehicle.
# Business/ML Purpose: Convert polar coordinates $(r, \theta)$ into 2D Cartesian coordinates $(x, y)$ for spatial neural network inputs.
# Expected Skill: Vectorized NumPy trigonometric functions (`np.cos`, `np.sin`).
# Task: Compute 2D position coordinates: $x = \text{radar\_distance\_m} \times \cos(\text{steering\_angle\_rad})$ and $y = \text{radar\_distance\_m} \times \sin(\text{steering\_angle\_rad})$.
# Your solution:


# Q6 [Low-Frequency Categorical Grouping & Noise Reduction]:
# Context: Rare or ambiguous detection classes ('unknown', 'static_obstacle') introduce label noise into object tracking models.
# Business/ML Purpose: Consolidate low-frequency or noisy detection classes into a generic 'other' category to stabilize classifier outputs.
# Expected Skill: `.value_counts()` or `.isin()` conditional replacement with `np.where()`.
# Task: Replace any 'object_detected_class' that is NOT 'pedestrian', 'vehicle', or 'cyclist' with the string 'other' in a column 'class_clean'.
# Your solution:


# Q7 [Target Risk Encoding for Object Classes]:
# Context: Certain detected objects (e.g., pedestrians, cyclists) carry higher baseline accident severity risk than vehicles or static obstacles.
# Business/ML Purpose: Encode detection categories as historical critical-event probabilities to enhance decision-tree splitting efficiency.
# Expected Skill: Grouped probability calculation using `.groupby().transform('mean')`.
# Task: Calculate the empirical 'is_critical_event' rate for each 'class_clean' category and assign it to 'class_risk_score'.
# Your solution:


# Q8 [Min-Max Scaling on Kinematic Features]:
# Context: Kinematic features (speed, acceleration, distance) span disparate numerical scales that slow gradient descent.
# Business/ML Purpose: Scale kinematic predictors to a uniform $[0.0, 1.0]$ numerical range for Transformer/LSTM trajectory models.
# Expected Skill: Vectorized Min-Max transformation `(x - min) / (max - min)`.
# Task: Scale 'vehicle_speed_mps' and 'radar_distance_m' into new columns 'scaled_speed' and 'scaled_distance' bounded between 0.0 and 1.0.
# Your solution:


# Q9 [Time-Series Deterministic Sequential Train/Val Split]:
# Context: Telemetry data exhibits high temporal autocorrelation; random splitting causes severe data leakage between consecutive frames.
# Business/ML Purpose: Partition data into sequential train (first 80%) and validation (final 20%) sets based strictly on timestamp order.
# Expected Skill: Index-based slicing `.iloc` on chronologically sorted DataFrames.
# Task: Sort `df` by 'timestamp' and split it into an 80% training set (`train_df`) and 20% validation set (`val_df`) without shuffling.
# Your solution:


# Q10 [Dense Float32 Sensor Matrix Export for Trajectory Neural Nets]:
# Context: Production inference engines require clean, zero-NaN NumPy float32 matrices for tensor allocation.
# Business/ML Purpose: Verify complete feature matrix integrity (0 NaNs) and export numeric arrays.
# Expected Skill: Column selection, verification with `.isna().sum()`, and `.to_numpy(dtype=np.float32)` conversion.
# Task: From `train_df`, select features ['scaled_speed', 'scaled_distance', 'vehicle_accel_mps2', 'bbox_area', 'class_risk_score'], verify 0 NaNs, and export 2D matrix `X` and 1D target array `y` ('is_critical_event').
# Your solution: