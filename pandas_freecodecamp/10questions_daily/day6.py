import numpy as np
import pandas as pd

# Dataset: Pokemon Stats (Numerical combat features, dual-type categoricals, generation markers)
url = "https://gist.githubusercontent.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/raw/92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv"
df = pd.read_csv(url)
df.info()
# ==============================================================================
# AI ENGINEERING PRACTICE: VECTORIZATION, FEATURE GENERATION & PREPROCESSING
# ==============================================================================

# Q1 [Handling Sparse/Optional Categoricals]:
# Check for null values across all columns. The 'Type 2' column has missing values because
# many entities are single-typed. Fill missing 'Type 2' values with 'None' (as a distinct category).
# Your solution:

df.isna().sum()
df["Type 2"] = df["Type 2"].replace({np.nan: None})

# Q2 [Vectorized Combat Score Calculation]:
# Calculate a new feature 'effective_power' using vectorized weighted features:
# `0.4 * Attack + 0.3 * Special_Attack + 0.2 * Speed + 0.1 * HP`.
# Round the result to 2 decimal places.
# Your solution:
df["effective_power"] = (
    (0.4 * df["Attack"])
    + (0.3 * df["Sp. Atk"])
    + (0.2 * df["Speed"])
    + (0.1 * df["HP"])
).round(2)
df["effective_power"]
# Q3 [Multi-Condition Filtering for Training Data]:
# Filter for high-stat non-legendary candidates: Entities where 'Legendary' is False,
# 'Total' >= 500, AND 'Speed' >= 90.
# Your solution:
df.info()
df[(df["Total"] >= 500) & (df["Speed"] >= 90) & (~df["Legendary"])]

# Q4 [Group Aggregation for Feature Baseline]:
# Calculate the mean and standard deviation of 'Attack' grouped by 'Type 1'
# in a single `.agg()` call.
# Your solution:

df.groupby("Type 1")["Attack"].agg(mean_attack="mean", std_attack="std")

# Q5 [Z-Score Outlier Detection]:
# Compute the Z-score for the 'Defense' column using NumPy/Pandas vectorization: `(x - mean) / std`.
# Filter for any entities where the absolute Z-score of 'Defense' is greater than 3.0.
# Your solution:

df["Defense"]
defense_mean = df["Defense"].mean()
defense_std = df["Defense"].std()
# Z-score = `(x - mean) / std`
z_score = (df["Defense"] - defense_mean) / defense_std
(z_score > 3.0).sum()  # gave me 10
df[z_score > 3.0]

# Q6 [Multi-Label Feature Expansion]:
# Create a binary column 'has_dual_type' that is 1 if 'Type 2' is NOT 'None' (or not null),
# and 0 otherwise, using `np.where`.
# Your solution:
# interesting
df['Type 1'].isna().sum() # 0 mean everyone has a type 1

has_dual_type_condition=~df['Type 2'].isna()

df['has_dual_type'] = np.where(has_dual_type_condition,1,0)
df['has_dual_type']
df.head()
# Q7 [Log Transformation for Skewed Features]:
# Applying log transformation helps models handle skewed continuous distributions.
# Create a new column 'log_total' containing `np.log1p(df['Total'])` (log(1 + x)).
# Your solution:


log_total = np.log1p(df['Total'])
log_total2 = np.log1p(1+df['Total'])
# Q8 [Categorical Filtering via Set Membership]:
# Filter the DataFrame to keep only entities where 'Type 1' is in the set
# `['Dragon', 'Psychic', 'Ghost', 'Electric']` using `.isin()`.
# Your solution:

subset_powers_filter = ['Dragon', 'Psychic', 'Ghost', 'Electric']
df[df['Type 1'].isin(subset_powers_filter)]

# Q9 [Standardization / Min-Max Normalization Matrix]:
# Select the numerical columns `['HP', 'Attack', 'Defense', 'Speed']`
# and apply Min-Max scaling `(x - min) / (max - min)` across these columns so all values range from 0 to 1.
# Your solution:
df.info()
numeric_colums=df.select_dtypes(include="number").columns
numeric_min = df[numeric_colums].min()
numeric_max = df[numeric_colums].max()
df[numeric_colums]= (df[numeric_colums] - numeric_min)/(numeric_max-numeric_min)


# Q10 [Train/Test Dataset Partitioning]:
# Perform a deterministic 70/30 train/test split. Shuffle `df` with `random_state=42`,
# assign the first 70% of rows to `train_set` and the remaining 30% to `test_set`.
# Your solution:
sample_frac=df.sample(frac=1,random_state=42)
max_index = int(len(sample_frac)*0.70)
train_set = df.iloc[:max_index]
test_set = df.iloc[max_index:]
