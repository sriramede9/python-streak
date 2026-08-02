import pandas as pd
import numpy as np

# Dataset: E-Commerce Women's Clothing Reviews (Text reviews, ratings, recommendations, sentiment features)
url = "https://raw.githubusercontent.com/raghavan-s/Womens-E-Commerce-Clothing-Reviews/master/Womens%20Clothing%20E-Commerce%20Reviews.csv"

df = pd.read_csv(url)

# ==============================================================================
# AI ENGINEERING PRACTICE: NLP PREPROCESSING & TEXT FEATURE EXTRACTION
# ==============================================================================

# Q1 [Data Hygiene & Null Dropping for Text Pipelines]:
# Context: Fine-tuning an NLP sentiment classifier (e.g., BERT/DeBERTa) requires non-empty text strings.
# Business/ML Purpose: Ensure raw training corpus contains valid text entries without null or whitespace-only records.
# Expected Skill: .dropna() on specific text columns and filtering whitespace with .str.strip().
# Task: Drop rows where 'Review Text' is missing (NaN) or contains only blank spaces. Store in `clean_reviews_df`.
# Your solution:


# Q2 [Text Feature Engineering - Word & Character Metrics]:
# Context: Text length features are strong signals for detecting spam, bot-generated reviews, or low-quality training samples.
# Business/ML Purpose: Engineer auxiliary numerical features alongside text for hybrid multimodal architectures.
# Expected Skill: Vectorized string length calculation (.str.len()) and word counting (.str.split().str.len()).
# Task: Create two columns in `clean_reviews_df`: 'char_count' (total characters) and 'word_count' (total words) from 'Review Text'.
# Your solution:


# Q3 [Text Normalization & Token Hygiene]:
# Context: Raw user reviews contain inconsistent casing, punctuation, and special characters that clutter vocabulary dictionaries.
# Business/ML Purpose: Clean text strings prior to classical TF-IDF or Bag-of-Words tokenization.
# Expected Skill: Regex replacement with .str.replace().
# Task: Create a column 'normalized_review' where all text in 'Review Text' is lowercased and all non-alphanumeric characters (except spaces) are removed.
# Your solution:


# Q4 [Binary Target Encoding]:
# Context: Binary classification models predict whether a customer would recommend a product based on review content.
# Business/ML Purpose: Transform ordinal ratings (1 to 5) into a clean binary classification label vector.
# Expected Skill: Vectorized conditional labeling with np.where or boolean assignment.
# Task: Create a binary target column 'is_positive' where 'Rating' >= 4 is labeled 1, and 'Rating' < 4 is labeled 0.
# Your solution:


# Q5 [High-Cardinality Categorical Imputation & Mapping]:
# Context: Missing department names in product catalogs cause pipeline errors during categorical encoding.
# Business/ML Purpose: Handle missing categorical data by creating an explicit 'Unknown' class category.
# Expected Skill: .fillna() on categorical text columns.
# Task: Fill missing values in 'Department Name' and 'Class Name' with the string 'Unassigned'.
# Your solution:


# Q6 [Outlier Removal - Length-Based Sample Filtering]:
# Context: Transformer models (like BERT) have fixed token limits (e.g., 512 tokens). Extreme text outliers waste compute or get truncated.
# Business/ML Purpose: Filter training samples to a stable context window size.
# Expected Skill: Quantile thresholding with .quantile() and boolean indexing.
# Task: Filter `clean_reviews_df` to keep only rows where 'word_count' is between the 5th percentile and 95th percentile of the dataset.
# Your solution:


# Q7 [Text Feature Extraction - Domain Keyword Flagging]:
# Context: Specific domain terms (e.g., fit issues like 'small', 'large', 'tight', 'loose') serve as key signals for e-commerce sentiment.
# Business/ML Purpose: Construct binary domain-specific indicator features for downstream tabular + text models.
# Expected Skill: String pattern matching with .str.contains().
# Task: Create a binary column 'has_fit_issue' that is 1 if 'normalized_review' contains any of the words ('small', 'large', 'tight', 'loose'), and 0 otherwise.
# Your solution:


# Q8 [One-Hot Encoding with Reference Dropping]:
# Context: Categorical metadata ('Department Name') provides key structural context when combined with text embeddings.
# Business/ML Purpose: One-hot encode categorical features while avoiding collinearity (dummy variable trap).
# Expected Skill: pd.get_dummies() with `drop_first=True`.
# Task: Generate one-hot encoded dummy variables for 'Department Name' with `drop_first=True` and concat them back to the main DataFrame.
# Your solution:


# Q9 [Stratified Train/Validation Partitioning]:
# Context: Imbalanced classification targets require stratified splits so train and validation sets maintain identical class distributions.
# Business/ML Purpose: Ensure accurate performance evaluation without class distribution skew between splits.
# Expected Skill: Stratified sampling using Pandas / GroupBy sampling or scikit-learn helper integration.
# Task: Partition `clean_reviews_df` into an 80% train set (`train_df`) and 20% validation set (`val_df`), preserving the ratio of 'is_positive' targets.
# Your solution:


# Q10 [Tabular ML Feature Matrix Export]:
# Context: Preprocessing output must be exported as pure float32 NumPy arrays ready for XGBoost or PyTorch models.
# Business/ML Purpose: Isolate numerical features and target labels into clean arrays, checking for any remaining NaNs.
# Expected Skill: Feature column selection, verification with .isna().sum(), and .to_numpy(dtype=np.float32).
# Task: Select features ['Age', 'Positive Feedback Count', 'char_count', 'word_count', 'has_fit_issue'], verify zero nulls, and convert to 2D float32 array `X` and 1D array `y` ('is_positive').
# Your solution:

