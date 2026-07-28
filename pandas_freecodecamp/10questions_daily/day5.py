import pandas as pd
import numpy as np

# Dataset: Titanic Passenger Manifest (Binary classification, missing values, categorical features)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ==============================================================================
# AI ENGINEERING PRACTICE: FEATURE ENGINEERING & DATASET PREPARATION
# ==============================================================================

# Q1 [Imputation / Data Integrity]:
# Check for null values in 'Age', 'Cabin', and 'Embarked'. 
# Impute missing 'Age' values with the median age grouped by 'Pclass' (Passenger Class), 
# and fill missing 'Embarked' values with the mode (most common value).
# Your solution:


# Q2 [Categorical Encoding - Binary Mapping]:
# Convert the categorical 'Sex' column into a binary numerical column 'is_female' 
# where 'female' = 1 and 'male' = 0 using a vectorized mapping approach or np.where.
# Your solution:


# Q3 [Feature Engineering - Interaction Term]:
# Create a new feature 'family_size' calculated as 'SibSp' (# of siblings/spouses) + 'Parch' (# of parents/children) + 1.
# Display the first 5 rows with the new feature.
# Your solution:


# Q4 [Feature Engineering - Extraction from Text]:
# Extract the passenger title (e.g., 'Mr', 'Mrs', 'Miss', 'Master', 'Dr') from the 'Name' column 
# using regex (`.str.extract(r'([A-Za-z]+)\.')`) and store it in a column named 'title'.
# Your solution:


# Q5 [Cardinality Reduction]:
# Rare categories can cause overfitting in ML models. In the 'title' column from Q4, 
# keep the top 4 most common titles ('Mr', 'Miss', 'Mrs', 'Master') and replace all other titles with 'Rare'.
# Your solution:


# Q6 [Outlier Removal / Noise Filtering]:
# Machine learning models like Logistic Regression are sensitive to extreme fare values. 
# Calculate the 99th percentile of the 'Fare' column and filter out any passengers with fares above this threshold.
# Your solution:


# Q7 [Binning Continuous Features]:
# Use `pd.qcut()` to divide 'Age' into 4 equal-sized quantile bins (quartiles) 
# and store the result in a column 'age_group'.
# Your solution:


# Q8 [One-Hot Encoding for Model Training]:
# Generate one-hot encoded dummy columns for 'Pclass', 'Embarked', and 'title' (from Q5). 
# Set `drop_first=True` to avoid the dummy variable trap.
# Your solution:


# Q9 [Train/Validation Stratified Split Simulation]:
# Shuffle the DataFrame deterministically using `sample(frac=1, random_state=42)`. 
# Then, split it into an 80% training set (`train_df`) and a 20% validation set (`val_df`).
# Your solution:


# Q10 [ML Feature Matrix Export]:
# Drop non-predictive columns ('PassengerId', 'Name', 'Ticket', 'Cabin') and ensure all remaining columns 
# are purely numerical. Convert the final DataFrame into a 2D NumPy array `X` ready for model input.
# Your solution: