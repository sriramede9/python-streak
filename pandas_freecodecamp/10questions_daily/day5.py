import numpy as np
import pandas as pd

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

df["Age"].isna().sum()
df["Cabin"].isna().sum()
df["Embarked"].isna().sum()
median_age_by_class = df.groupby("Pclass")["Age"].median()
fill_map = median_age_by_class.to_dict()

# result of fill_map
# {1: 37.0, 2: 29.0, 3: 24.0}

# now to map these according to class , we have to use dict to map and not assing but present it to fill
df["Age"] = df["Age"].fillna(value=df["Pclass"].map(fill_map))

type(df["Embarked"].mode()[0])  # extract string from series
df["Embarked"] = df["Embarked"].fillna(value=df["Embarked"].mode()[0])

df[df["PassengerId"] == 889]

# Q2 [Categorical Encoding - Binary Mapping]:
# Convert the categorical 'Sex' column into a binary numerical column 'is_female'
# where 'female' = 1 and 'male' = 0 using a vectorized mapping approach or np.where.
# Your solution:

# sex_map = {'female':1,'male':0}
# df['Sex']=df['Sex'].map(sex_map)
# df=df.rename(columns={'Sex':'is_female'})
sex_condition = df["Sex"] == "female"
df["Sex"] = np.where(sex_condition, 1, 0)
df.rename(columns={"Sex": "is_female"}, inplace=True)
df.head()
# Q3 [Feature Engineering - Interaction Term]:
# Create a new feature 'family_size' calculated as 'SibSp' (# of siblings/spouses) + 'Parch' (# of parents/children) + 1.
# Display the first 5 rows with the new feature.
# Your solution:

df["family_size"] = df["SibSp"] + df["Parch"] + 1

# Q4 [Feature Engineering - Extraction from Text]:
# Extract the passenger title (e.g., 'Mr', 'Mrs', 'Miss', 'Master', 'Dr') from the 'Name' column
# using regex (`.str.extract(r'([A-Za-z]+)\.')`) and store it in a column named 'title'.
# Your solution:
df["title"] = df["Name"].str.extract(r"([A-Za-z]+)\.")

# Q5 [Cardinality Reduction]:
# Rare categories can cause overfitting in ML models. In the 'title' column from Q4,
# keep the top 4 most common titles ('Mr', 'Miss', 'Mrs', 'Master') and replace all other titles with 'Rare'.
# Your solution:
normal_titles_list = ["Mr", "Miss", "Mrs", "Master"]
rare_title_condition = df["title"].isin(normal_titles_list)
df["title"] = np.where(rare_title_condition, df["title"], "Rare")
df[df["title"] == "Rare"]


# Q6 [Outlier Removal / Noise Filtering]:
# Machine learning models like Logistic Regression are sensitive to extreme fare values.
# Calculate the 99th percentile of the 'Fare' column and filter out any passengers with fares above this threshold.
# Your solution:

df[df["Fare"] > df["Fare"].quantile(0.99)]

# Q7 [Binning Continuous Features]:
# Use `pd.qcut()` to divide 'Age' into 4 equal-sized quantile bins (quartiles)
# and store the result in a column 'age_group'.
# Your solution:

df["age_group"] = pd.qcut(df["Age"], q=4)
df["age_group"].head()
df.head()
# Q8 [One-Hot Encoding for Model Training]:
# Generate one-hot encoded dummy columns for 'Pclass', 'Embarked', and 'title' (from Q5).
# Set `drop_first=True` to avoid the dummy variable trap.
# Your solution:

df = pd.get_dummies(
    df, columns=["Pclass", "Embarked", "title"], dtype=int, drop_first=True
)

# Q9 [Train/Validation Stratified Split Simulation]:
# Shuffle the DataFrame deterministically using `sample(frac=1, random_state=42)`.
# Then, split it into an 80% training set (`train_df`) and a 20% validation set (`val_df`).
# Your solution:


# Q10 [ML Feature Matrix Export]:
# Drop non-predictive columns ('PassengerId', 'Name', 'Ticket', 'Cabin') and ensure all remaining columns
# are purely numerical. Convert the final DataFrame into a 2D NumPy array `X` ready for model input.
# Your solution:
