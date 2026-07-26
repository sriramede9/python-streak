import numpy as np
import pandas as pd

# Dataset: IMDB Top 1000 Movie Metadata
url = "https://raw.githubusercontent.com/LearnDataSci/articles/master/Python%20Pandas%20Tutorial%20A%20Complete%20Introduction%20for%20Beginners/IMDB-Movie-Data.csv"
df = pd.read_csv(url)

# ==============================================================================
# AI ENGINEERING PRACTICE: DATA PREPROCESSING & DATASET CURATION
# ==============================================================================

# Q1 [Data Validation / Imputation]:
# Check for null values. Fill missing 'Revenue (Millions)' with 0,
# and fill missing 'Metascore' with the median value of the column (to prevent model training crashes).
# Your solution:
df["Revenue (Millions)"].fillna(value=0)
df["Metascore"].fillna(value=df.Metascore.median())

# Q2 [RAG Prep - Document Construction]:
# Create a new column 'rag_document' that combines 'Title', 'Director', and 'Description'
# into a formatted string per row: "Title: <Title> | Director: <Director> | Overview: <Description>"
# Your solution:

df["rag_document"] = (
    "Title:"
    + df["Title"].fillna("")
    + " | Director: "
    + df["Director"].fillna("")
    + " | Overview: "
    + df["Description"].fillna("")
)

df["rag_document"].head(1).tolist()

# Q3 [Token/Context Limit Pruning]:
# Calculate the character length of each 'rag_document' into a column 'char_count'.
# Filter out any documents with fewer than 50 characters.
# Your solution:
df["char_count"] = df.rag_document.str.len()
df[df["char_count"] < 50]

# Q4 [NLP Normalization]:
# Create a column 'clean_description' where all text in 'Description' is converted to lowercase
# and all punctuation (periods, commas, exclamations) is stripped out using regex or str methods.
# Your solution:
df["clean_description"] = (
    df["Description"].str.replace(pat=r"[^\w\s]", repl="").str.lower()
)
df.head(1)

# Q5 [Feature Engineering - Multi-label Handling]:
# The 'Genre' column contains comma-separated categories (e.g., "Action,Adventure,Sci-Fi").
# Count how many unique genres exist across the entire dataset.
# (Hint: You can split strings and explode them, or use a set).
# Your solution:
df['Genre'].str.split(",").explode().nunique()

# Q6 [Categorical Feature Encoding]:
# Create One-Hot Encoded dummy columns for the top 5 most common primary genres.
# Your solution:
# 1. Get top 5 genre names
dummies = df['Genre'].str.replace(r'\s*,\s*', ',', regex=True).str.get_dummies(',')
dummies.head()
top_5_genres = dummies.sum().nlargest(5).index
df=df.join(dummies[top_5_genres])
df.head()
# so the above guy exploded into let's say first row has 
#Action,Adventure,Sci-Fi

# the top 5 genres are Drama, Action, Comedy , Adventure , Thriller
# Action - index - 0
# Adventure - index - 0
# Sci-Fi - index - 0
# when you ask qualified in top 5 it gives
# Action - index - 0 - True
# Adventure - index - 0 -True
# Sci-Fi - index - 0 - False

# when you group by level 0 , they are staying together as group




# Q7 [Outlier Removal / Pruning]:
# Filter the dataset to keep only movies where 'Runtime (Minutes)' is within 2 standard deviations of the mean runtime.
# (Formula: mean - 2*std <= runtime <= mean + 2*std)
# Your solution:


# Q8 [Training Quality Thresholding]:
# Filter for movies that have BOTH 'Rating' >= 8.0 AND 'Votes' >= 100,000 to construct a high-quality fine-tuning set.
# How many rows qualify?
# Your solution:


# Q9 [Batch Processing Simulation]:
# Split the DataFrame into chunks of 100 rows each and store them in a list of DataFrames called 'batches'.
# Your solution:


# Q10 [Vector DB Payload Construction]:
# Convert the top 5 rows of the DataFrame into a list of Python dictionaries formatted for API payload ingestion:
# [{'id': Rank, 'text': rag_document, 'metadata': {'rating': Rating, 'year': Year}}]
# Your solution:
