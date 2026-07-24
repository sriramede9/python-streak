import pandas as pd
import numpy as np

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


# Q2 [RAG Prep - Document Construction]:
# Create a new column 'rag_document' that combines 'Title', 'Director', and 'Description'
# into a formatted string per row: "Title: <Title> | Director: <Director> | Overview: <Description>"
# Your solution:


# Q3 [Token/Context Limit Pruning]:
# Calculate the character length of each 'rag_document' into a column 'char_count'.
# Filter out any documents with fewer than 50 characters.
# Your solution:


# Q4 [NLP Normalization]:
# Create a column 'clean_description' where all text in 'Description' is converted to lowercase
# and all punctuation (periods, commas, exclamations) is stripped out using regex or str methods.
# Your solution:


# Q5 [Feature Engineering - Multi-label Handling]:
# The 'Genre' column contains comma-separated categories (e.g., "Action,Adventure,Sci-Fi").
# Count how many unique genres exist across the entire dataset.
# (Hint: You can split strings and explode them, or use a set).
# Your solution:


# Q6 [Categorical Feature Encoding]:
# Create One-Hot Encoded dummy columns for the top 5 most common primary genres.
# Your solution:


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