import pandas as pd
import numpy as np

# ==============================================================================
# DATASET SETUP
# ==============================================================================
data = {
    "user_id": range(101, 111),
    "primary_genre": [
        "Action",
        "Comedy",
        "Action",
        "Drama",
        "Action",
        "Sci-Fi",
        "Comedy",
        "Horror",
        "Drama",
        "Action",
    ],
    "tags": [
        "Action|Sci-Fi",
        "Comedy",
        "Action|Drama",
        "Drama|Indie",
        "Action|Thriller",
        "Sci-Fi",
        "Comedy|Romance",
        "Horror",
        "Drama",
        "Action|Sci-Fi",
    ],
}
df = pd.DataFrame(data)

print("--- ORIGINAL DATAFRAME ---")
print(df)
print("\n" + "=" * 60 + "\n")


# ==============================================================================
# CHALLENGE 1: The "Top N + Other" Pattern
# ==============================================================================
# Goal:
# 1. Find the top 2 most frequent genres in 'primary_genre'.
# 2. Keep those top 2, but label all other genres as 'Other'.
# 3. One-hot encode the column using pd.get_dummies(..., dtype=int).

top_2_most_frequent_genres_index = df["primary_genre"].value_counts().head(2).index

print(top_2_most_frequent_genres_index)

# wow remember there is something but forgot where is the one had to google it

df["primary_genre"] = np.where(
    df["primary_genre"].isin(top_2_most_frequent_genres_index),
    df["primary_genre"],
    "OTHER",
)

# df.primary_genre.str.replace(pat=())

mathing_dummy_prep = df["primary_genre"].isin(top_2_most_frequent_genres_index)

df = df.join(pd.get_dummies(df["primary_genre"], dtype=int))

# those that are true are 1 and False are other.

df_c1 = df.copy()

# --- YOUR CODE HERE ---


# ----------------------

print("--- CHALLENGE 1 RESULT ---")
print(df_c1)

"""
EXPECTED OUTPUT SUMMARY:
One-hot columns for Action, Comedy, and Other.
Row 3 (Drama) and Row 5 (Sci-Fi) should both have a 1 under 'Other'.
"""
print("\n" + "=" * 60 + "\n")


# ==============================================================================
# CHALLENGE 2: The Delimited Multi-Value String
# ==============================================================================
# Goal:
# Take the 'tags' column with pipe separators ('Action|Sci-Fi') and create
# binary 1/0 columns for every individual tag.
#
# Try solving this two ways:
# Way A: Using `.str.get_dummies('|')`
# Way B: Using `.str.split('|')`, `.explode()`, and `pd.get_dummies()`

df_c2 = df.copy()

# --- YOUR CODE HERE ---

df.tags.str.get_dummies("|")

# approach 2
explode_values = df.tags.str.split("|").explode()


pd.get_dummies(explode_values).groupby(level=0).sum()


# ----------------------

print("--- CHALLENGE 2 RESULT ---")
# print(df_c2)

"""
EXPECTED OUTPUT SUMMARY:
Columns created: Action, Comedy, Drama, Horror, Indie, Romance, Sci-Fi, Thriller.
Row 0 should have 1s under both 'Action' and 'Sci-Fi'.
"""
print("\n" + "=" * 60 + "\n")


# ==============================================================================
# CHALLENGE 3: Frequency / Count Encoding
# ==============================================================================
# Goal:
# Replace each genre in 'primary_genre' with its proportion/percentage of
# appearance across the dataset, saving it into a new column 'genre_frequency'.

df_c3 = df.copy()

# --- YOUR CODE HERE ---


# ----------------------

print("--- CHALLENGE 3 RESULT ---")
# print(df_c3[['primary_genre', 'genre_frequency']])

"""
EXPECTED OUTPUT SUMMARY:
Action appears 4/10 times -> 0.4
Comedy appears 2/10 times -> 0.2
Drama appears 2/10 times  -> 0.2
Sci-Fi appears 1/10 times -> 0.1
Horror appears 1/10 times -> 0.1
"""
