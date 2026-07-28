import numpy as np
import pandas as pd

url = "https://gist.githubusercontent.com/clairehq/79acab35be50eaf1c383948ed3fd1129/raw/407a02139ae1e134992b90b4b2b8c329b3d73a6a/winemag-data-130k-v2.csv"
df = pd.read_csv(url)

# ==============================================================================
# DAY 3: PANDAS PRACTICE (Transforms, String Methods, Binning & Pivot Tables)
# ==============================================================================

# Q1: Extract the four-digit year from the 'title' column and store it in a new column 'year'.
# (Hint: You can use .str.extract(r'(\d{4})') or string parsing methods)
# Your solution:
df["year"] = df["title"].str.extract(r"(\d{4})")
df["year"]

# Q2: Convert the 'year' column to a numeric data type (e.g., float or Int64), turning non-valid years into NaN.
# Your solution:

df.year.astype(dtype="Int64")
# Q3: Create a Pivot Table showing the average wine 'price' with 'country' as rows
# and 'points' as columns.
# Your solution:
# wow never in my life created a pivot table

# Q4: Use pd.cut() to divide wine 'points' into 3 bins: 'Low' (80-89), 'Medium' (90-95), and 'High' (96-100).
# Assign this to a new column called 'point_tier'.
# Your solution:

bins_list = [79, 90, 95, 100]

df["point_tier"] = pd.cut(
    df.points, labels=["Low", "Medium", "High"], bins=bins_list, precision=3
)

# Q5: Count the number of wines in each 'point_tier' created in Q4.
# Your solution:
df.point_tier.value_counts()


# Q6: Find all rows where the 'designation' column contains the word "Reserve" or "Selection" (case-insensitive).
# Your solution:

df[
    (
        df.designation.str.contains(pat="Reserve", case=False)
        | df.designation.str.contains(pat="Selection", case=False)
    )
]

# Q7: Group the DataFrame by both 'country' AND 'province', and calculate the mean 'price' and 'points'.
# Reset the index so 'country' and 'province' become regular columns again.
# Your solution:

df.groupby("country", "province")
df.groupby(["country", "province"]).agg(
    {"price": "mean", "points": "mean"}
).reset_index()
# Q8: Calculate the relative percentage frequency of each 'country' in the dataset
# (i.e., normalize value counts to show proportions instead of raw counts).
# Your solution:

# I am not a math genius to understand this yet, I am at mean median sum count min school
# Q9: Replace all occurrences of "US" in the 'country' column with "United States".
# Your solution:

df["country"] = df.country.str.replace(pat="US", repl="Stupid States")

# Q10: Find the 3 most expensive wines for each 'country' using .groupby() and .nlargest().
# Your solution:

df.loc[df.groupby("country").price.nlargest(3).index.get_level_values(1)]["title"]

# sort multiple columns 

df.sort_values(['country','province'],ascending=[True,False])

# transform

mask=df.groupby('country')['province'].transform(lambda x: len(x)>10005)
df[mask]