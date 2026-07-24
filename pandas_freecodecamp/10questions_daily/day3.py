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


# Q2: Convert the 'year' column to a numeric data type (e.g., float or Int64), turning non-valid years into NaN.
# Your solution:


# Q3: Create a Pivot Table showing the average wine 'price' with 'country' as rows
# and 'points' as columns.
# Your solution:


# Q4: Use pd.cut() to divide wine 'points' into 3 bins: 'Low' (80-89), 'Medium' (90-95), and 'High' (96-100).
# Assign this to a new column called 'point_tier'.
# Your solution:


# Q5: Count the number of wines in each 'point_tier' created in Q4.
# Your solution:


# Q6: Find all rows where the 'designation' column contains the word "Reserve" or "Selection" (case-insensitive).
# Your solution:


# Q7: Group the DataFrame by both 'country' AND 'province', and calculate the mean 'price' and 'points'.
# Reset the index so 'country' and 'province' become regular columns again.
# Your solution:


# Q8: Calculate the relative percentage frequency of each 'country' in the dataset
# (i.e., normalize value counts to show proportions instead of raw counts).
# Your solution:


# Q9: Replace all occurrences of "US" in the 'country' column with "United States".
# Your solution:


# Q10: Find the 3 most expensive wines for each 'country' using .groupby() and .nlargest().
# Your solution:
