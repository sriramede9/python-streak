import numpy as np
import pandas as pd

url = "https://gist.githubusercontent.com/clairehq/79acab35be50eaf1c383948ed3fd1129/raw/407a02139ae1e134992b90b4b2b8c329b3d73a6a/winemag-data-130k-v2.csv"
df = pd.read_csv(url)

# ==============================================================================
# DAY 2: PANDAS PRACTICE (Groupby, Aggregation & Data Cleaning)
# ==============================================================================

# Q1: Find the minimum and maximum wine 'price' for each 'country'.
# Your solution:

df.groupby("country").agg({"price": ["min", "max"]})
# Q2: Calculate both the mean AND median 'points' per 'country' in a single command using .agg().
# Your solution:

df.groupby("country").agg({"points": ["mean", "median"]})


# Q3: Which 'winery' has produced the highest number of wines in this dataset?
# Your solution:

df.winery.value_counts().idxmax()

# Q4: Create a new column called 'price_per_point' which is calculated as price / points.
# Display the top 5 rows with the new column.
# Your solution:

df["price_per_point"] = df.price / df.points
df["price_per_point"].head()
# Q5: Find the most expensive wine for each 'province'. Return a DataFrame showing 'province', 'title', and 'price'.
# Your solution:
# remember to use subset when droping selected column to use idxmax, this is a brilliant problem to think before act
df.loc[df.dropna(subset="price").groupby("province")["price"].idxmax()][
    ["province", "title", "price"]
]

# Q6: Fill all missing (NaN) values in the 'price' column with the median price of the entire dataset.
# Your solution:
df.price.fillna(value=df.price.median(), inplace=True)

# Q7: Count how many unique wine varieties ('variety') exist in the dataset.
# Your solution:

len(df.variety.value_counts())

# Q8: Group by 'country' and filter out any countries that have fewer than 100 wines in the dataset.
# Your solution:

df.groupby(by="country").filter(lambda x: x.winery.count() > 100)

# Q9: Create a new column 'value_category' where wines with points >= 95 and price < 20 are labeled 'Great Value',
# and all other wines are labeled 'Standard'.
# Your solution:

def value_category_func(val):
    return (val.points >= 95 and val.price < 20)


df['value_category'] = np.where(value_category_func,'Great Value','Standard')
#df.value_category.head()
# Q10: Find duplicate rows based on 'title' and 'winery', and remove them, keeping only the first occurrence.
# How many total rows remain?
# Your solution:
dropped_duplicate_title_and_wine=df.drop_duplicates(subset=['title','winery'])
len(dropped_duplicate_title_and_wine)