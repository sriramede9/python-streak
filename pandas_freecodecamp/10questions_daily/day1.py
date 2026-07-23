import pandas as pd

url = "https://gist.githubusercontent.com/clairehq/79acab35be50eaf1c383948ed3fd1129/raw/407a02139ae1e134992b90b4b2b8c329b3d73a6a/winemag-data-130k-v2.csv"
df = pd.read_csv(url)

# ==============================================================================
# DAY 1: PANDAS PRACTICE (Filtering, Sorting & Summary Stats)
# ==============================================================================

# Q1: Select only the 'country', 'points', and 'price' columns for the first 10 rows.
# Your solution:

df.iloc[0:10][["country", "points", "price"]]  # position based
df.loc[0:9, ["country", "points", "price"]]  # label based


# Q2: Find all wines from 'Italy' that scored ('points') 95 or higher. How many are there?
# Your solution:

mask_by_country_and_points = (df["country"] == "Italy") & (df["points"] >= 95)

len(df[mask_by_country_and_points])
# or we can call sum
mask_by_country_and_points.sum()

# Q3: Find all wines from either 'US' or 'France' that cost under $15.
# Your solution:

mask_us_france_under_15 = ((df["country"] == "US") | (df["country"] == "France")) & (
    df["price"] < 15
)

df[mask_us_france_under_15]


# Q4: Sort the dataset by 'price' descending (highest first) and display
# the 'title', 'country', and 'price' of the top 5 most expensive wines.
# Your solution:

mask_sort_wine_price_asc = df.sort_values(by="price", ascending=False)

mask_sort_wine_price_asc.iloc[0:4][["title", "country", "price"]]

# mask_sort_wine_price_asc.loc[0:4, ["title", "country", "price"]]
# mask_sort_wine_price_asc.index
# the above two doesn't work because loc is index based and index changed

# Q5: Find all wines where the word "cherry" appears in the 'description' (case-insensitive).
# Your solution:

cherry_flav = df["description"].str.contains(pat="cherry", case=False, na=False)

df[cherry_flav]

# Q6: Count how many rows in the dataset have a missing (NaN) value in the 'price' column.
# Your solution:
df[
    "price"
].isna().sum()  # wow isna gives you how many are not a number and size to count true what a mess

# also learned about df['price'] = df['price'].to_numeric(df['price'],errors="coerce")
# Q7: Calculate the median 'price' and average 'points' across the whole dataset.
# Your solution:


median_price = df.price.median()
average_points = df.points.sum() / len(df.points)
average_points = df["points"].mean()  # doesn't know mean is avg lol
pd.DataFrame([{"median_price": median_price, "avg_points": average_points}])

# Q8: Identify the top 5 most frequent wine varieties ('variety') and how many times each appears.
# Your solution:
# we have 590 variety
df.variety.value_counts().head() # value counts gives uniqueness damn

# .sort_values(by='Unnamed: 0',ascending=False).iloc[0:5][['Unnamed: 0']]


# Q9: Find all perfect-score wines ('points' == 100), then sort them by 'price'
# ascending to find the cheapest 100-point wine.
# Your solution:

hundred_points_filter = df['points'] == 100
df[hundred_points_filter].sort_values(by='price',ascending=True).iloc[0]


# Q10: Calculate the average 'price' of wines grouped by 'country'.
# Which country has the highest average wine price?
# Your solution:
# the lambdas itch from java
df.groupby('country').apply(lambda x: x.price.mean()).idxmax()

# oover use of agg
df.groupby('country').agg(
   {'price':'mean'}
).idxmax()

# alternative
df.groupby('country').price.mean().idxmax()
# top 5 with head
df.groupby('country').price.mean().sort_values(ascending=False).head()
# why sort if nlargest can literally get you top 5?
df.groupby('country').price.mean().nlargest(5)
# we have n smallest values as well
df.groupby('country').price.mean().nsmallest(5)
