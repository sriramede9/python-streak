import pandas as pd
import numpy as np

url = "https://gist.githubusercontent.com/clairehq/79acab35be50eaf1c383948ed3fd1129/raw/407a02139ae1e134992b90b4b2b8c329b3d73a6a/winemag-data-130k-v2.csv"
df = pd.read_csv(url)

# ==============================================================================
# DAY 2: PANDAS PRACTICE (Groupby, Aggregation & Data Cleaning)
# ==============================================================================

# Q1: Find the minimum and maximum wine 'price' for each 'country'.
# Your solution:

df.groupby('country').agg({'price':['min','max']})

# Q2: Calculate both the mean AND median 'points' per 'country' in a single command using .agg().
# Your solution:

df.groupby('country').agg({'points':['mean','median']})

# Q3: Which 'winery' has produced the highest number of wines in this dataset?
# Your solution:

df.winery.value_counts().nlargest(1)
# Q4: Create a new column called 'price_per_point' which is calculated as price / points.
# Display the top 5 rows with the new column.
# Your solution:
df['price_per_point'] = df['price']/df['points']
df.price_per_point.head()
df.head()

# Q5: Find the most expensive wine for each 'province'. Return a DataFrame showing 'province', 'title', and 'price'.
# Your solution:

# df.groupby('province').agg({'price':'max'})
df.iloc[df.dropna(subset='price').groupby('province').price.idxmax()][['province','title','price']]
# Thanks the above is eye opener 


# Q6: Fill all missing (NaN) values in the 'price' column with the median price of the entire dataset.
# (Hint: Use .fillna() on the price column)
# Your solution:

df.fillna({'price':df.price.median()},inplace=True)
# Q7: Count how many unique wine varieties ('variety') exist in the dataset.
# Your solution:

df.variety.nunique()

# Q8: Group by 'country' and filter out any countries that have fewer than 100 wines in the dataset.
# (Hint: Use .groupby() with .filter() or value_counts indexing)
# Your solution:

# damn so filter is the mask stensil you push on grouped data to filter from grouped data
df.groupby('country').filter(lambda x : x.winery.count()>=100)['country'].unique()





# Q9: Create a new column 'value_category' where wines with points >= 95 and price < 20 are labeled 'Great Value',
# and all other wines are labeled 'Standard'.
# (Hint: You can use np.where or a custom function with .apply())
# Your solution:

def value_fun(x):
    # return x
    if (x.points >=95) & (x.price <20 ):
        return 'Great Value'
    else:
        return 'Standard'

    

df['value_category'] = df.apply(lambda x: value_fun(x),axis=1,)
df[df['value_category'] == 'Standard']

# better alternative and super fast way is 
# Verdict: It works! However, using axis=1 with .apply() iterates row-by-row in pure Python, which is very slow on big datasets (~130k rows).
# Pro-tip (np.where): Vectorized operations with np.where are ~100x faster:

condition = (df['points'] >= 95) & (df['price'] < 20)
df['value_category'] = np.where(condition, 'Great Value', 'Standard')
df.head()


# Q10: Find duplicate rows based on 'title' and 'winery', and remove them, keeping only the first occurrence.
# How many total rows remain?
# Your solution:

# find duplicate rows based on title and winery

# Keeps the FIRST occurrence of each ('title', 'winery') pair, drops the rest
df_unique = df.drop_duplicates(subset=['title', 'winery'])   
len(df_unique)