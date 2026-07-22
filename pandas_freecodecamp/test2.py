import pandas as pd

url = "https://gist.githubusercontent.com/clairehq/79acab35be50eaf1c383948ed3fd1129/raw/407a02139ae1e134992b90b4b2b8c329b3d73a6a/winemag-data-130k-v2.csv"

df = pd.read_csv(url)

print(df.shape)
print(df.head())

# group by points and then count how many times the points appeared

points_group =df.groupby('points')

len(points_group.head(1)) # 21 * 1 21 values selected

print(df.points.nunique()) # we have 21 unique points

points_group_aggregate_count = df.groupby('points').points.count()
points_group_aggregate_count()

# chepest wine in each points group
chepeast_in_each_group=df.groupby('points').price.min()
chepeast_in_each_group.head(21)

# we can also apply lambda functions on the group we selected
# selecting the name of the first wine reviewed from each winery in the dataset
df.groupby('winery').nunique()
df.winery.nunique() # there are 13549 wineries for now

# we want the name of first wine reviews by each winery
df.groupby('winery').first()['title']

# if use head(1)['title'] we are selecting rows and then keeping only names

# to use lambdas
df.groupby('winery').apply(lambda w: w.title.iloc[0])

#linkcode
# For even more fine-grained control, you can also group by more than one column. For an example, here's how we would pick out the best wine by country and province:
df.groupby(['country','province']).apply(lambda w: w.loc[w.points.idxmax()])

# mentioning is agg(), which lets you run a bunch of different functions on your DataFrame simultaneously. For example, we can generate a simple statistical summary of the dataset as follows

df.groupby('country').price.agg([len, min, max])

# count how many each taster reviewd
df.head()
df.groupby('taster_name').size()

# how many provinces under each country are reviewed?

df.groupby("country")["province"].nunique()
# How many wineries per country?
df.groupby("country")["winery"].nunique()

# How many tasters per country?
df.groupby("country")["taster_name"].nunique()

# How many varieties per winery?
df.groupby("winery")["variety"].nunique()

# "For each country AND province, how many wineries are there?"

df.groupby(['country','province'])['winery'].nunique()

df.groupby(["country", "province"]).agg({
    "points": "mean",
    "price": "mean",
    "winery": "nunique"
})

# show countries that has more than 100 wineries

df.groupby('country').filter(lambda x: x.winery.nunique() >100)
df.groupby("country").filter(lambda x: x["winery"].nunique() > 100)

df.groupby("country")["winery"].nunique() > 100)
