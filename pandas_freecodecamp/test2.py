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
