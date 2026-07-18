import os
import pandas as pd

print(os.getcwd())

data_path = "data/sales.csv"

if os.path.exists(data_path):
    print("found csv file")
else:
    print("cannot find the path")

df = pd.read_csv(data_path)

print(df.info())

# calculate total for each row
df['total'] = df["quantity"] * df["price"]

print(df['total'])

os.mkdirs('output',exist_ok=True)

# save as different formats
df.to_json("output/sales.json",orient="records",date_format="iso")