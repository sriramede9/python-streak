import os
import pandas as pd
from helpers import calculate_total,format_currency

print(os.getcwd())

data_path = "data/sales.csv"

if os.path.exists(data_path):
    print("found csv file")
else:
    print("cannot find the path")

df = pd.read_csv(data_path)

print(df.info())

# calculate total for each row

totals =[]
for index,row in df.iterrows():
    total = calculate_total(quantity=row['quantity'],price=row['price'])
    totals.append(total)

print(totals)

# df['total'] = df["quantity"] * df["price"]

# print(df['total'])

##display formatted totals



os.mkdirs('output',exist_ok=True)

# save as different formats
df.to_json("output/sales.json",orient="records",date_format="iso")