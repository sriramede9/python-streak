import pandas as pd
import numpy as np

import os

print(os.getcwd())

print(os.path.exists("../data/sample_employees.csv"))

df = pd.read_csv("../data/sample_employees.csv")


df.info()
# check last before row which is empty
print(df.iloc[-2])
# Remove completely empty rows
print("Before:", len(df))

df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

df = df.replace("", np.nan)

df = df.dropna(how="all")

# check if last before row is dropped
print(df.iloc[-2])

print("After:", len(df))


# 1. FORCE the column names to strip spaces and overwrite 'df' permanently
df.columns = df.columns.str.strip()

print(df.iloc[1])
# Fill missing text values
text_columns = df.select_dtypes(include="object").columns
df[text_columns] = df[text_columns].fillna("Unknown")

# Find all columns that hold numbers
# Grabs numeric columns by stripping the headers and converting data on the fly



# 2. Double-check what pandas actually sees right this millisecond
print("Current columns available:", df.columns.tolist())

# 3. Safe conversion using get() to prevent the script from crashing
if 'Salary' in df.columns:
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    print("Successfully converted Salary!")
else:
    print("WARNING: 'Salary' was not found. Look at the list above to see its exact spelling.")

if 'EmployeeID' in df.columns:
    df['EmployeeID'] = pd.to_numeric(df['EmployeeID'], errors='coerce')
    print("Successfully converted EmployeeID!")

# 4. Check your numeric count again
numeric_columns = df.select_dtypes(include="number").columns
print(f"Total numeric columns now: {len(numeric_columns)}")


df[numeric_columns] = df[numeric_columns].fillna(0)

# Convert each row into a sentence
print(df.columns.tolist())
documents = []

for _, row in df.iterrows():
    text = (
        f"{row['Name']} works as a {row['JobTitle']} in the "
        f"{row['Department']} department in {row['Location']}. "
        f"Salary: {row['Salary']}. "
        f"Started: {row['StartDate']}. "
        f"Manager: {row['Manager']}. "
        f"Email: {row['Email']}. "
        f"Notes: {row['Notes']}."
    )
    documents.append(text)
    
# avoid for loop 

# Strip all column names once at the top
df.columns = df.columns.str.strip()

# Combine entire columns instantly!
df['document'] = (
    df['Name'] + " works as a " + df['JobTitle'] + " in the " +
    df['Department'] + " department in " + df['Location'] + ". " +
    "Salary: " + df['Salary'].astype(str) + ". " +
    "Started: " + df['StartDate'].astype(str) + ". " +
    "Manager: " + df['Manager'].astype(str) + ". " +
    "Email: " + df['Email'] + ". " +
    "Notes: " + df['Notes'] + "."
)

# Pull the list of sentences out if needed
documents = df['document'].tolist()


print(documents[:len(documents)])
