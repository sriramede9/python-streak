# Pandas Core Concepts & Method Cheatsheet

A concise reference guide covering data selection, filtering, aggregation, string manipulation, reshaping, and categorical feature encoding based on practical Pandas exercises.

---

## 1. Selection & Indexing

| Method | Behavior | Example |
| :--- | :--- | :--- |
| `.iloc[]` | Position-based (0-indexed, endpoint exclusive) | `df.iloc[0:10, [0, 1, 2]]` |
| `.loc[]` | Label-based (endpoint inclusive) | `df.loc[0:9, ["country", "points"]]` |

> ⚠️ **Key Gotcha:** After sorting or filtering, index labels change order! Using `.loc[0:4]` will look for rows with literal index labels `0..4`, not necessarily the top 5 rows. Use `.iloc[0:5]` or `.head()` for top rows after sorting.

---

## 2. Filtering & Boolean Logic

Combine boolean masks using `&` (AND), `|` (OR), and `~` (NOT). **Always wrap individual conditions in parentheses.**

```python
# Multiple conditions
mask = (df["country"] == "Italy") & (df["points"] >= 95)
filtered_df = df[mask]

# Counting matching rows
total_matches = mask.sum()  # Booleans convert to 1 (True) and 0 (False)
```

---

## 3. Summary Stats & Frequency

```python
# Basics
median_price = df["price"].median()
avg_points = df["points"].mean()
unique_count = df["variety"].nunique()

# Value counts & proportions
df["variety"].value_counts().head(5)

# Get relative proportions / percentages (Normalize)
df["country"].value_counts(normalize=True)  # Returns relative proportions (0.0 to 1.0)
df["country"].value_counts(normalize=True) * 100  # Percentage format
```

---

## 4. Grouping, Aggregations & Extremes

### Aggregation with `.agg()`
```python
# Multiple summary stats on one or more columns
df.groupby("country").agg({"price": ["min", "max"], "points": ["mean", "median"]})

# Group by multiple columns and reset index to flattened DataFrame
df.groupby(["country", "province"]).agg({"price": "mean", "points": "mean"}).reset_index()
```

### Finding Extremes & Indices
```python
# Get single index label of max/min value
top_country = df.groupby("country")["price"].mean().idxmax()

# Get Top N / Bottom N directly (faster and cleaner than sort_values + head)
df.groupby("country")["price"].mean().nlargest(5)
df.groupby("country")["price"].mean().nsmallest(5)

# Extract full rows corresponding to max per group
max_price_idx = df.dropna(subset="price").groupby("province")["price"].idxmax()
df.loc[max_price_idx, ["province", "title", "price"]]
```

### Filtering Groups with `.filter()`
```python
# Keep only groups that meet a specific condition (e.g., countries with >100 entries)
df.groupby("country").filter(lambda x: len(x) > 100)
```

---

## 5. Data Cleaning & Transformations

```python
# Missing Values
df["price"].isna().sum()  # Count NaNs
df["price"].fillna(value=df["price"].median(), inplace=True)  # Fill missing values

# Type Casting & Parsing
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["year"] = df["year"].astype("Int64")  # Nullable integer type

# Vectorized Conditional Column Creation (np.where)
# Structure: np.where(condition, value_if_true, value_if_false)
df["value_category"] = np.where(
    (df["points"] >= 95) & (df["price"] < 20), "Great Value", "Standard"
)

# Deduplication
df.drop_duplicates(subset=["title", "winery"], keep="first")
```

---

## 6. String Operations (`.str`)

```python
# Substring search (Case-insensitive)
df[df["description"].str.contains("cherry", case=False, na=False)]

# Regex extraction (e.g., 4-digit year from title)
df["year"] = df["title"].str.extract(r"(\d{4})")

# Multiple word search with regex OR (|)
df[df["designation"].str.contains("Reserve|Selection", case=False, na=False)]

# Text replacement
df["country"] = df["country"].str.replace("US", "United States", regex=False)

# Count unique tokens across delimited strings
df["tags"].str.split("|").explode().nunique()
```

---

## 7. Categorical Feature Encoding Patterns

### Pattern A: "Top N + Other" Bucketing & One-Hot Encoding
Keep high-frequency categories and group rare values into an `"OTHER"` category to avoid dimension explosion before one-hot encoding.

```python
# 1. Identify top N categories
top_2 = df["primary_genre"].value_counts().head(2).index

# 2. Replace non-top categories with 'OTHER'
df["primary_genre"] = np.where(
    df["primary_genre"].isin(top_2), df["primary_genre"], "OTHER"
)

# 3. One-hot encode with integer output
df = df.join(pd.get_dummies(df["primary_genre"], dtype=int))
```

### Pattern B: Delimited Multi-Value String Encoding
When a single column contains multiple categories joined by a delimiter (e.g., `"Action|Sci-Fi"`):

#### Option 1: Native `.str.get_dummies()` (Cleanest & fastest)
```python
# Directly generates binary 1/0 columns for each delimited tag
tag_dummies = df["tags"].str.get_dummies(sep="|")
df = df.join(tag_dummies)
```

#### Option 2: `.explode()` + `pd.get_dummies()` (Ideal when filtering top N categories first)
```python
# Explode -> get dummies -> collapse back to original row index
exploded = df["tags"].str.split("|").explode().str.strip()
dummies = pd.get_dummies(exploded, dtype=int)
grouped_dummies = dummies.groupby(level=0).sum()

df = df.join(grouped_dummies)
```

> ⚠️ **Key Gotcha on `.explode()`:** `.explode()` preserves original row index labels across multiple split rows. To collapse dummy-encoded exploded rows back to individual original rows, group by index level using `.groupby(level=0).sum()`.

### Pattern C: Frequency / Count Encoding
Replace categorical text values with their relative proportion (or raw frequency count) in the dataset to convert high-cardinality text into a single continuous numerical feature without introducing extra dummy columns.

```python
# Map category proportions (0.0 to 1.0) to rows
genre_proportions = df["primary_genre"].value_counts(normalize=True)
df["genre_frequency"] = df["primary_genre"].map(genre_proportions)

# For raw count encoding instead:
genre_counts = df["primary_genre"].value_counts()
df["genre_count"] = df["primary_genre"].map(genre_counts)
```

---

## 8. Reshaping & Binning

### Binning Continuous Data with `pd.cut()`
```python
bins = [79, 90, 95, 100]
labels = ["Low", "Medium", "High"]

df["point_tier"] = pd.cut(df["points"], bins=bins, labels=labels)
```

### Pivot Tables
```python
# Create pivot table: rows=country, cols=points, values=mean price
pivot = df.pivot_table(
    index="country", columns="points", values="price", aggfunc="mean"
)
```