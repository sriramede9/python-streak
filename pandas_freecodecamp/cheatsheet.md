# Pandas Core Concepts & Method Cheatsheet

A concise reference guide covering data selection, filtering, aggregation, string manipulation, reshaping, and feature encoding based on practical Pandas exercises.

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

## 6. String Operations (`.str`) & Multi-Label Splitting

```python
# Substring search (Case-insensitive)
df[df["description"].str.contains("cherry", case=False, na=False)]

# Regex extraction (e.g., 4-digit year from title)
df["year"] = df["title"].str.extract(r"(\d{4})")

# Multiple word search with regex OR (|)
df[df["designation"].str.contains("Reserve|Selection", case=False, na=False)]

# Text replacement
df["country"] = df["country"].str.replace("US", "United States", regex=False)

# Multi-label string splitting & counting unique items across all rows
df['Genre'].str.split(",").explode().nunique()
```

---

## 7. Reshaping, Binning & Categorical Encoding

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

### Multi-Label Exploding & One-Hot Dummy Encoding
When turning a delimited string column (e.g., `"Action,Adventure,Sci-Fi"`) into separate binary dummy columns for the **top $N$ categories**:

```python
# 1. Split strings, explode into separate rows, and strip whitespace
exploded = df['Genre'].str.split(',').explode().str.strip()

# 2. Get top 5 most common categories
top_5_genres = exploded.value_counts().head(5).index

# 3. Filter exploded series for top 5 only and generate dummy variables
top_5_exploded = exploded[exploded.isin(top_5_genres)]
dummies = pd.get_dummies(top_5_exploded)

# 4. Collapse exploded rows back to original DataFrame index using level=0
grouped_dummies = dummies.groupby(level=0).sum()

# 5. Drop existing conflicting columns (if any) and join back to original DataFrame
df = df.drop(columns=top_5_genres, errors='ignore').join(grouped_dummies)
```

> ⚠️ **Key Gotcha on `.explode()`:** `.explode()` preserves original index labels across multiple rows. To aggregate dummy encoded features back to individual original rows, group by index level using `.groupby(level=0).sum()`.