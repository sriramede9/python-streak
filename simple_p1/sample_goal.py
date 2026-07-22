import pandas as pd
import chromadb
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import json

# 1. Simulated raw customer feedback data
raw_data = {
    "review_id": [101, 102, 103, 104, 105],
    "rating": [5, 1, 1, None, 4],
    "feedback": [
        "The mobile app is super fast and slick! Love the new dark mode.",
        "N/A",
        "App keeps crashing whenever I try to checkout. Horrible experience.",
        "Great service overall.",
        "The support team took 3 days to reply, but they fixed my billing issue.",
    ],
}

df = pd.DataFrame(raw_data)

print(df)

# 2. Filter out missing ratings OR reviews that are too short (junk)
# Creates a boolean mask checking for non-null ratings AND string length > 10
valid_rows_mask = df["rating"].notna() & (df["feedback"].str.len() > 10)

# 3. Use .loc to keep only valid rows
clean_df = df.loc[valid_rows_mask].copy()

# 4. Create a new column using .loc with condition matching
critical_mask = clean_df["rating"] <= 2
clean_df.loc[critical_mask, "priority"] = "HIGH"
clean_df.loc[~critical_mask, "priority"] = "NORMAL"

print(clean_df[["review_id", "rating", "priority", "feedback"]])

## prepping/chunking data

formatted_chunks = []

for _, row in clean_df.iterrows():
    chunk = f'[Review ID: {row["review_id"]}] | Priority: {row["priority"]} | Rating: {row["rating"]}/5\nContent: "{row["feedback"]}"'
    formatted_chunks.append(chunk)
print(formatted_chunks)

# 3. Combine individual chunks with a divider for batch processing
document_payload = "\n\n---\n\n".join(formatted_chunks)

print(document_payload)


# ==========================================
# 1. OUR REAL TEXT CHUNKS (From Step 2)
# ==========================================
chunks = [
    "[Review 101] The mobile app is super fast and slick! Love the new dark mode.",
    "[Review 103] App keeps crashing whenever I try to checkout. Horrible experience.",
    "[Review 105] The support team took 3 days to reply, but they fixed my billing issue.",
]

metadata_ids = ["review_101", "review_103", "review_105"]

# ==========================================
# 2. INITIALIZE FREE LOCAL VECTOR DATABASE
# ==========================================
# Creates a local vector database in memory
client = chromadb.Client()

# Create a "collection" (think of this like a table in SQL)
collection = client.create_collection(name="customer_reviews")

# ==========================================
# 3. EMBED & STORE TEXT IN VECTOR DB
# ==========================================
# Chroma automatically runs an embedding model locally to vectorize and store these!
collection.add(documents=chunks, ids=metadata_ids)

# ==========================================
# 4. RUN SEMANTIC VECTOR SEARCH
# ==========================================
# Notice: The query word "payment failure" isn't in review 103,
# but the model understands "payment failure" means "checkout crashing"!
user_query = "payment failure during purchase"

results = collection.query(
    query_texts=[user_query],
    n_results=1,  # Get top 1 closest vector match
)
# Ask for top 2 results and include distance scores
results = collection.query(
    query_texts=["payment failure during purchase"],
    n_results=2,
    include=["documents", "distances"],
)

for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"Distance Score: {dist:.4f} | Chunk: {doc}")
# ==========================================
# 5. PRINT RETRIEVED RESULT
# ==========================================
print(f"User Search Query: '{user_query}'\n")
print("Top Relevant Chunk Retrieved:")
print(results["documents"][0][0])


client = genai.Client()


# ==========================================
# 1. DEFINE THE DATA SCHEMA (What you want back)
# ==========================================
class ReviewAnalysis(BaseModel):
    review_id: int = Field(description="The ID of the review analyzed")
    category: str = Field(
        description="Issue category, e.g., Technical Bug, Billing, UI"
    )
    sentiment: str = Field(description="1-2 word summary of user emotion")
    recommended_action: str = Field(
        description="1-sentence recommendation for dev team"
    )


# ==========================================
# 2. CALL LLM WITH STRUCTURED OUTPUT CONFIG
# ==========================================
retrieved_chunk = (
    "[Review 103] App keeps crashing whenever I try to checkout. Horrible experience."
)

prompt = f"Analyze this customer feedback and fill out the schema: {retrieved_chunk}"

for model in client.models.list():
    print(model.name)


response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",  # Force JSON output
        response_schema=ReviewAnalysis,  # Force exact shape
    ),
)

# ==========================================
# 3. LOAD JSON BACK INTO PANDAS DATAFRAME
# ==========================================
# Converts the raw JSON string straight into a Python dictionary
structured_json = json.loads(response.text)

# Convert dictionary straight back into a Pandas DataFrame row!
ai_insights_df = pd.DataFrame([structured_json])

print(ai_insights_df)
