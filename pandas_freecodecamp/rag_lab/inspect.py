import chromadb

client = chromadb.Client()

collection = client.get_collection("episode_002_chunks")

print("Documents:", collection.count())

data = collection.get(
    include=["documents", "metadatas"]
)

for i, doc in enumerate(data["documents"]):
    print("\nID:", data["ids"][i])
    print("TEXT:", doc)
    print("META:", data["metadatas"][i])