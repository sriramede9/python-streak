import pandas as pd

# Load open-source English word dictionary directly from GitHub
url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
df = pd.read_csv(url, names=["word"], keep_default_na=False)

# Save to Excel or CSV if you want a local file
df.to_excel("english_words.xlsx", index=False)
df.to_csv("english_words.csv", index=False)

scores = {
    **dict.fromkeys(list("AEILNORSTU"), 1),
    **dict.fromkeys(list("DG"), 2),
    **dict.fromkeys(list("BCMP"), 3),
    **dict.fromkeys(list("FHVWY"), 4),
    "K": 5,
    **dict.fromkeys(list("JX"), 8),
    **dict.fromkeys(list("QZ"), 10),
}
df.info()

df["Char_Count"] = df["word"].str.len()

df["Value"] = df["word"].str.upper().apply(
    lambda word: sum(scores[c] for c in word)
)

df = df.set_index("word")
df.head()
df.info()
df.shape

df.loc["microscope","Value"]

# highest possible value of a word?
df['Value'].max()
df['Value'].idxmax()
df.loc[df['Value'].idxmax()]

# who has the highest character count
df['Char_Count'].max()
df.loc[df['Char_Count'].idxmax()]

# find df's whose value is equal to 41
df[df["Value"] == 41]
