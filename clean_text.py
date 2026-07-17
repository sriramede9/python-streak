import re


def clean_text(text: str) -> str:
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue
                
        if re.fullmatch(r"Page\s+\d+\s+of\s+\d+", line):
            continue

        if line == "Confidential Company Logo":
            continue

        line = re.sub(r"\s+", " ", line)

        lines.append(line)

    merged = []

    for line in lines:
        if not merged:
            merged.append(line)
            continue

        # If previous line doesn't end a sentence,
        # merge this line into it.
        if not re.search(r"[.!?:]$", merged[-1]):
            merged[-1] += " " + line
        else:
            merged.append(line)

    return "\n".join(merged)


with open("data/sample2.txt", mode="r", encoding="utf-8") as data:
    text = data.read()
print(text)

output = clean_text(text)
print(output)
