import re
import pandas as pd
from docx import Document


rt = pd.read_csv("data/SPRT_LogLin_216.csv")
true_item_ids = sorted(rt.ITEM.unique())

doc = Document("data/Stimuli_Appendix_format.docx")
table = doc.tables[0]

rows = []
triplet_buffer = []
triplet_index = 0

for row in table.rows:
    cell = row.cells[0]

    raw_text = cell.text.strip().replace("*", "")
    if not raw_text:
        continue

    if raw_text.startswith(("H:", "M:", "L:")):
        critical_word_raw = ""
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                if run.underline:
                    critical_word_raw += run.text

        critical_word = critical_word_raw.strip().strip(".,;!?'\"")
        condition_letter = raw_text[0]
        condition_map = {"H": "HC", "M": "MC", "L": "LC"}
        condition = condition_map[condition_letter]

        sentence_part = raw_text[2:].strip()
        sentence_clean = re.sub(r"\(\d+%?\)", "", sentence_part).strip()

        if critical_word in sentence_clean:
            parts = sentence_clean.rsplit(critical_word, 1)
            prefix = parts[0].rstrip() 
        else:
            prefix = "ERROR" 

        target_llm = " " + critical_word

        triplet_buffer.append({
            "condition": condition,
            "sentence": sentence_clean,
            "prefix": prefix,
            "target_llm": target_llm,
            "critical_word": critical_word
        })

        if len(triplet_buffer) == 3:
            true_item = true_item_ids[triplet_index]
            triplet_index += 1

            for item in triplet_buffer:
                item["ITEM"] = true_item
                rows.append(item)

            triplet_buffer = []

df = pd.DataFrame(rows)

print("Rows:", len(df))
print("Unique ITEM:", df.ITEM.nunique())

errors = df[df['prefix'] == 'ERROR']
if len(errors) > 0:
    print(f"Warning: {len(errors)} rows failed to match the critical word.")

df.to_csv("data/bk21_stimuli.csv", index=False)
print("Stimuli extracted successfully!")