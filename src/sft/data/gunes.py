import pandas as pd
import json
from pathlib import Path

def clean_text(value):
    if isinstance(value, str):
        return value.strip().replace("\n", " ")
    return ""

def extract_qas(df):
    qas = []
    current = {}

    for _, row in df.iterrows():
        cols = row.values.tolist()
        q_cell = clean_text(cols[2])  # Assuming column C holds the text

        if "Soru:" in q_cell:
            if current:  # Save previous one if exists
                qas.append(current)
                current = {}

            current["id"] = int(cols[1]) if pd.notna(cols[1]) else None
            current["question"] = q_cell.replace("Soru:", "").strip()

        elif "Cevap:" in q_cell:
            current["answer"] = q_cell.replace("Cevap:", "").strip()

        elif "İlgili Yasa:" in q_cell:
            current["law"] = q_cell.replace("İlgili Yasa:", "").strip()

    # Add last item
    if current:
        qas.append(current)

    return qas

def main(xlsx_path):
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    qas = extract_qas(df)

    output_path = Path(xlsx_path).with_suffix(".json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qas, f, ensure_ascii=False, indent=2)

    print(f"[✓] Extracted {len(qas)} Q&A items to {output_path}")

if __name__ == "__main__":
    main("test_questions.xlsx")

