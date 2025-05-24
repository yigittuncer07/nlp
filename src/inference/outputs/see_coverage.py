import os
import json
from datasets import load_dataset
from tqdm import tqdm

# Load 1000 eval inputs
dataset = load_dataset("turkishnlp/ictihat_summation_30k_gpt41")["eval"]
eval_inputs = set(row["messages"][1]["content"] for row in dataset.select(range(1000)))

# Check each JSON file in the current directory
for file in os.listdir("."):
    if not file.endswith(".json"):
        continue

    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            file_inputs = set(entry.get("input") for entry in data if "input" in entry)

        overlap = eval_inputs & file_inputs
        coverage = len(overlap) / len(eval_inputs) * 100
        print(f"{file}: Covered {len(overlap)}/{len(eval_inputs)} inputs ({coverage:.2f}%)")

    except Exception as e:
        print(f"⚠️ Failed to process {file}: {e}")
