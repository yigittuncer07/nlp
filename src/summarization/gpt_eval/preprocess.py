import json
from pathlib import Path

files = ["cohere.json", "gemma3.json", "llama3.json", "llama8.json", "phi4.json"]

data = {fname: json.load(open(fname)) for fname in files}

prompt_sets = {
    fname: set(item["prompt"] for item in items)
    for fname, items in data.items()
}

common_prompts = set.intersection(*prompt_sets.values())

print(f"Common prompts found: {len(common_prompts)}")

filtered_data = {
    fname: [item for item in items if item["prompt"] in common_prompts]
    for fname, items in data.items()
}

for fname, items in filtered_data.items():
    outname = Path(fname).stem + "_filtered.json"
    with open(outname, "w") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
    print(f"Saved: {outname} ({len(items)} items)")
