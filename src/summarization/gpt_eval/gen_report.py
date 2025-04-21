#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from statistics import mean, median, stdev

output_dir = Path("outputs")
summary_path = output_dir / "summary.md"

results = []

for file_path in sorted(output_dir.glob("*.json")):
    with open(file_path) as f:
        data = json.load(f)

    scores = [item["gpt_score"] for item in data if isinstance(item["gpt_score"], int)]

    if not scores:
        continue

    avg_score = mean(scores)
    median_score = median(scores)
    std_dev = stdev(scores) if len(scores) > 1 else 0.0
    model_name = file_path.stem.replace("_filtered", "")

    results.append({
        "model_name": model_name,
        "avg_score": avg_score,
        "median_score": median_score,
        "std_dev": std_dev,
    })

# Sort by average score descending
results.sort(key=lambda x: x["avg_score"], reverse=True)

# Generate markdown
markdown_lines = ["# GPT Evaluation Summary\n"]
for res in results:
    markdown_lines.append(f"## {res['model_name']}")
    markdown_lines.append(f"- **Average Score**: {res['avg_score']:.2f}")
    markdown_lines.append(f"- **Median Score**: {res['median_score']}")
    markdown_lines.append(f"- **Standard Deviation**: {res['std_dev']:.2f}\n")

# Save summary
with open(summary_path, "w") as f:
    f.write("\n".join(markdown_lines))

print(f"Markdown summary saved to {summary_path}")
