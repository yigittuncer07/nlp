#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
from datasets import load_from_disk
from tqdm import tqdm
from pathlib import Path


url = "http://localhost:8003/infer"  

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE=f"{output_dir}/gemma-1-hr.json" # CHANGE THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def infer(prompt):
    payload = {
        "system_prompt": "Sen yardımcı bir asistansın, verilen metni özetle.",
        "user_prompt": f"{prompt}",
        "temperature": 1.0,
        "max_tokens": 1024
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result["response"]
    else:
        print("Error:", response.status_code, response.text)

data = load_from_disk("/home/yigit/nlp/artifacts/datasets/summarization-ykd-data")
data = data["test"]
data = data.select(range(50))

responses = []
for row in tqdm(data, "I am inferring it"):
    response = infer(row["input"])
    if response is None:
        continue
    response = response.split("model\n")[-1]
    responses.append({
        "input": row["input"],
        "output": row["output"],
        "model_response": response
    })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)
