#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

url = "http://localhost:5005/infer"

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = output_dir / "gemma4b-ictihat-ft.json"  # CHANGE THIS

def infer(prompt):
    payload = {
        "system_prompt": "Sen deneyimli bir hukukçusun ve görevin, verilen içtihat metinlerini sadece karar metnine dayanarak özetlemektir. Özet, avukatların dava sürecinde kullanımı için teknik ve tarafsız bir şekilde hazırlanmalıdır. Özetin kişisel yorum, değerlendirme veya çıkarım içermemelidir. Özette yargıtay kararında geçen dava tarihlerini, miktarları, karar tarihini, kanun ve üst mahkeme atıflarını özet içinde birebir muhafaza et. Sadece karar metnine sadık kal, ek bilgi ekleme. Cevap olarak sadece paragraf şeklindeki özetini ver.",
        "user_prompt": prompt,
        "temperature": 1.0,
        "max_tokens": 1024
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response")
    else:
        print("Error:", response.status_code, response.text)
        return None

# Load already saved responses if file exists
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        responses = json.load(f)
else:
    responses = []

# Track already processed inputs
seen_inputs = set(entry["input"] for entry in responses)

# Load dataset
data = load_dataset("turkishnlp/ictihat_summation_30k_gpt41")["eval"]
data = data.select(range(1000))

# Process only new items
new_responses = []
for row in tqdm(data, desc="Inferring"):
    input_text = row["messages"][1]["content"]
    output_text = row["messages"][2]["content"]

    if input_text in seen_inputs:
        continue

    model_response = infer(input_text)
    if model_response is None:
        continue

    model_response = model_response.split("model\n")[-1]
    entry = {
        "input": input_text,
        "output": output_text,
        "model_response": model_response
    }
    responses.append(entry)
    new_responses.append(entry)

    # Save progress incrementally
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)
