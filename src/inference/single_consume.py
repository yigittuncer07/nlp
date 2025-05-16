#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import requests

url = "http://localhost:5005/infer"  

payload = {
    # "system_prompt": "Sen yardımcı bir asistansın.",
    "user_prompt": "Merhaba, nasilsin? Özetlemeye hazır mısın?",
    # "user_prompt": "Hello! Are you ready to summarize?",
    "temperature": 1.0,
    "max_tokens": 128
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print(result["response"])
else:
    print("Error:", response.status_code, response.text)
