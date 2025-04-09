import requests

url = "http://localhost:8000/infer"  

payload = {
    "system_prompt": "Sen yardımcı bir asistansın, verilen metni özetle.",
    "user_prompt": "Merhaba, nasilsin.",
    "temperature": 0.1,
    "max_tokens": 64
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print("User Prompt:", payload["user_prompt"])
    print("Response:", result["response"])
else:
    print("Error:", response.status_code, response.text)
