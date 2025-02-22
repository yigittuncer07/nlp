import torch
from transformers import pipeline
from datasets import load_dataset
import json
from tqdm import tqdm
DATASET = "turkishnlp/conversation_summarization"


model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipe = pipeline(model=model_id, torch_dtype=torch.bfloat16, device_map="auto")

def infer(user, system):

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=512,
    )
    return outputs[0]["generated_text"][-1].get('content')

dataset = load_dataset(DATASET)
dataset = dataset['test']
results = []
output_filename = f"{model_id.split('/')[-1]}_mass_inference.json"

for entry in tqdm(dataset):
    system_prompt = entry.get('messages')[0].get('content')
    user_prompt = entry.get('messages')[1].get('content')
    expected_response = entry.get('messages')[2].get('content')
    generated_response = infer(user_prompt, system_prompt)
    result = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "expected_response": expected_response,
        "generated_response": generated_response
    }
    results.append(result)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

