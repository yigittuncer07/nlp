import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from tqdm import tqdm

DATASET = "turkishnlp/conversation_summarization"

model_id = "turkishnlp/llama3.2-1b-sum-finetune-final"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

def infer(user, system):
    # Prepare the messages
    input_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|><|start_header_id|>user<|end_header_id|>{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # Generate output
    torch.manual_seed(42)
    outputs = model.generate(**inputs,max_new_tokens=256, eos_token_id=tokenizer.eos_token_id)

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract the assistant's response
    if "assistant<|end_header_id|>" in generated_text:
        return generated_text.split("assistant<|end_header_id|>")[-1].strip()
    else:
        return generated_text.strip()

# Load the dataset
dataset = load_dataset(DATASET)
dataset = dataset['test']
results = []
output_filename = f"inferences/{model_id.split('/')[-1]}_mass_inference.json"

# Iterate over the dataset
for entry in tqdm(dataset.select(range(1000))):
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
