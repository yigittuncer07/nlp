import json
import os
from unsloth import FastModel
from transformers import TextStreamer
from tqdm import tqdm
model_name = "/media/drive1/gemma-4-cpt-sft/final"
# model_name = "unsloth/gemma-3-4b-it"
input_file = "input_messages.json"

model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
)

with open(input_file, "r", encoding="utf-8") as f:
    sentences = json.load(f)
    
sentences = [item.get('prompt') for item in sentences]

messages_list = [{"role": "user", "content": [{"type": "text", "text": "Soruya en fazla 2 cumle ile cevap ver:\n" + sentence}]} for sentence in sentences]

os.makedirs("outputs", exist_ok=True)

output_data = {"model": model_name, "inferences": []}
output_file = f"outputs/{model_name.split("/")[-1]}_chat_output.json"

for messages in tqdm(messages_list):
    text = tokenizer.apply_chat_template([messages], add_generation_prompt=True)

    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=512 * 4,
        temperature=1.0, top_p=0.95, top_k=64,
    )
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    output_text = output_text.split("model\n")[1]
    print("Input:", messages["content"][0]["text"])
    print("Output:", output_text)
    output_data["inferences"].append({
        "user_query": messages["content"][0]["text"],
        "llm_response": output_text
    })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Output saved to {output_file}")
