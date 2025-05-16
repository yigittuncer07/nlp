from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import torch

app = FastAPI()

model_name = "/media/drive3/yigit_artifacts/gemma3-4b-it-ykd-sft-2-epoch/final" 
#model_name = "unsloth/gemma-3-4b-it"
max_seq_length = 4096
dtype = torch.float32
    
model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=False,
    load_in_8bit=False,
)
FastModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)

class InferenceRequest(BaseModel):
    system_prompt: str = ""
    user_prompt: str
    temperature: float = 1.0
    max_tokens: int = 512

@app.post("/infer")
def infer(req: InferenceRequest):
    messages = []

    if req.system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": req.system_prompt}]
        })

    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": req.user_prompt}]
    })

    # Format input using Unsloth's chat template with generation prompt
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True  # Must add for generation
    )

    # Tokenize the final prompt string
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    # Generate output using recommended Gemma-3 settings
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=0.95,
        top_k=64,
    )

    # Decode and return response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"response": response}


