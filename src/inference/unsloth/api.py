from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import torch

app = FastAPI()

model_name = "/home/yigit/nlp/artifacts/models/gemma-3-1b-it-ykd-sft-hr/final" 
# model_name = "google/gemma-3-1b-it"
max_seq_length = 4096
dtype = torch.float16

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

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        use_cache=True,
    )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"response": response}

