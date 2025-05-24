from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import torch
import json

app = FastAPI()

model_name = "/home/yigit/nlp/artifacts/models/gemma3-4b-it-ictihat/final" 
#model_name = "unsloth/gemma-3-4b-it"
max_seq_length = 2048
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
    seed: int | None = 42 # Add seed parameter with default value 42

import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename="inference_errors.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

@app.post("/infer")
def infer(req: InferenceRequest):
    try:
        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

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

        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = tokenizer([text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=0.95,
                top_k=64,
            )

        response = tokenizer.batch_decode(outputs,truncate=True, skip_special_tokens=True)[0]
        return {"response": response}

    except Exception as e:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "system_prompt": req.system_prompt,
            "user_prompt": req.user_prompt,
        }
        logging.error(json.dumps(log_entry, ensure_ascii=False))
        raise HTTPException(status_code=500, detail="Inference failed")
