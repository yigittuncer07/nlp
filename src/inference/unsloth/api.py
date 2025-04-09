from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

app = FastAPI()

model_name = "/home/nlp/projects/nlp/artifacts/models/Llama-3.2-3B-Instruct/final"
max_seq_length = 4096
dtype = torch.float16
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

class InferenceRequest(BaseModel):
    system_prompt: str = ""
    user_prompt: str
    temperature: float = 1.0
    max_tokens: int = 512

@app.post("/infer")
def infer(req: InferenceRequest):
    try:
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages.append({"role": "user", "content": req.user_prompt})

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
