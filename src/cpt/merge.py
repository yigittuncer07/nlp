from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer    
import torch

# config = PeftConfig.from_pretrained("turkishnlp/Llama-3.2-1B-Instruct-CPT-oscar-Unsloth")
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(model=model, model_id="/home/yigittuncer/nlp/artifacts/models/CPT_MODEL")

model = model.merge_and_unload()

model.save_pretrained("/home/yigittuncer/nlp/artifacts/models/MERGED_CPT_MODEL")
