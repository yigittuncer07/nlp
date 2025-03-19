import dotenv
import torch
from pick import pick
import json
from peft.utils import _get_submodules
import os
import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import copy

def dequantize_model(model, dtype=torch.bfloat16, device="cuda"):
    """
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    'to': directory to save the dequantized model
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """

    cls = bnb.nn.Linear4bit
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)
                quant_state.dtype = dtype
                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)
                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)
                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)
        model.is_loaded_in_4bit = False
        return model


def main():
    push_to_hf = False

    adapters = [adapter for adapter in os.listdir('artifacts/models') if '_merged' not in adapter]

    adapter = pick(adapters)[0]

    adapter_path = f'{os.getenv("MODEL_SAVE_PATH")}/{adapter}'
    output_path = f'{os.getenv("MODEL_SAVE_PATH")}/{adapter_path}_merged'

    if os.path.exists(output_path):
        print("Merged model already exist. Please delete it first.")
        return False

    with open(f'{adapter_path}/adapter_config.json', 'r') as file:
        model_path = json.load(file)['base_model_name_or_path']

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading base model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=os.getenv('MODEL_CACHE'))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        cache_dir=os.getenv('MODEL_CACHE')
    )
    model = dequantize_model(model)

    model = PeftModel.from_pretrained(model=model, model_id=adapter_path)
    model = model.merge_and_unload()
    print("Successfully loaded and merged model, saving...")
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(output_path)

    config_data = json.loads(open(os.path.join(output_path, 'config.json'), 'r').read())
    config_data.pop("quantization_config", None)
    config_data.pop("pretraining_tp", None)
    with open(os.path.join(output_path, 'config.json'), 'w') as config:
        config.write(json.dumps(config_data, indent=4))
    print(f"Model saved: {output_path}")

    #Push model to HuggingFace
    if push_to_hf:
        print(f"Saving to hub ...")
        model.push_to_hub(output_path, use_temp_dir=False)
        tokenizer.push_to_hub(output_path, use_temp_dir=False)
        print("Model successfully pushed to hf.")

    return True

if __name__ == "__main__":
    dotenv.load_dotenv()
    main()