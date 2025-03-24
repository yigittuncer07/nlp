if True:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        # model_name = "unsloth/gemma-3-4b-it", # YOUR MODEL YOU USED FOR TRAINING
        model_name = "./gemma-4-cpt",
        max_seq_length = 2048,
        load_in_4bit = True,
    )

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Merhaba, nasılsın?",}]
}]
print(tokenizer.eos_token)
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)