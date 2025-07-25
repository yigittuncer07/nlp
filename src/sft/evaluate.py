#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run local inference on a HuggingFace / Unsloth‐compatible model
and save the predictions incrementally to data/evals/<model>_eval.json.

Usage
-----
python evaluate.py /path/to/model /path/to/prompts.json \
    --save_dir ./ --dtype bfloat16 --max_tokens 1024 --temperature 1.0 --limit 10
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

SYSTEM_PROMPT = (
    "Sen sorulan sorulara cevap veren bir yapay zeka asistanısın. "
)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline evaluator with JSON prompts and incremental saving.")
    p.add_argument("model_path", type=str, help="Checkpoint or Hub repo to load.")
    p.add_argument("input_json", type=str, help="Path to JSON file containing prompts (list of strings or objects with 'prompt').")
    p.add_argument("--save_dir", default="./", help="Where to put *_eval.json.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate.")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    p.add_argument("--limit", type=int, default=None, help="If set, only process the first N prompts.")
    return p

def load_or_create_output(path: Path) -> list[dict]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_json(path: Path, obj: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

@torch.inference_mode()
def main() -> None:
    args = build_argparser().parse_args()

    # Logging setup
    logging.basicConfig(
        filename="evaluate_errors.log",
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
    )

    # Prepare output file
    eval_dir = Path(args.save_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Use entire model_path, replacing slashes with underscores for a safe filename
    safe_name = args.model_path.replace("/", "_")
    out_file = eval_dir / f"{safe_name}_eval.json"
    results = load_or_create_output(out_file)
    seen_inputs = {r["input"] for r in results}

    # Load model & tokenizer
    print("Loading model …")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        dtype=getattr(torch, args.dtype),
        load_in_4bit=False,
        load_in_8bit=False,
    )
    FastModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")  # adjust template as needed

    # Load prompts JSON
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for item in data:
        if isinstance(item, dict) and "question" in item:
            prompt = item["question"]
            reference = item.get("answer", None)
            items.append({"input": prompt, "reference_response": reference})
        elif isinstance(item, dict) and "prompt" in item:
            items.append({"input": item["prompt"], "reference_response": None})
        elif isinstance(item, str):
            items.append({"input": item, "reference_response": None})

    if args.limit is not None:
        items = items[:args.limit]

    # Inference loop
    for item in tqdm(items, desc="Inferring"):
        prompt = item["input"]
        reference = item["reference_response"]

        if prompt in seen_inputs:
            continue

        # Build chat prompt
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            top_k=64,
        )
        raw = tokenizer.batch_decode(outputs, truncate=True, skip_special_tokens=True)[0]

        if "model\n" in raw:
            model_response = raw.split("model\n", 1)[1].strip()
        else:
            model_response = raw.strip()

        entry = {
            "input": prompt,
            "model_response": model_response,
        }
        if reference:
            entry["reference_response"] = reference

        results.append(entry)
        seen_inputs.add(prompt)
        save_json(out_file, results)


    print(f"Done. {len(results)} total entries written to {out_file}")

if __name__ == "__main__":
    main()
