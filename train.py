#!/usr/bin/env python3
"""
QLoRA fine-tune Meta-Llama-3-8B (4-bit NF4) with LoRA r16.
No TRL – uses plain transformers.Trainer, so the grad_fn error disappears.
"""

import torch, os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    DataCollatorForLanguageModeling, Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = "models/llama3-8b"
DATA_FILES = {"train": "data/train.jsonl", "eval": "data/eval.jsonl"}

# ---------- tokenizer ------------------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": tok.eos_token})
tok.padding_side = "right"

# ---------- 4-bit base model ----------------------------------------------
bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)

base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map={"": 0}, trust_remote_code=True)
base.resize_token_embeddings(len(tok))
base.config.pad_token_id = tok.pad_token_id
base = prepare_model_for_kbit_training(base)

# ---------- LoRA adapter ---------------------------------------------------
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"])
model = get_peft_model(base, lora_cfg)
model.gradient_checkpointing_enable()

# ---------- dataset --------------------------------------------------------
ds = load_dataset("json", data_files=DATA_FILES)

def join(example):
    prompt = " ".join(example["instruction"]) if isinstance(example["instruction"], list) else str(example["instruction"])
    answer = " ".join(example["output"])       if isinstance(example["output"],       list) else str(example["output"])
    example["text"] = f"{prompt}\n{answer}{tok.eos_token}"
    return example

ds = ds.map(join, remove_columns=None)

# tokenise once up-front
def tok_fn(batch): return tok(batch["text"], truncation=True, max_length=2048)
ds = ds.map(tok_fn, batched=True,
            remove_columns=ds["train"].column_names)

collator = DataCollatorForLanguageModeling(tok, mlm=False)

# ---------- training args --------------------------------------------------
args = TrainingArguments(
        output_dir="out-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch 8
        num_train_epochs=2,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        report_to="none",
)

trainer = Trainer(model=model, args=args,
                  train_dataset=ds["train"],
                  eval_dataset=ds["eval"],
                  data_collator=collator)

trainer.train()
trainer.save_model("out-lora")
print("✅ finished – LoRA adapter saved to out-lora/")
