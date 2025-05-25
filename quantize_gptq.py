from optimum.gptq import GPTQQuantizer
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "merged-llama3"        
tok     = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model   = AutoModelForCausalLM.from_pretrained(model_id,
                    device_map="auto", torch_dtype="auto")

quantizer = GPTQQuantizer(
    bits            = 4,
    dataset         = "wikitext2",   
    group_size      = 128,
    damp_percent    = 0.01,
)

q_model = quantizer.quantize_model(model, tokenizer=tok)
q_model.save_pretrained("models/llama3-8b-gptq")
tok.save_pretrained("models/llama3-8b-gptq")
print("✅ 4-bit GPTQ model saved")

