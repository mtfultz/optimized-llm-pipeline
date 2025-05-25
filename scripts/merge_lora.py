from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE    = "models/llama3-8b"
ADAPTER = "out-lora"
OUTDIR  = "merged-llama3"

# 1. load base entirely on GPU-0
base = AutoModelForCausalLM.from_pretrained(
        BASE,
        device_map={"": 0},         
        torch_dtype="float16",
        trust_remote_code=True,
)

# 2. attach and merge the LoRA
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(OUTDIR)

# 3. copy tokenizer
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.save_pretrained(OUTDIR)

print("✅ merged model written to", OUTDIR)
