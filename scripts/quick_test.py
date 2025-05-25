#!/usr/bin/env python3
"""
quick_test.py  –  tiny CLI to sanity-check a merged / quantised Llama-3 model

Examples
--------
# test the fp16 merge on the 4090
python scripts/quick_test.py \
        -m merged-llama3 \
        -p "How is fire-barrier penetration PI3480864 evaluated?"

# test a 4-bit GPTQ export
python scripts/quick_test.py \
        -m models/llama3-8b-gptq \
        -p "Summarise NRC Generic Letter 86-10" \
        --temperature 0.2 --max_new_tokens 192
"""
import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="Path or HF repo-id of merged / quantised model")
    ap.add_argument("-p", "--prompt", required=True,
                    help="Question or instruction to test")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature",     type=float, default=0.3)
    ap.add_argument("--top_p",           type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--dtype", choices=["fp16","bf16"], default="fp16",
                    help="Precision to load the model in (default fp16)")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print(f"🔹 loading model {args.model} …")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map={"": 0},          # single-GPU (4090) test
                torch_dtype=dtype,
                trust_remote_code=True,
            )

    pipe = pipeline("text-generation", model=model, tokenizer=tok)

    system = "You are a licensed nuclear fire-protection engineer."
    full_prompt = f"{system}\nQ: {args.prompt}\nA:"
    print("🔹 generating …\n")

    out = pipe(
        full_prompt,
        max_new_tokens     = args.max_new_tokens,
        temperature        = args.temperature,
        top_p              = args.top_p,
        repetition_penalty = args.repetition_penalty,
        do_sample          = True,
    )[0]["generated_text"]

    print(out)

if __name__ == "__main__":
    main()
