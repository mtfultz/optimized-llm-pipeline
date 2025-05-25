#!/usr/bin/env python3
"""
Generate instruction/answer pairs from text chunks using an OpenAI chat model
(new openai-python ≥ 1.0 interface).

Example:
    python scripts/gen_pairs.py                     # 300 pairs, gpt-4o-mini
    python scripts/gen_pairs.py --pairs 500 --model gpt-4o
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import random
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------

def load_chunks(path: pathlib.Path) -> List[dict]:
    """Return a list of dicts from chunks.jsonl."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_prompt(chunk: str) -> str:
    return (
        "Create ONE factual question answerable ONLY by the text below. "
        "Return a JSON object with keys 'question' and 'answer'.\n\n"
        f'TEXT:\n"""{chunk}"""'
    )

def ask_model(client, model, chunk):
    prompt = (
        "You are a data engine. Create ONE factual question answerable ONLY "
        "with the text provided. Respond in JSON: "
        '{"question": "...", "answer": "..."}\n\n'
        f'TEXT:\n"""{chunk}"""'
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},  # ★ ensures JSON
    )
    return json.loads(resp.choices[0].message.content)

def write_jsonl(records: List[dict], path: pathlib.Path):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))
    print("Wrote", path)

# ---------------------------------------------------------------------------

def main():
    # load .env and check key ------------------------------------------------
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("❌  OPENAI_API_KEY not set (env var or .env file)")

    client = OpenAI()  # api_key and org auto-read from env

    # CLI args --------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=300,
                   help="total Q&A pairs to generate (default 300)")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI chat model to use (default gpt-4o-mini)")
    args = p.parse_args()

    # load chunks -----------------------------------------------------------
    chunks_path = pathlib.Path("data/processed/chunks.jsonl")
    if not chunks_path.exists():
        raise SystemExit("❌  chunks.jsonl not found – run scripts/chunk.py first")

    chunks = load_chunks(chunks_path)
    random.shuffle(chunks)

    pairs = []
    print(f"Generating {args.pairs} pairs with {args.model} …")
    for chunk in tqdm(chunks[: args.pairs]):
        try:
            chunk_text = chunk["text"][:2000]
            qa = ask_model(client, args.model, build_prompt(chunk["text"]))
            pairs.append({
                "instruction": qa["question"].strip(),
                "input": "",
                "output": qa["answer"].strip(),
            })
        except Exception as e:
            print("⚠️  Skipped one chunk:", e)

    if len(pairs) < 10:
        raise SystemExit("💥  Less than 10 pairs generated; aborting.")

    # split train / eval ----------------------------------------------------
    split = int(len(pairs) * 0.9)
    train, eval_ = pairs[:split], pairs[split:]

    data_dir = pathlib.Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, data_dir / "train.jsonl")
    write_jsonl(eval_,  data_dir / "eval.jsonl")

    print(f"✅  Done: {len(train)} train pairs, {len(eval_)} eval pairs")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
