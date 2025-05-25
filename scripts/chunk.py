#!/usr/bin/env python3
import json, textwrap, uuid, pathlib, tqdm
from unstructured.partition.pdf import partition_pdf

RAW   = pathlib.Path("data/raw")
PROC  = pathlib.Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)
OUT   = PROC / "chunks.jsonl"

def chunk_text(txt, width=3000):            
    return textwrap.wrap(txt, width)

with OUT.open("w", encoding="utf-8") as f_out:
    for pdf in tqdm.tqdm(list(RAW.glob("*.pdf"))):
        elements = partition_pdf(str(pdf))
        md = "\n".join(e.text for e in elements if e.text)
        for c in chunk_text(md):
            rec = {"id": str(uuid.uuid4()), "text": c, "source": pdf.name}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("Wrote", OUT)
