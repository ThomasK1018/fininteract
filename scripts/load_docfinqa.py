"""
Load Kensho/DocFinQA from HuggingFace, normalize to our constructor-input schema.

DocFinQA fields (per Kensho 2024):
  - question:       analyst question
  - context:        long document (avg ~123k words, the full 10-K)
  - answer:         numeric or short-text gold answer
  - gold_inds:      gold evidence indices into context
  - doc_id / cik:   filing identifier

Output: data/docfinqa/docfinqa_normalized.jsonl
Each line is one (question, passage, answer, source) record we can feed to the constructor.

Usage:
    pip install datasets
    python scripts/load_docfinqa.py [--split train|validation|test] [--limit N]
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


REPO = "kensho/DocFinQA"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "docfinqa"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--kensho--DocFinQA"
SPLIT_FILES = {
    "train": "train.json",
    "validation": "dev.json",
    "test": "test.json",
}


def normalize_record(rec: dict) -> dict:
    """DocFinQA schema (Kensho release): {Context, Question, Program, Answer}.
    We pull each with capitalized- and lowercase-fallbacks so the loader is
    robust across any future schema drift."""
    return {
        "id": rec.get("id") or rec.get("doc_id") or rec.get("uid"),
        "question": rec.get("Question") or rec.get("question"),
        "answer": rec.get("Answer") or rec.get("answer") or rec.get("gold_answer"),
        "context": rec.get("Context") or rec.get("context") or rec.get("text"),
        # Program is FinQA-style numeric reasoning (e.g. "subtract(100, 50)") —
        # useful later for verifying constructor-generated candidate answers.
        "program": rec.get("Program") or rec.get("program"),
        "gold_inds": rec.get("gold_inds") or rec.get("evidence") or [],
        "cik": rec.get("cik") or rec.get("CIK"),
        "filing_date": rec.get("filing_date") or rec.get("date"),
        "source": "kensho/DocFinQA",
    }


def find_cached_split(split: str) -> Path | None:
    filename = SPLIT_FILES[split]
    snapshots = HF_CACHE_DIR / "snapshots"
    if not snapshots.exists():
        return None

    for snapshot_dir in sorted(snapshots.iterdir(), reverse=True):
        candidate = snapshot_dir / filename
        if candidate.exists():
            return candidate
    return None


def iter_local_json(path: Path, limit: int | None):
    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected {path} to contain a JSON list, got {type(records).__name__}")

    for i, rec in enumerate(records):
        if limit is not None and i >= limit:
            break
        yield rec


def iter_hf_stream(split: str, limit: int | None):
    ds = load_dataset(REPO, split=split, streaming=True)
    if limit is not None:
        ds = ds.take(limit)
    yield from ds


def main(split: str, limit: int | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cached_path = find_cached_split(split)
    if cached_path:
        print(f"Loading {REPO} split={split} from cached {cached_path.name} ...")
        ds = iter_local_json(cached_path, limit)
    else:
        # Streaming mode bypasses the pyarrow block_size overflow that hits on
        # DocFinQA's 123k-word records. Avoids the prepare_split code path entirely.
        print(f"Loading {REPO} split={split} (streaming) ...")
        ds = iter_hf_stream(split, limit)

    out_path = OUT_DIR / f"docfinqa_{split}.jsonl"
    n_written = 0
    last_keys = None
    warned_missing = False
    with out_path.open("w", encoding="utf-8") as f:
        for rec in ds:
            normalized = normalize_record(rec)
            last_keys = list(normalized.keys())
            if normalized["question"] is None or normalized["context"] is None:
                if not warned_missing:
                    raw_preview = {k: f"{type(v).__name__}: {repr(v)[:120]}" for k, v in rec.items()}
                    print(f"  [warn] missing question/context. Raw keys: {list(rec.keys())}")
                    print(f"  [warn] raw value preview: {raw_preview}")
                    warned_missing = True
                continue
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records to {out_path}")
    if last_keys:
        print(f"Normalized record fields: {last_keys}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--limit", type=int, default=None, help="Cap records (useful for testing)")
    args = p.parse_args()
    main(args.split, args.limit)
