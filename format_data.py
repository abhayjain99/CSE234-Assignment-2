"""
format_data.py  --  Build Hugging Face Dataset objects from train/validation JSON files.

Creates two artifacts used by train.py:
  - data/train_dataset/   (HF Dataset, arrow format)
  - data/val_dataset/     (HF Dataset, arrow format)

Each row contains the raw fields needed by the formatting_func in train.py:
  question_id, db_id, question, schema_text, schema_links_json

Run:
    python format_data.py [--schemas_dir schemas] [--train train.json]
                          [--val validation.json] [--out_dir data]
"""

import argparse
import json
import os
import re

from datasets import Dataset


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def load_schema_as_dict(db_id: str, schemas_dir: str = "./schemas") -> dict:
    """Return {table: [col, ...]} from a Spider-format schema file."""
    fname = db_id.replace(" ", "_").replace("/", "_") + ".json"
    path = os.path.join(schemas_dir, fname)
    with open(path) as f:
        s = json.load(f)
    schema: dict = {t: [] for t in s["table_names_original"]}
    for tidx, cname in s["column_names_original"]:
        if tidx == -1:          # skip synthetic '*' wildcard entry
            continue
        schema[s["table_names_original"][tidx]].append(cname)
    return schema


def serialize_schema(schema: dict) -> str:
    """
    Compact, readable schema serialization.
    Example output line:  INJURY(AIS, CASEID, REGION, ...)
    Tables with no columns still appear: EMPTY_TABLE()
    """
    lines = []
    for table, cols in schema.items():
        lines.append(f"{table}({', '.join(cols)})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders  (prompt + completion as chat message lists)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a schema linking assistant for natural-language-to-SQL. "
    "Given a database schema and a question, output ONLY a JSON object mapping "
    "table names to the list of column names referenced by the question. "
    "Tables referenced without specific columns (e.g. in COUNT(*)) must still "
    "appear with an empty list. Use the exact casing shown in the schema. "
    "Output valid JSON and nothing else."
)


def build_user_message(question: str, db_id: str, schema_text: str) -> str:
    return (
        f"Database: {db_id}\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Question: {question}"
    )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(examples: list, schemas_dir: str) -> Dataset:
    """
    Build a HF Dataset from a list of example dicts (train or validation format).
    Each row stores the raw text so that formatting_func in train.py can
    assemble the final chat messages.
    """
    rows = []
    for ex in examples:
        schema = load_schema_as_dict(ex["db_id"], schemas_dir)
        schema_text = serialize_schema(schema)
        # Gold schema links may be absent (e.g. validation_input.json)
        schema_links = ex.get("schema_links", {})
        schema_links_json = json.dumps(schema_links, separators=(",", ":"))
        rows.append(
            {
                "question_id": ex["question_id"],
                "db_id": ex["db_id"],
                "question": ex["question"],
                "schema_text": schema_text,
                "schema_links_json": schema_links_json,
            }
        )
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemas_dir", default="./schemas")
    ap.add_argument("--train",       default="./train.json")
    ap.add_argument("--val",         default="./validation.json")
    ap.add_argument("--out_dir",     default="./data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading training examples...")
    with open(args.train) as f:
        train_examples = json.load(f)
    train_ds = build_dataset(train_examples, args.schemas_dir)
    train_ds.save_to_disk(os.path.join(args.out_dir, "train_dataset"))
    print(f"  Saved {len(train_ds)} training rows → {args.out_dir}/train_dataset/")

    print("Loading validation examples...")
    with open(args.val) as f:
        val_examples = json.load(f)
    val_ds = build_dataset(val_examples, args.schemas_dir)
    val_ds.save_to_disk(os.path.join(args.out_dir, "val_dataset"))
    print(f"  Saved {len(val_ds)} validation rows → {args.out_dir}/val_dataset/")

    # Quick sanity check -- print one formatted row
    print("\n--- Sample row ---")
    row = train_ds[0]
    user_msg = build_user_message(row["question"], row["db_id"], row["schema_text"])
    # Truncate schema for display
    display = re.sub(r"\n.+", " ...", user_msg, count=3)
    print("USER :", display[:300], "...")
    print("ASST :", row["schema_links_json"])


if __name__ == "__main__":
    main()
