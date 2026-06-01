"""
augment_data.py -- Build an augmented training dataset from the Spider NL-to-SQL benchmark.

Spider (HuggingFace: spider) has ~7000 NL questions with gold SQL across 140+ databases.
Each example uses the same Spider schema format as our project, so we can reuse
sql_to_schema_links.extract_schema_links to get gold schema links.

Output: data/spider_aug_dataset/  (HF Dataset, same schema as train_dataset)

Run:
    python augment_data.py [--out_dir data] [--max_examples 3000]
"""

import argparse
import json
import os

from datasets import Dataset, load_dataset

from sql_to_schema_links import extract_schema_links


def spider_to_schema(example: dict) -> dict:
    """Convert a Spider HuggingFace example's schema fields to {table: {col: type}}."""
    table_names = example["db_table_names"]
    col_table_ids = example["db_column_names"]["table_id"]
    col_names     = example["db_column_names"]["column_name"]
    col_types     = example["db_column_types"]

    schema = {t: {} for t in table_names}
    for tidx, cname, ctype in zip(col_table_ids, col_names, col_types):
        if tidx == -1:          # skip synthetic '*'
            continue
        schema[table_names[tidx]][cname] = (ctype or "TEXT").upper()
    return schema


def spider_serialize(schema: dict) -> str:
    """Same compact format as format_data.py: TABLE(col1, col2, ...)"""
    return "\n".join(f"{t}({', '.join(cols)})" for t, cols in schema.items())


def build_aug_dataset(max_examples: int) -> Dataset:
    print("Loading Spider dataset from HuggingFace...")
    spider = load_dataset("spider", trust_remote_code=True)
    train_split = spider["train"]
    print(f"  Spider train split: {len(train_split)} examples")

    rows = []
    skipped = 0
    for i, ex in enumerate(train_split):
        if max_examples and len(rows) >= max_examples:
            break

        schema = spider_to_schema(ex)
        links, err = extract_schema_links(ex["query"], schema, dialect="sqlite")
        if err or not links:
            skipped += 1
            continue

        schema_text = spider_serialize(schema)
        rows.append({
            "question_id": i,
            "db_id":       ex["db_id"],
            "question":    ex["question"],
            "schema_text": schema_text,
            "schema_links_json": json.dumps(links, separators=(",", ":")),
        })

    print(f"  Built {len(rows)} augmented rows  (skipped {skipped} parse errors)")
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",      default="./data")
    ap.add_argument("--max_examples", type=int, default=3000,
                    help="Max Spider examples to include (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = build_aug_dataset(args.max_examples)
    out_path = os.path.join(args.out_dir, "spider_aug_dataset")
    ds.save_to_disk(out_path)
    print(f"Saved → {out_path}")

    # Quick sanity check
    row = ds[0]
    print("\n--- Sample row ---")
    print("DB    :", row["db_id"])
    print("Q     :", row["question"])
    print("SCHEMA:", row["schema_text"][:120], "...")
    print("LINKS :", row["schema_links_json"])


if __name__ == "__main__":
    main()
