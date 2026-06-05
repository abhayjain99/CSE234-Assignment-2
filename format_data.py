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

from datasets import Dataset, concatenate_datasets, load_from_disk


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def load_schema_with_keys(db_id: str, schemas_dir: str = "./schemas") -> dict:
    """Return {table: {cols:[col,...], pk:set, fk:{col:ref_table}}}."""
    fname = db_id.replace(" ", "_").replace("/", "_") + ".json"
    path = os.path.join(schemas_dir, fname)
    with open(path) as f:
        s = json.load(f)

    tables = s["table_names_original"]
    result = {t: {"cols": [], "pk": set(), "fk": {}} for t in tables}

    col_list = []
    for tidx, cname in s["column_names_original"]:
        col_list.append((tidx, cname))
        if tidx == -1:
            continue
        result[tables[tidx]]["cols"].append(cname)

    for pk_idx in s.get("primary_keys", []):
        if isinstance(pk_idx, list):
            for sub in pk_idx:
                if isinstance(sub, int) and sub < len(col_list):
                    tidx, cname = col_list[sub]
                    if tidx != -1:
                        result[tables[tidx]]["pk"].add(cname)
        elif isinstance(pk_idx, int) and pk_idx < len(col_list):
            tidx, cname = col_list[pk_idx]
            if tidx != -1:
                result[tables[tidx]]["pk"].add(cname)

    for fk_src, fk_dst in s.get("foreign_keys", []):
        if fk_src < len(col_list) and fk_dst < len(col_list):
            s_tidx, s_col = col_list[fk_src]
            d_tidx, _ = col_list[fk_dst]
            if s_tidx != -1 and d_tidx != -1:
                result[tables[s_tidx]]["fk"][s_col] = tables[d_tidx]

    return result


def serialize_compact(schema_keys: dict, tables: list = None) -> str:
    """TABLE(col*,col>RefTable,col) — no spaces, * = PK, >Table = FK."""
    if tables is None:
        tables = list(schema_keys.keys())
    lines = []
    for t in tables:
        if t not in schema_keys:
            continue
        info = schema_keys[t]
        parts = []
        for col in info["cols"]:
            marker = col
            if col in info["pk"]:
                marker += "*"
            if col in info["fk"]:
                marker += f">{info['fk'][col]}"
            parts.append(marker)
        lines.append(f"{t}({','.join(parts)})")
    return "\n".join(lines)


def prune_tables(question: str, schema_keys: dict) -> list:
    """Keyword-match tables/columns to question, then close over FK neighbours."""
    q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    relevant = set()

    for table, info in schema_keys.items():
        if set(re.findall(r"[a-z0-9]+", table.lower())) & q_words:
            relevant.add(table)
            continue
        for col in info["cols"]:
            if set(re.findall(r"[a-z0-9]+", col.lower())) & q_words:
                relevant.add(table)
                break

    if not relevant:
        return list(schema_keys.keys())

    for t in list(relevant):
        for ref_t in schema_keys[t]["fk"].values():
            if ref_t in schema_keys:
                relevant.add(ref_t)

    return [t for t in schema_keys.keys() if t in relevant]


def build_schema_text(db_id: str, schemas_dir: str) -> str:
    """Compact schema for all tables (no pruning — avoids hiding tables at inference)."""
    sk = load_schema_with_keys(db_id, schemas_dir)
    return serialize_compact(sk)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Schema linker: given a question and DB schema, output JSON {table:[columns]}.\n"
    "List ALL columns the question references: SELECT outputs, WHERE/HAVING filters, "
    "JOIN keys, GROUP BY and ORDER BY columns.\n"
    "Tables used without specific columns (e.g. COUNT(*)) get []. "
    "Use exact schema casing. Output JSON only."
)


def build_user_message(question: str, db_id: str, schema_text: str) -> str:
    return (
        f"Q: {question}\n\n"
        f"DB: {db_id}\n"
        f"Schema:\n{schema_text}"
    )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(examples: list, schemas_dir: str) -> Dataset:
    rows = []
    for ex in examples:
        schema_links = ex.get("schema_links", {})
        schema_text = build_schema_text(ex["db_id"], schemas_dir)
        rows.append({
            "question_id": ex["question_id"],
            "db_id":       ex["db_id"],
            "question":    ex["question"],
            "schema_text": schema_text,
            "schema_links_json": json.dumps(schema_links, separators=(",", ":")),
        })
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
    ap.add_argument("--aug_dir",     default=None,
                    help="Path to spider_aug_dataset/ to merge into train split")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading training examples...")
    with open(args.train) as f:
        train_examples = json.load(f)
    train_ds = build_dataset(train_examples, args.schemas_dir)

    if args.aug_dir and os.path.exists(args.aug_dir):
        aug_ds = load_from_disk(args.aug_dir)
        train_ds = concatenate_datasets([train_ds, aug_ds])
        print(f"  Merged augmented data: {len(train_ds)} total rows ({len(aug_ds)} from Spider)")

    train_ds.save_to_disk(os.path.join(args.out_dir, "train_dataset"))
    print(f"  Saved {len(train_ds)} training rows → {args.out_dir}/train_dataset/")

    print("Loading validation examples...")
    with open(args.val) as f:
        val_examples = json.load(f)
    val_ds = build_dataset(val_examples, args.schemas_dir)
    val_ds.save_to_disk(os.path.join(args.out_dir, "val_dataset"))
    print(f"  Saved {len(val_ds)} validation rows → {args.out_dir}/val_dataset/")

    print("\n--- Sample row ---")
    row = train_ds[0]
    user_msg = build_user_message(row["question"], row["db_id"], row["schema_text"])
    print("USER :", user_msg[:500])
    print("ASST :", row["schema_links_json"])


if __name__ == "__main__":
    main()
