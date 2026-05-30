import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from schema_linking import SchemaCatalog, build_messages, compact_json, load_json


def make_record(row, catalog):
    return {
        "question_id": row["question_id"],
        "db_id": row["db_id"],
        "messages": build_messages(
            row["question"],
            row["db_id"],
            catalog,
            force_tables=row["schema_links"].keys(),
        )
        + [{"role": "assistant", "content": compact_json(row["schema_links"])}],
    }


def balanced_rows(rows, min_per_db):
    if min_per_db <= 0:
        return rows
    by_db = {}
    for row in rows:
        by_db.setdefault(row["db_id"], []).append(row)
    out = []
    for db_id, group in sorted(by_db.items()):
        out.extend(group)
        idx = 0
        while len(group) + idx < min_per_db:
            out.append(dict(group[idx % len(group)]))
            idx += 1
    return out


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.json")
    parser.add_argument("--validation", default="validation.json")
    parser.add_argument("--schemas_dir", default="schemas")
    parser.add_argument("--out_dir", default="data/sft")
    parser.add_argument("--min_train_per_db", type=int, default=18)
    args = parser.parse_args()

    catalog = SchemaCatalog(args.schemas_dir)
    train_rows = balanced_rows(load_json(args.train), args.min_train_per_db)
    valid_rows = load_json(args.validation)
    train_records = [make_record(row, catalog) for row in train_rows]
    valid_records = [make_record(row, catalog) for row in valid_rows]
    write_jsonl(train_records, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(valid_records, os.path.join(args.out_dir, "validation.jsonl"))
    print(f"Wrote {len(train_records)} train records and {len(valid_records)} validation records to {args.out_dir}")


if __name__ == "__main__":
    main()
