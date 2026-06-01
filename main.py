"""
main.py  --  Inference script for CSE234 Project 2 schema linking.

Loads the fine-tuned LoRA adapter from ./adapter/ (base model pulled from HF Hub)
and predicts schema links for every question in the input file.

CLI contract (must work exactly as shown for grading):
    python3 main.py --input  <questions_input.json> \
                    --output <predictions.json>

Optional flags:
    --schemas_dir   path to Spider-format schema files  (default: ./schemas)
    --adapter_path  path to LoRA adapter folder         (default: ./adapter)
    --base_model    HuggingFace model ID                (default: Qwen/Qwen2.5-1.5B-Instruct)
    --batch_size    inference batch size                (default: 4)
    --max_new_tokens  max tokens to generate            (default: 512)
"""

import argparse
import json
import os
import re
import sys


# ---------------------------------------------------------------------------
# Schema helpers  (must match format_data.py exactly)
# ---------------------------------------------------------------------------

def load_schema_as_dict(db_id: str, schemas_dir: str = "./schemas") -> dict:
    """Return {table: [col, ...]} — used for post-processing / filter_to_schema."""
    fname = db_id.replace(" ", "_").replace("/", "_") + ".json"
    path  = os.path.join(schemas_dir, fname)
    with open(path) as f:
        s = json.load(f)
    schema: dict = {t: [] for t in s["table_names_original"]}
    for tidx, cname in s["column_names_original"]:
        if tidx == -1:
            continue
        schema[s["table_names_original"][tidx]].append(cname)
    return schema


def load_schema_with_keys(db_id: str, schemas_dir: str = "./schemas") -> dict:
    """Return {table: {cols:[col,...], pk:set, fk:{col:ref_table}}}."""
    fname = db_id.replace(" ", "_").replace("/", "_") + ".json"
    path  = os.path.join(schemas_dir, fname)
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
    """Keyword-match tables/columns to question, close over FK neighbours."""
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
    """Compact schema for all tables."""
    sk = load_schema_with_keys(db_id, schemas_dir)
    return serialize_compact(sk)


# ---------------------------------------------------------------------------
# Post-processing: filter hallucinated identifiers
# ---------------------------------------------------------------------------

def filter_to_schema(raw_links: dict, schema: dict) -> dict:
    """
    Drop any table or column not present in the actual schema (case-insensitive).
    Also ensures tables with no valid columns keep an empty list rather than
    being dropped.
    """
    # Build lowercase lookup maps
    lc_table_map = {t.lower(): t for t in schema}
    lc_col_map   = {t: {c.lower(): c for c in cols} for t, cols in schema.items()}

    cleaned = {}
    for raw_t, raw_cols in raw_links.items():
        canonical_t = lc_table_map.get(str(raw_t).lower())
        if canonical_t is None:
            continue                    # hallucinated table — drop
        valid_cols = []
        if isinstance(raw_cols, list):
            for raw_c in raw_cols:
                canonical_c = lc_col_map.get(canonical_t, {}).get(str(raw_c).lower())
                if canonical_c is not None:
                    valid_cols.append(canonical_c)
        cleaned[canonical_t] = valid_cols
    return cleaned


# ---------------------------------------------------------------------------
# JSON extraction: tolerant parsing of model output
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """
    Try to parse a JSON object from the model output.
    Falls back progressively:
      1. Direct json.loads
      2. Extract first {...} block
      3. Return {} on complete failure
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Find first {...} block (handles extra preamble/postamble)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 3. Complete failure — return empty (will score 0 for this question)
    print(f"  [WARN] Could not parse JSON from output: {text[:120]!r}", file=sys.stderr)
    return {}


# ---------------------------------------------------------------------------
# Prompt construction (matches format_data.py / train.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Schema linker: given a question and DB schema, output JSON {table:[columns]}. "
    "Include tables used without specific columns as {table:[]}. "
    "Use exact schema casing. Output JSON only."
)


def build_prompt(question: str, db_id: str, schema_text: str,
                 tokenizer) -> str:
    """Apply the model's chat template to build the full inference prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Q: {question}\n\n"
                f"DB: {db_id}\n"
                f"Schema:\n{schema_text}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def run_inference(items: list, schemas_dir: str, base_model: str,
                  adapter_path: str, batch_size: int, max_new_tokens: int) -> list:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Auto-detect base model from adapter config if not explicitly overridden
    adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            _acfg = json.load(f)
        detected = _acfg.get("base_model_name_or_path")
        if detected and detected != base_model:
            print(f"  [INFO] Overriding base_model to match adapter: {detected}")
            base_model = detected

    # -- Load model + adapter --
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"     # important for batch generation

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    preds = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        prompts, schemas = [], []

        for item in batch:
            schema      = load_schema_as_dict(item["db_id"], schemas_dir)
            schema_text = build_schema_text(item["db_id"], schemas_dir)
            prompt      = build_prompt(item["question"], item["db_id"], schema_text, tokenizer)
            prompts.append(prompt)
            schemas.append(schema)

        # Tokenize
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(device)

        # Generate
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=None,   # suppress deprecation warning when do_sample=False
                top_p=None,
                top_k=None,
            )

        # Decode only the newly generated tokens
        input_len = enc["input_ids"].shape[1]
        for j, item in enumerate(batch):
            new_ids  = out_ids[j, input_len:]
            raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            raw_links = extract_json(raw_text)
            clean_links = filter_to_schema(raw_links, schemas[j])
            preds.append({"question_id": item["question_id"], "schema_links": clean_links})

        done = min(i + batch_size, len(items))
        print(f"  [{done}/{len(items)}] processed", end="\r")

    print()
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",          required=True)
    ap.add_argument("--output",         required=True)
    ap.add_argument("--schemas_dir",    default="./schemas")
    ap.add_argument("--adapter_path",   default="./adapter")
    ap.add_argument("--base_model",     default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch_size",     type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    with open(args.input) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} questions from {args.input}")

    preds = run_inference(
        items,
        schemas_dir    = args.schemas_dir,
        base_model     = args.base_model,
        adapter_path   = args.adapter_path,
        batch_size     = args.batch_size,
        max_new_tokens = args.max_new_tokens,
    )

    with open(args.output, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions → {args.output}")


if __name__ == "__main__":
    main()
