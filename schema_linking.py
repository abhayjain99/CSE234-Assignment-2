import json
import os
import re
from collections import Counter, defaultdict


SYSTEM_PROMPT = (
    "You are a schema-linking model. Return only one valid JSON object. "
    "The object maps referenced table names to arrays of referenced column "
    "names. Use exactly the table and column casing from the schema. Include "
    "a referenced table even when no specific column is referenced, using an "
    "empty array. Do not explain."
)

STOP_WORDS = set(
    """
    the a an and or of to for in on by with where is are was were be been being
    that this those these show list give get what which who how many count total
    number average avg maximum max minimum min each all any per from as their its
    it at into currently located lookup code value values indicated should ignore
    ignored require requires requiring documents document records rows
    """.split()
)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def schema_filename(db_id):
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


class SchemaCatalog:
    def __init__(self, schemas_dir="./schemas"):
        self.schemas_dir = schemas_dir
        self._schemas = {}
        self._raw = {}
        self._load_all()

    def _load_all(self):
        for name in os.listdir(self.schemas_dir):
            if not name.endswith(".json") or name == "_index.json":
                continue
            path = os.path.join(self.schemas_dir, name)
            with open(path) as f:
                raw = json.load(f)
            db_id = raw["db_id"]
            tables = raw["table_names_original"]
            cols = {t: [] for t in tables}
            col_types = {t: {} for t in tables}
            types = raw.get("column_types", ["TEXT"] * len(raw["column_names_original"]))
            for (tidx, cname), ctype in zip(raw["column_names_original"], types):
                if tidx == -1:
                    continue
                table = tables[tidx]
                cols[table].append(cname)
                col_types[table][cname] = ctype or "TEXT"
            self._schemas[db_id] = {"tables": tables, "columns": cols, "types": col_types}
            self._raw[db_id] = raw

    def db_ids(self):
        return sorted(self._schemas)

    def schema(self, db_id):
        if db_id not in self._schemas:
            path = os.path.join(self.schemas_dir, schema_filename(db_id))
            with open(path) as f:
                raw = json.load(f)
            tables = raw["table_names_original"]
            cols = {t: [] for t in tables}
            col_types = {t: {} for t in tables}
            types = raw.get("column_types", ["TEXT"] * len(raw["column_names_original"]))
            for (tidx, cname), ctype in zip(raw["column_names_original"], types):
                if tidx == -1:
                    continue
                table = tables[tidx]
                cols[table].append(cname)
                col_types[table][cname] = ctype or "TEXT"
            self._schemas[db_id] = {"tables": tables, "columns": cols, "types": col_types}
            self._raw[db_id] = raw
        return self._schemas[db_id]

    def serialize(
        self,
        db_id,
        include_types=True,
        include_keys=True,
        question=None,
        max_tables=18,
        force_tables=None,
    ):
        schema = self.schema(db_id)
        raw = self._raw.get(db_id, {})
        selected_tables = self._select_tables(schema, question, max_tables, force_tables)
        selected_set = set(selected_tables)
        primary_by_table = defaultdict(set)
        for idx in raw.get("primary_keys", []):
            if idx < len(raw.get("column_names_original", [])):
                tidx, cname = raw["column_names_original"][idx]
                if tidx != -1:
                    primary_by_table[raw["table_names_original"][tidx]].add(cname)

        fk_lines = []
        if include_keys:
            col_names = raw.get("column_names_original", [])
            tables = raw.get("table_names_original", [])
            for left, right in raw.get("foreign_keys", []):
                if left < len(col_names) and right < len(col_names):
                    lt, lc = col_names[left]
                    rt, rc = col_names[right]
                    if lt != -1 and rt != -1 and tables[lt] in selected_set and tables[rt] in selected_set:
                        fk_lines.append(f"{tables[lt]}.{lc} -> {tables[rt]}.{rc}")

        lines = []
        if len(selected_tables) < len(schema["tables"]):
            lines.append("Candidate tables are filtered from the full schema for prompt length.")
            lines.append("All table names: " + ", ".join(schema["tables"]))
        for table in selected_tables:
            rendered_cols = []
            for col in schema["columns"][table]:
                bits = [col]
                if include_types:
                    bits.append(f":{schema['types'][table].get(col, 'TEXT')}")
                if col in primary_by_table[table]:
                    bits.append(":PK")
                rendered_cols.append("".join(bits))
            lines.append(f"{table}({', '.join(rendered_cols)})")
        if fk_lines:
            lines.append("Foreign keys: " + "; ".join(fk_lines[:60]))
        return "\n".join(lines)

    def _select_tables(self, schema, question, max_tables, force_tables=None):
        tables = list(schema["tables"])
        force = [table for table in (force_tables or []) if table in schema["columns"]]
        if not question or len(tables) <= max_tables:
            out = []
            for table in force + tables:
                if table not in out:
                    out.append(table)
            return out
        toks = question_tokens(question)
        qset = set(toks)
        qphrase = " ".join(toks)
        ranked = []
        for pos, table in enumerate(tables):
            score = 0.0
            table_toks = split_identifier(table)
            if table_toks:
                score += 1.5 * sum(tok in qset for tok in table_toks) / len(table_toks)
                if " ".join(table_toks) in qphrase:
                    score += 2.0
            for col in schema["columns"][table]:
                col_toks = split_identifier(col)
                if not col_toks:
                    continue
                overlap = sum(tok in qset for tok in col_toks)
                if overlap:
                    score += 0.8 * overlap / len(col_toks)
                if " ".join(col_toks) in qphrase:
                    score += 1.2
                if col.lower() in question.lower():
                    score += 1.2
            if table in force:
                score += 100.0
            ranked.append((score, -pos, table))
        ranked.sort(reverse=True)
        selected = []
        for score, _neg_pos, table in ranked:
            if table in force or score > 0 or len(selected) < min(6, max_tables):
                if table not in selected:
                    selected.append(table)
            if len(selected) >= max_tables:
                break
        for table in force:
            if table not in selected:
                selected.append(table)
        order = {table: idx for idx, table in enumerate(tables)}
        return sorted(selected, key=lambda table: order.get(table, 10**9))

    def canonical_links(self, db_id, links):
        schema = self.schema(db_id)
        table_map = {t.lower(): t for t in schema["tables"]}
        col_maps = {
            t: {c.lower(): c for c in schema["columns"][t]}
            for t in schema["tables"]
        }
        if not isinstance(links, dict):
            return {}
        out = {}
        for raw_table, raw_cols in links.items():
            table = table_map.get(str(raw_table).lower())
            if table is None:
                continue
            out.setdefault(table, [])
            if not isinstance(raw_cols, list):
                continue
            seen = set(out[table])
            for raw_col in raw_cols:
                col = col_maps[table].get(str(raw_col).lower())
                if col is not None and col not in seen:
                    out[table].append(col)
                    seen.add(col)
        return {
            table: [col for col in schema["columns"][table] if col in set(cols)]
            for table, cols in out.items()
        }


def compact_json(obj):
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def build_messages(question, db_id, catalog, force_tables=None):
    user = (
        f"Database id: {db_id}\n\n"
        f"Schema:\n{catalog.serialize(db_id, question=question, force_tables=force_tables)}\n\n"
        f"Question:\n{question}\n\n"
        "Return schema_links JSON only."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_json_object(text):
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(cleaned[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break
    try:
        return json.loads(cleaned[start:])
    except json.JSONDecodeError:
        return {}


def split_identifier(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return [
        tok.lower()
        for tok in text.split()
        if tok and tok.lower() not in {"tbl", "dbo"}
    ]


def question_tokens(question):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", question)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).lower()
    return [tok for tok in text.split() if tok not in STOP_WORDS and len(tok) > 1]


def example_labels(example):
    labels = []
    for table, cols in example.get("schema_links", {}).items():
        labels.append("T::" + table)
        for col in cols:
            labels.append("C::" + table + "::" + col)
    return labels


class HybridBaselineLinker:
    """Small local fallback. The final submission should use an adapter/checkpoint."""

    def __init__(self, catalog, train_path="train.json"):
        self.catalog = catalog
        self.by_db = defaultdict(list)
        if os.path.exists(train_path):
            for row in load_json(train_path):
                self.by_db[row["db_id"]].append(row)

    def _lexical_scores(self, question, db_id):
        schema = self.catalog.schema(db_id)
        q_lower = question.lower()
        toks = question_tokens(question)
        qset = set(toks)
        qphrase = " ".join(toks)
        scores = Counter()
        for table in schema["tables"]:
            table_toks = split_identifier(table)
            overlap = sum(1 for tok in table_toks if tok in qset)
            if overlap:
                scores["T::" + table] += 0.5 * overlap / max(1, len(table_toks))
            if table_toks and " ".join(table_toks) in qphrase:
                scores["T::" + table] += 1.2
            for col in schema["columns"][table]:
                col_toks = split_identifier(col)
                if not col_toks:
                    continue
                col_overlap = sum(1 for tok in col_toks if tok in qset)
                if col_overlap:
                    scores["C::" + table + "::" + col] += 0.9 * col_overlap / len(col_toks)
                col_phrase = " ".join(col_toks)
                if col_phrase and col_phrase in qphrase:
                    scores["C::" + table + "::" + col] += 1.4
                if col.lower() in q_lower:
                    scores["C::" + table + "::" + col] += 1.6
        return scores

    def _knn_scores(self, question, db_id, k=5):
        rows = self.by_db.get(db_id, [])
        if not rows:
            return Counter()

        def grams(text):
            toks = question_tokens(text)
            return set(toks) | {a + "_" + b for a, b in zip(toks, toks[1:])}

        qgrams = grams(question)
        sims = []
        for row in rows:
            rgrams = grams(row["question"])
            denom = len(qgrams | rgrams) or 1
            sims.append((len(qgrams & rgrams) / denom, row))
        sims.sort(key=lambda item: item[0], reverse=True)
        if not sims or sims[0][0] <= 0:
            return Counter()
        best = sims[0][0]
        scores = Counter()
        for sim, row in sims[:k]:
            if sim <= 0:
                continue
            weight = sim / best
            for label in example_labels(row):
                scores[label] += weight
        return scores

    def _labels_to_links(self, labels, db_id):
        schema = self.catalog.schema(db_id)
        out = {}
        for label in labels:
            parts = label.split("::")
            if len(parts) == 2 and parts[0] == "T" and parts[1] in schema["columns"]:
                out.setdefault(parts[1], [])
            elif len(parts) == 3 and parts[0] == "C":
                table, col = parts[1], parts[2]
                if table in schema["columns"] and col in schema["columns"][table]:
                    out.setdefault(table, []).append(col)
        return self.catalog.canonical_links(db_id, out)

    def predict(self, question, db_id):
        scores = Counter()
        scores.update(self._lexical_scores(question, db_id))
        for label, score in self._knn_scores(question, db_id).items():
            scores[label] += 0.55 * score

        col_items = sorted(
            [(score, label) for label, score in scores.items() if label.startswith("C::")],
            reverse=True,
        )
        table_items = sorted(
            [(score, label) for label, score in scores.items() if label.startswith("T::")],
            reverse=True,
        )
        labels = []
        best_col = col_items[0][0] if col_items else 0.0
        best_table = table_items[0][0] if table_items else 0.0
        for score, label in col_items[:8]:
            if score >= 1.15 or (best_col and score >= best_col * 0.58):
                labels.append(label)
        for label in list(labels):
            labels.append("T::" + label.split("::")[1])
        for score, label in table_items[:4]:
            if score >= 1.0 or (best_table and score >= best_table * 0.62):
                labels.append(label)
        if not labels and table_items:
            labels.append(table_items[0][1])
        return self._labels_to_links(labels, db_id)
