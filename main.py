import argparse
import os
import sys

from schema_linking import (
    HybridBaselineLinker,
    SchemaCatalog,
    build_messages,
    dump_json,
    extract_json_object,
    load_json,
)


class ModelLinker:
    def __init__(
        self,
        catalog,
        base_model,
        adapter_path="./adapter",
        checkpoint_path="./checkpoint",
        max_new_tokens=256,
    ):
        self.catalog = catalog
        self.max_new_tokens = max_new_tokens
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(f"missing model dependencies: {exc}") from exc

        self.torch = torch
        model_source = checkpoint_path if os.path.isdir(checkpoint_path) else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if os.path.isdir(adapter_path) and not os.path.isdir(checkpoint_path):
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError(f"adapter exists but peft is unavailable: {exc}") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def _render_prompt(self, messages):
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def predict_batch(self, items, batch_size):
        out = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            prompts = [
                self._render_prompt(build_messages(row["question"], row["db_id"], self.catalog))
                for row in batch
            ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with self.torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            for row, text in zip(batch, decoded):
                raw_links = extract_json_object(text)
                out.append(self.catalog.canonical_links(row["db_id"], raw_links))
        return out


def choose_linker(args, catalog):
    has_artifact = os.path.isdir(args.adapter_path) or os.path.isdir(args.checkpoint_path)
    if args.mode == "baseline" or (args.mode == "auto" and not has_artifact):
        if args.mode == "auto" and not has_artifact:
            print("WARN: no adapter/ or checkpoint/ found; using local baseline", file=sys.stderr)
        return "baseline", HybridBaselineLinker(catalog, train_path=args.train_path)
    try:
        return "model", ModelLinker(
            catalog,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            checkpoint_path=args.checkpoint_path,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as exc:
        if args.mode == "model":
            raise
        print(f"WARN: model load failed ({exc}); using local baseline", file=sys.stderr)
        return "baseline", HybridBaselineLinker(catalog, train_path=args.train_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schemas_dir", default="./schemas")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", default="./adapter")
    parser.add_argument("--checkpoint_path", default="./checkpoint")
    parser.add_argument("--train_path", default="./train.json")
    parser.add_argument("--mode", choices=["auto", "model", "baseline"], default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    catalog = SchemaCatalog(args.schemas_dir)
    items = load_json(args.input)
    mode, linker = choose_linker(args, catalog)
    if mode == "model":
        predictions = linker.predict_batch(items, args.batch_size)
    else:
        predictions = [
            linker.predict(row["question"], row["db_id"])
            for row in items
        ]
    dump_json(
        [
            {"question_id": row["question_id"], "schema_links": links}
            for row, links in zip(items, predictions)
        ],
        args.output,
    )
    print(f"Wrote {len(items)} predictions to {args.output}")


if __name__ == "__main__":
    main()
