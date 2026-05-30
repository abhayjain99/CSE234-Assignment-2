"""
train.py  --  Baseline SFT training via RapidFire AI.

Runs a single baseline config (Qwen2.5-1.5B-Instruct, LoRA r=16) so you
can verify the end-to-end pipeline quickly before launching grid searches.

Prerequisites:
    pip install transformers peft trl datasets accelerate bitsandbytes

Run on DSMLP (after format_data.py):
    python train.py [--experiment_name baseline-v1]
                    [--model Qwen/Qwen2.5-1.5B-Instruct]
                    [--epochs 3] [--batch_size 2] [--lr 2e-4]

The trained LoRA adapter is saved to ./adapter/ for use by main.py.
"""

import argparse
import os
import shutil

import torch
from datasets import load_from_disk
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFLoraConfig, RFModelConfig, RFSFTConfig

# ---------------------------------------------------------------------------
# Precision helpers  -- called once at module load so RapidFire can pickle
# ---------------------------------------------------------------------------

_HAS_CUDA = torch.cuda.is_available()
_USE_BF16 = _HAS_CUDA and torch.cuda.is_bf16_supported()
_USE_FP16 = _HAS_CUDA and not _USE_BF16
# CPU-only: both False → fp32 (slow, only for testing)

print(f"[precision] has_cuda={_HAS_CUDA}  bf16={_USE_BF16}  fp16={_USE_FP16}")



# ---------------------------------------------------------------------------
# Formatting function  (top-level so RapidFire can pickle it)
# ---------------------------------------------------------------------------

def formatting_func(row):
    """Convert a dataset row → RapidFire prompt/completion chat message format.

    NOTE: SYSTEM_PROMPT is defined locally so the function is fully self-contained
    when RapidFire pickles and ships it to a worker subprocess.
    """
    system_prompt = (
        "You are a schema linking assistant for natural-language-to-SQL. "
        "Given a database schema and a question, output ONLY a JSON object mapping "
        "table names to the list of column names referenced by the question. "
        "Tables referenced without specific columns (e.g. in COUNT(*)) must still "
        "appear with an empty list. Use the exact casing shown in the schema. "
        "Output valid JSON and nothing else."
    )
    user_content = (
        f"Database: {row['db_id']}\n\n"
        f"Schema:\n{row['schema_text']}\n\n"
        f"Question: {row['question']}"
    )
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": row["schema_links_json"]},
        ],
    }


# ---------------------------------------------------------------------------
# Model factory  (top-level so RapidFire can pickle it)
# ---------------------------------------------------------------------------

def create_model(model_config):
    """Load base model + tokenizer; return (model, tokenizer).

    Uses standard bf16/fp16 LoRA (no QLoRA) so RapidFire's PEFT wrapper
    can apply get_peft_model without gradient issues.
    With batch_size=1 and max_seq_length=2048 the 1.5B model fits in ~20 GB.
    All imports and computations are local — module-level globals are not
    available in RapidFire's worker subprocess.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name   = model_config["model_name"]
    model_kwargs = model_config["model_kwargs"]

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        **model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_name", default="baseline-v1")
    ap.add_argument("--data_dir",        default="./data")
    ap.add_argument("--model",           default="Qwen/Qwen2.5-1.5B-Instruct")  # 1.5B > 0.5B for accuracy
    ap.add_argument("--epochs",          type=int,   default=3)
    ap.add_argument("--batch_size",      type=int,   default=1)
    ap.add_argument("--lr",              type=float, default=2e-4)
    ap.add_argument("--max_seq_length",  type=int,   default=2048)
    # 2048: safe for OOM; covers ~80% of examples. Raise to 4096 once stable.
    args = ap.parse_args()

    # -- Load datasets (built by format_data.py) --
    print("Loading datasets...")
    train_ds = load_from_disk(os.path.join(args.data_dir, "train_dataset"))
    val_ds   = load_from_disk(os.path.join(args.data_dir, "val_dataset"))
    print(f"  train: {len(train_ds)} rows | val: {len(val_ds)} rows")

    # -- Clean up any stale experiment with the same name --
    try:
        stale = Experiment(experiment_name=args.experiment_name, mode="fit")
        stale.end()
        print(f"Ended stale experiment '{args.experiment_name}' from a previous run.")
    except Exception:
        pass

    experiment = Experiment(experiment_name=args.experiment_name, mode="fit")

    # -- LoRA configs to try --
    lora_r16 = RFLoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    lora_r32 = RFLoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )

    def _sft_cfg(lr):
        return RFSFTConfig(
            learning_rate=lr,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            max_length=args.max_seq_length,
            logging_steps=10,
            eval_strategy="epoch",
            bf16=_USE_BF16,
            fp16=_USE_FP16,
            warmup_ratio=0.05,
            weight_decay=0.01,
        )

    def _model_cfg(lora_cfg, lr):
        return RFModelConfig(
            model_name=args.model,
            peft_config=List([lora_cfg]),
            training_args=_sft_cfg(lr),
            model_type="causal_lm",
            model_kwargs={"device_map": "auto", "use_cache": False},
            formatting_func=formatting_func,
        )

    # Grid: 2 LoRA ranks x 2 learning rates = 4 configs
    configs = List([
        _model_cfg(lora_r16, 2e-4),
        _model_cfg(lora_r16, 5e-5),
        _model_cfg(lora_r32, 2e-4),
        _model_cfg(lora_r32, 5e-5),
    ])

    config_group = RFGridSearch(configs=configs, trainer_type="SFT")

    print(f"\nLaunching: {args.experiment_name}")
    print(f"  model={args.model}  epochs={args.epochs}  "
          f"batch={args.batch_size}x4={args.batch_size*4}  "
          f"lr={args.lr}  max_len={args.max_seq_length}\n")

    # num_chunks=1: run each config to completion before swapping.
    # num_chunks=4 was slicing training into tiny fragments (only 19 steps/epoch).
    experiment.run_fit(config_group, create_model, train_ds, val_ds,
                       num_chunks=1, seed=42)
    experiment.end()

    # -- Copy best adapter to ./adapter/ --
    print("\nSaving best adapter → ./adapter/ ...")
    # RapidFire saves the final checkpoint to a predictable path; don't rely
    # on get_results() which returns an empty DataFrame in some versions.
    exp_dir  = os.path.join(os.path.expanduser("~/rapidfireai/rapidfire_experiments"),
                            experiment.experiment_name)
    # Walk runs/*/checkpoints/final_checkpoint (pick run with lowest eval_loss)
    best_src, best_loss = None, float("inf")
    import glob as _glob
    for run_dir in sorted(_glob.glob(os.path.join(exp_dir, "runs", "*", "checkpoints", "final_checkpoint"))):
        ts_path = os.path.join(run_dir, "trainer_state.json")
        if os.path.exists(ts_path):
            import json as _json
            ts = _json.load(open(ts_path))
            hist = ts.get("log_history", [])
            eval_losses = [e["eval_loss"] for e in hist if "eval_loss" in e and not (isinstance(e["eval_loss"], float) and e["eval_loss"] != e["eval_loss"])]
            loss = min(eval_losses) if eval_losses else float("inf")
        else:
            loss = float("inf")
        if loss <= best_loss:
            best_loss, best_src = loss, run_dir

    if best_src is None:
        # fallback: take the most recently modified final_checkpoint
        candidates = sorted(
            [p for p in __import__("glob").glob(os.path.join(exp_dir, "runs", "*", "checkpoints", "final_checkpoint")) if os.path.isdir(p)],
            key=os.path.getmtime,
        )
        best_src = candidates[-1] if candidates else None

    if best_src and os.path.isdir(best_src):
        if os.path.exists("./adapter"):
            shutil.rmtree("./adapter")
        shutil.copytree(best_src, "./adapter")
        print(f"  Copied {best_src} → ./adapter/  (eval_loss={best_loss:.4f})")
    else:
        print(f"  [WARN] No checkpoint found in {exp_dir}")
        print("  Copy manually: cp -r <exp_dir>/runs/1/checkpoints/final_checkpoint ./adapter")

    print("\nVerify with:")
    print("  python main.py --input validation_input.json --output preds.json")
    print("  python eval.py --predictions preds.json "
          "--gold validation_gold_schema_links.json "
          "--schemas_dir schemas/ --questions_input validation_input.json")


if __name__ == "__main__":
    main()
