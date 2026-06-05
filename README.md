# CSE/DSC 234 Project 2 — Schema Linking with SFT

**Team:** Jay Chaudhary (`jgchaudhary@ucsd.edu`) · Abhay Jain

---

## Quick start (grading)

```bash
python3 main.py --input validation_input.json --output preds.json
```

The script loads the base model from HuggingFace Hub and the LoRA adapter from `./adapter/`.
No manual setup required.

Evaluate:
```bash
python eval.py --predictions preds.json \
               --gold validation_gold_schema_links.json \
               --schemas_dir schemas/ \
               --questions_input validation_input.json
```

---

## Model artifact

- **Type:** LoRA adapter (PEFT)
- **Location:** `adapter/` at repo root
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (loaded automatically from HuggingFace Hub)
- **Adapter size:** ~67 MB (committed to git)

`main.py` auto-detects the base model from `adapter/adapter_config.json` so no flags are needed.

---

## Dependencies

```
transformers
peft
torch
datasets
rapidfireai
trl
accelerate
```

All packages are available in the `cse234` conda environment on DSMLP.

---

## Repository structure

```
adapter/                  LoRA adapter (best checkpoint, score 0.6117)
schemas/                  17 Spider-format schema JSON files
logs/                     RapidFire AI experiment logs for all reported configs
  r64-5ep/                C6: r=64, simple format, lr=5e-5, 5 epochs
  compact-r64-5ep/        C7: compact + pruning (ablation)
  compact-noprun-r64-5ep/ C8: compact, question-first, no pruning (best)
  r128-col-5ep/           C9: r=128, column-aware prompt
format_data.py            Dataset construction (schema serialisation)
train.py                  LoRA fine-tuning via RapidFire AI
main.py                   Inference script (grading entry point)
eval.py                   Evaluation script
scores.json               Leaderboard scores per run
report.pdf                Project report
preds.json                Validation set predictions from final pipeline (C8)
```

---

## Reproducing training

```bash
python format_data.py
python train.py --experiment_name compact-noprun-r64-5ep \
                --model Qwen/Qwen2.5-1.5B-Instruct \
                --epochs 5
```

Training takes ~40 minutes on a 24 GB MIG slice.

---

## Best configuration (C8)

| Knob | Value |
|------|-------|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| LoRA r / alpha | 64 / 128 |
| LoRA targets | q\_proj, k\_proj, v\_proj, o\_proj |
| Learning rate | 5e-5, cosine, 5% warmup |
| Epochs | 5 |
| Effective batch | 4 (1 × 4 grad accum) |
| Schema format | `TABLE(col*,col>RefTable)` compact PK/FK |
| Prompt layout | Question-first |
| **Leaderboard score** | **0.6117** (Table 0.683, Column 0.541) |
