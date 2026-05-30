import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="preds.json")
    parser.add_argument("--mode", default="auto", choices=["auto", "model", "baseline"])
    args = parser.parse_args()

    subprocess.check_call([
        sys.executable,
        "main.py",
        "--input",
        "validation_input.json",
        "--output",
        args.predictions,
        "--mode",
        args.mode,
    ])
    subprocess.check_call([
        sys.executable,
        "eval.py",
        "--predictions",
        args.predictions,
        "--gold",
        "validation_gold_schema_links.json",
        "--schemas_dir",
        "schemas/",
        "--questions_input",
        "validation_input.json",
        "--per_question_out",
        "per_question_validation.csv",
    ])


if __name__ == "__main__":
    main()
