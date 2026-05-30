import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/sft/train.jsonl")
    parser.add_argument("--eval_file", default="data/sft/validation.jsonl")
    parser.add_argument("--experiment_name", default="cse234-p2-schema-linking")
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset
    from rapidfireai import Experiment
    from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.eval_file},
    )

    def formatting_function(row):
        return {
            "prompt": row["messages"][:-1],
            "completion": [row["messages"][-1]],
        }

    peft_configs = List([
        RFLoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj"],
            bias="none",
        ),
        RFLoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        ),
    ])
    config_set = List([
        RFModelConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            peft_config=peft_configs,
            training_args=RFSFTConfig(
                learning_rate=List([2e-4, 5e-4]),
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                num_train_epochs=6,
                max_length=1536,
                bf16=True,
                logging_steps=5,
                eval_strategy="epoch",
                save_strategy="epoch",
            ),
            model_type="causal_lm",
            model_kwargs={"device_map": "auto", "torch_dtype": "auto", "use_cache": False},
            formatting_func=formatting_function,
        ),
        RFModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            peft_config=peft_configs,
            training_args=RFSFTConfig(
                learning_rate=List([1e-4, 2e-4]),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                num_train_epochs=5,
                max_length=2048,
                bf16=True,
                logging_steps=5,
                eval_strategy="epoch",
                save_strategy="epoch",
            ),
            model_type="causal_lm",
            model_kwargs={"device_map": "auto", "torch_dtype": "auto", "use_cache": False},
            formatting_func=formatting_function,
        ),
    ])

    def create_model(model_config):
        tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name"],
            **model_config["model_kwargs"],
            trust_remote_code=True,
        )
        return model, tokenizer

    experiment = Experiment(experiment_name=args.experiment_name)
    config_group = RFGridSearch(configs=config_set, trainer_type="SFT")
    experiment.run_fit(
        config_group,
        create_model,
        dataset["train"],
        dataset["validation"],
        num_chunks=args.num_chunks,
        seed=args.seed,
        num_gpus=1,
    )
    print(experiment.get_runs_info())
    experiment.end()


if __name__ == "__main__":
    main()
