#!/usr/bin/env python3
import argparse
import os
from datasets import load_dataset

def save_jsonl(ds, path):
    ds.to_json(path, orient="records", lines=True)

def download_and_save(dataset):

    try:
        dataset = dataset.lower()
        # GSM8k
        if dataset in ("gsm8k", "all"):
            for split in ("train", "test"):
                ds = load_dataset("openai/gsm8k", "main", split=split)
                save_jsonl(ds, f"gsm8k_{split}.jsonl")

        # MATH
        if dataset in ("math", "all"):
            for split in ("train", "test"):
                ds_train = load_dataset("nlile/hendrycks-MATH-benchmark", split=split)
                save_jsonl(ds_train, f"math_{split}.jsonl")

        # HumanEval
        if dataset in ("humaneval", "all"):
            ds = load_dataset("openai/openai_humaneval", split="test")
            ds_splits = ds.train_test_split(test_size=0.05, seed=42)
            # save_jsonl(ds_splits["train"], f"humaneval_train.jsonl") ## HACK for quick testing! UNDO
            save_jsonl(ds_splits["test"], f"humaneval_train.jsonl") ## HACK for quick testing! UNDO
            save_jsonl(ds_splits["test"], f"humaneval_test.jsonl")
            save_jsonl(ds_splits["test"], f"humaneval_public_test.jsonl")

        # VeriThoughts
        if dataset in ("verithoughts", "all"):
            # The "Training" set: this is the ~10k queries: B dataset (Paper. Table 2)
            ds = load_dataset("wilyub/VeriThoughtsTrainSetInconsistentInstructionGT", split="train")
            ds_splits = ds.train_test_split(test_size=0.01, seed=42)
            save_jsonl(ds_splits["test"], f"verithoughts_train.jsonl") ## HACK for quick testing! UNDO
            save_jsonl(ds_splits["test"], f"verithoughts_test.jsonl") ## HACK for quick testing! UNDO

            # The "Benchmark" set: this is the "219 queries" benchmark (Paper. Table 7)
            ds = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")
            save_jsonl(ds, f"verithoughts_public_test.jsonl")

    except Exception as e:

        print(f"[Error] {e}")
        print("Might need to run 'huggingface-cli login' first and try again!!?!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["gsm8k", "math", "humaneval", "verithoughts", "all"], required=True,)
    args = parser.parse_args()
    download_and_save(args.dataset)