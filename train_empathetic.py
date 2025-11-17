#!/usr/bin/env python
"""
train_empathetic.py

Fine-tune a DialoGPT-style conversational model (e.g., microsoft/DialoGPT-medium)
on the facebook/empathetic_dialogues dataset using causal language modeling.

The training text for each example looks like:

    [emotion: <tag>] <context> <eos> <utterance>

After training, point EMPATHETIC_MODEL_NAME in config.py to the output_dir
so chat_empathetic.py can use your fine-tuned checkpoint.
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import BASE_MODEL_NAME, EMOTION_DATASET_NAME


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune DialoGPT on EmpatheticDialogues."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=BASE_MODEL_NAME,
        help="Base model to fine-tune (default: config.BASE_MODEL_NAME).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=EMOTION_DATASET_NAME,
        help="HF dataset name (default: config.EMOTION_DATASET_NAME).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="empathetic-dialogpt-medium",
        help="Where to save the fine-tuned model and tokenizer.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Train batch size per device.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Eval batch size per device.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Run evaluation every X steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    args = parser.parse_args()
    return args


def build_tokenizer_and_model(model_name: str):
    """
    Load tokenizer and model, make sure pad_token is defined and consistent.
    """
    print(f"[Model] Loading '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # For GPT-2 / DialoGPT style models, there's usually no pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # If tokenizer vocab changed (e.g., pad token added), resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    print("[Model] Loaded.\n")
    return tokenizer, model


def prepare_datasets(
    dataset_name: str,
    tokenizer,
    max_seq_length: int,
):
    """
    Load facebook/empathetic_dialogues and tokenize it for causal LM training.

    Uses:
      - tags      : emotion label (string)
      - context   : dialogue context
      - utterance : current system utterance / response
    """

    print(f"[Data] Loading dataset '{dataset_name}'...")
    raw_datasets = load_dataset(dataset_name)
    # raw_datasets has splits: train, validation, test

    def preprocess_batch(batch):
        texts = []
        contexts = batch.get("context", [])
        utterances = batch.get("utterance", [])
        tags = batch.get("tags", [])

        for ctx, utt, tag in zip(contexts, utterances, tags):
            emotion = tag if tag else "neutral"
            ctx = ctx if ctx is not None else ""
            utt = utt if utt is not None else ""

            # Emotion-conditioned training text
            text = (
                f"[emotion: {emotion}] "
                f"{ctx} {tokenizer.eos_token} {utt}"
            )
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
        )
        return tokenized

    print("[Data] Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    print(
        f"[Data] Done. Train examples: {len(train_dataset)}, "
        f"Validation examples: {len(eval_dataset)}"
    )
    return train_dataset, eval_dataset


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model = build_tokenizer_and_model(args.model_name)

    train_dataset, eval_dataset = prepare_datasets(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
        report_to="none",  # disable wandb etc. for homework simplicity
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("[Train] Starting training...")
    trainer.train()
    print("[Train] Training finished.")

    print("[Eval] Running evaluation on validation set...")
    metrics = trainer.evaluate()
    print("[Eval] Metrics:", metrics)

    print(f"[Save] Saving model + tokenizer to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[Save] Done.")
    print(
        "\nNow set EMPATHETIC_MODEL_NAME in config.py to this folder "
        "so chat_empathetic.py loads your fine-tuned model."
    )


if __name__ == "__main__":
    main()
