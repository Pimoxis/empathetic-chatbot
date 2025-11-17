#!/usr/bin/env python
"""
chat_empathetic.py

Emotionally adaptive chatbot:

- Uses a pretrained emotion classifier to infer the user's emotion.
- Conditions the conversational model's input on this emotion to produce
  more empathetic responses.

You can later replace the emotion classifier and/or the generator with
models fine-tuned on the EmpatheticDialogues dataset.
"""

import sys
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from config import (
    BASE_MODEL_NAME,
    EMPATHETIC_MODEL_NAME,
    DEVICE,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
)

# Pretrained emotion classifier from Hugging Face)
EMOTION_CLASSIFIER_MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"


def get_generator_model_name() -> str:
    """
    Decide which generator to use:
    - If EMPATHETIC_MODEL_NAME looks like a real checkpoint (not the placeholder),
      use it.
    - Otherwise, fall back to BASE_MODEL_NAME (e.g., microsoft/DialoGPT-medium).
    """
    if EMPATHETIC_MODEL_NAME and "your-username" not in EMPATHETIC_MODEL_NAME:
        return EMPATHETIC_MODEL_NAME
    return BASE_MODEL_NAME


def load_generator_and_tokenizer(device: str):
    """
    Load the conversational model (DialoGPT-medium or your fine-tuned version).
    """
    model_name = get_generator_model_name()
    print(f"[Generator] Loading model '{model_name}' on device '{device}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    print("[Generator] Model loaded.\n")
    return tokenizer, model


def load_emotion_classifier(device: str):
    print(
        f"[Emotion] Loading emotion classifier '{EMOTION_CLASSIFIER_MODEL_NAME}'..."
    )
    if device == "cuda":
        device_index = 0
    else:
        device_index = -1

    emotion_pipe = pipeline(
        "text-classification",
        model=EMOTION_CLASSIFIER_MODEL_NAME,
        # remove top_k, default returns a single dict per input
        device=device_index,
    )
    print("[Emotion] Emotion classifier loaded.\n")
    return emotion_pipe


def detect_emotion(emotion_pipe, text: str):
    if not text.strip():
        return "neutral", 0.0

    result = emotion_pipe(text)[0]   # -> {"label": "joy", "score": 0.92}
    label = result["label"]
    score = float(result["score"])
    return label, score



def generate_empathetic_reply(
    tokenizer,
    model,
    user_input: str,
    detected_emotion: str,
    device: str,
    chat_history_ids: Optional[torch.Tensor] = None,
):
    """
    Generate a reply from the model, conditioning on the detected emotion.

    We prepend a small natural-language prefix describing the emotion to
    gently nudge the generator toward an empathetic style.

    Example of what the model sees as input:
    "[User seems sad] I had a rough day at work. <eos>"
    """

    emotion_prefix = f"[User seems {detected_emotion.lower()}] "
    conditioned_input = emotion_prefix + user_input

    new_user_input_ids = tokenizer.encode(
        conditioned_input + tokenizer.eos_token,
        return_tensors="pt",
    ).to(device)

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    output_ids = model.generate(
        bot_input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )

    generated_tokens = output_ids[:, bot_input_ids.shape[-1]:]
    reply = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
    )

    return reply, output_ids


def main():
    print("=== Empathetic Chatbot ===")
    print("This version detects your emotion and tries to respond empathetically.")
    print("Type 'exit' or 'quit' to stop.\n")

    tokenizer, gen_model = load_generator_and_tokenizer(DEVICE)
    emotion_pipe = load_emotion_classifier(DEVICE)

    chat_history_ids: Optional[torch.Tensor] = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        if not user_input:
            continue

        emotion_label, emotion_score = detect_emotion(emotion_pipe, user_input)
        print(f"[Detected emotion] {emotion_label} (confidence={emotion_score:.2f})")

        reply, chat_history_ids = generate_empathetic_reply(
            tokenizer=tokenizer,
            model=gen_model,
            user_input=user_input,
            detected_emotion=emotion_label,
            device=DEVICE,
            chat_history_ids=chat_history_ids,
        )

        print(f"Bot: {reply}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
