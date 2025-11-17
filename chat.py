#!/usr/bin/env python
"""
chat.py

Minimal ChatGPT-style terminal chatbot using a pre-trained Hugging Face model.
Relies on configuration values defined in config.py.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    BASE_MODEL_NAME,
    DEVICE,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
)

def load_model_and_tokenizer(model_name: str, device: str):
    """
    Load a pre-trained causal language model and tokenizer from Hugging Face.
    """
    print(f"Loading model '{model_name}' on device '{device}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Some models (like DialoGPT) don't have a pad token defined.
    # For generation, it's common to use eos_token as pad_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


def generate_reply(
    tokenizer,
    model,
    user_input: str,
    device: str,
    chat_history_ids=None,
):
    """
    Generate a reply from the model given the user input and previous chat history.
    Uses an incremental dialogue scheme (as in Hugging Face DialoGPT examples).
    """
    # Encode the new user input, append EOS token
    new_user_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt",
    ).to(device)

    # Concatenate new user input with chat history (if it exists)
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Generate response
    output_ids = model.generate(
        bot_input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )

    # The newly generated tokens are the ones after the input length
    generated_tokens = output_ids[:, bot_input_ids.shape[-1]:]
    reply = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
    )

    # Return both the reply and the full updated history
    return reply, output_ids


def main():
    tokenizer, model = load_model_and_tokenizer(BASE_MODEL_NAME, DEVICE)

    print("=== Simple Chatbot ===")
    print("Type 'exit' or 'quit' to stop.\n")

    chat_history_ids = None

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
            continue  # ignore empty lines

        # Generate model reply
        reply, chat_history_ids = generate_reply(
            tokenizer=tokenizer,
            model=model,
            user_input=user_input,
            device=DEVICE,
            chat_history_ids=chat_history_ids,
        )

        print(f"Bot: {reply}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
