import torch
import logging
import sys

EOS_IDX = 2


def encode_prompt(
    prompt, 
    tokenizer,
    vocab,
    block_size,
    device
):
    """
    Encodes a string prompt into a tensor suitable for model input.

    This function handles prompt validation, tokenization, truncation of long 
    prompts, and conversion to a correctly shaped tensor on the specified device.

    Returns:
        Tensor: The encoded prompt as a tensor of shape (sequence_length, 1).
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty or contain only whitespace.")

    tokens = tokenizer(prompt)

    if len(tokens) > block_size:
        tokens = tokens[-block_size:]

    indices = vocab(tokens)
    # Shape: [seq_len] -> [seq_len, 1]
    return torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(1)




def decode_tokens(token_ids, vocab):
    id_list = token_ids.flatten().tolist()
    tokens = vocab.lookup_tokens(id_list)
    return " ".join(tokens)



@torch.no_grad()
def generate_text(
    model,
    prompt,
    tokenizer,
    vocab,
    block_size,
    max_new_tokens,
    device
):
    """
    Generates a sequence of text autoregressively using a trained model.

    Returns:
        str: The generated text, including the prompt.
    """
    model.eval()

    # Encode the initial prompt
    context = encode_prompt(
        prompt=prompt,
        tokenizer=tokenizer,
        vocab=vocab,
        block_size=block_size,
        device=device
    ) # Shape: (prompt_len, 1)

    # The autoregressive generation loop
    for _ in range(max_new_tokens):
        context_cond = context[-block_size:]
        
        # Forward pass with the conditioned context
        # logits shape: (current_seq_len, 1, vocab_size)
        logits = model(context_cond)

        # Get logits for the very last token in the sequence
        # Shape: (1, vocab_size)
        last_token_logits = logits[-1, :, :]
        
        # Greedily select the most likely next token
        # Shape: (1, 1)
        next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        
        # Check for end-of-sequence token
        if next_token.item() == EOS_IDX:
            break
            
        # Append the predicted token to the running sequence
        context = torch.cat([context, next_token], dim=0)

    # Decode the final sequence of token IDs back to a string
    generated_text = decode_tokens(context, vocab)
    
    return generated_text



def setup_logging(log_file_path):
    """Sets up the logging to output to console and a file."""
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger