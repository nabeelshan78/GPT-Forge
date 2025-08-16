import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import List, Tuple, Any, Callable
from functools import partial
import pickle
import re
import unicodedata



def clean_text(text):
    """
    Completely cleans a text string.

    """
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --- Helper function (remains the same) ---
def get_training_sample(
    text: List[Any], 
    block_size: int,
    END_OF_TEXT_TOKEN: str
) -> Tuple[List[Any], List[Any]]:
    """Creates a single (input, target) training sample."""
    text_len = len(text)
    if text_len > block_size:
        max_start_idx = text_len - block_size - 1
        start_idx = torch.randint(low=0, high=max_start_idx + 1, size=(1,)).item()
        end_idx = start_idx + block_size
        src_sequence = text[start_idx:end_idx]
        tgt_sequence = text[start_idx + 1:end_idx + 1]
    else:
        src_sequence = text
        tgt_sequence = text[1:] + [END_OF_TEXT_TOKEN]
    return src_sequence, tgt_sequence


# --- Helper function (remains the same) ---
def collate_batch(
    batch: List[Tuple], 
    tokenizer: Callable, 
    vocab, 
    block_size: int, 
    device: torch.device, 
    PAD_IDX: int, 
    END_OF_TEXT_TOKEN: str
):
    """Collates a batch of text into source and target tensors."""
    src_batch, tgt_batch = [], []
    for _, text in batch:
        text = clean_text(text)
        src_seq, tgt_seq = get_training_sample(tokenizer(text), block_size, END_OF_TEXT_TOKEN)
        src_indices = vocab(src_seq)
        tgt_indices = vocab(tgt_seq)
        src_batch.append(torch.tensor(src_indices, dtype=torch.int64))
        tgt_batch.append(torch.tensor(tgt_indices, dtype=torch.int64))

    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=False)
    tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=False)
    return src_padded.to(device), tgt_padded.to(device)



def get_dataloaders(
    processed_data_path: str,
    vocab_path: str,
    tokenizer: Callable,
    block_size: int,
    batch_size: int,
    device: torch.device,
    PAD_IDX: int,
    END_OF_TEXT_TOKEN: str
):
    """
    Loads pre-processed data and creates train, validation, and test DataLoaders.
    
    Returns:
        A tuple containing (train_dataloader, val_dataloader, test_dataloader, vocab).
    """
    print("Loading pre-processed data and vocabulary...")
    with open(processed_data_path, 'rb') as f:
        processed_data = pickle.load(f)
    train_data = processed_data['train']
    val_data = processed_data['val']
    test_data = processed_data['test']
    
    vocab = torch.load(vocab_path)
    print("Vocabulary Size:", len(vocab))
    print("Loading complete.")

    collate_fn = partial(
        collate_batch,
        tokenizer=tokenizer,
        vocab=vocab,
        block_size=block_size,
        device=device,
        PAD_IDX=PAD_IDX,
        END_OF_TEXT_TOKEN=END_OF_TEXT_TOKEN
    )

    # Create the DataLoader instances
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("\n--- Inspecting a single batch from the train_dataloader ---")
    
    src_batch, tgt_batch = next(iter(train_dataloader))
    
    print(f"Shape of the source (src) batch:\t{src_batch.shape} -> [sequence_length, batch_size]")
    print(f"Shape of the target (tgt) batch:\t{tgt_batch.shape} -> [sequence_length, batch_size]")
    
    first_src_example = src_batch[:, 0] # Shape: [sequence_length]
    first_tgt_example = tgt_batch[:, 0] # Shape: [sequence_length]
    
    print(f"\n--- Inspecting the first example in the batch ---")
    print(f"Shape of a single source example:\t{first_src_example.shape}")
    
    src_text = " ".join(vocab.lookup_tokens(first_src_example.tolist()))
    tgt_text = " ".join(vocab.lookup_tokens(first_tgt_example.tolist()))
    
    print(f"\nDecoded Source (Input to Model):\n'{src_text}'")
    print(f"\nDecoded Target (What Model Predicts):\n'{tgt_text}'")
    print("-" * 60)
    
    return train_dataloader, val_dataloader, test_dataloader, vocab