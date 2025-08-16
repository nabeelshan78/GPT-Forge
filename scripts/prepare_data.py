import torch
from torch.utils.data import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
import re
import unicodedata
import pickle

# --- Configuration ---
RAW_DATA_PATH = "data/imdb_dataset.pt"
PROCESSED_DATA_PATH = "data/processed_data.pkl"
ARTIFACTS_DIR = "artifacts"
VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "vocab.pth")

UNK_IDX, PAD_IDX, EOS_IDX = 0, 1, 2
special_symbols = ['<unk>', '<pad>', '<|endoftext|>']
tokenizer = get_tokenizer("basic_english")

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


# --- Load and Split Raw Data ---
print("Loading raw data...")
train_data_raw, test_data_raw = torch.load(RAW_DATA_PATH)
print(f"Original train size: {len(train_data_raw)}, Test size: {len(test_data_raw)}")

all_data = train_data_raw + test_data_raw
print(f"Total combined data size: {len(all_data)}")

torch.manual_seed(42)
new_train_size = int(0.8 * len(all_data))
new_val_size = int(0.1 * len(all_data))
new_test_size = len(all_data) - new_train_size - new_val_size

train_data, val_data, test_data = random_split(
    all_data, [new_train_size, new_val_size, new_test_size]
)
print(f"New split sizes -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# --- Build and Save Vocabulary ---
print("\nBuilding vocabulary from the training data...")
def yield_tokens(data_iter):
    for _, data_sample in data_iter:
        cleaned_sample = clean_text(data_sample)
        yield tokenizer(cleaned_sample)

# Build vocab only on the training set
vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=special_symbols, min_freq=5)
vocab.set_default_index(UNK_IDX)
print("\nvocabulary Length:", len(vocab))

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
torch.save(vocab, VOCAB_PATH)
print(f"Vocabulary Length: {len(vocab)}")
print(f"Vocabulary saved to: {VOCAB_PATH}")

# --- Save Processed Datasets ---
print(f"\nSaving new split datasets to {PROCESSED_DATA_PATH}...")
processed_data = {
    'train': train_data,
    'val': val_data,
    'test': test_data
}
with open(PROCESSED_DATA_PATH, 'wb') as f:
    pickle.dump(processed_data, f)

print("\nData preparation complete!")