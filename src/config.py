import torch
from torchtext.data.utils import get_tokenizer


# EMBED_SIZE = 256      
# VOCAB_SIZE = 39270
# NUM_HEADS = 4     
# NUM_LAYERS = 2     
# DROPOUT = 0.1      
# BLOCK_SIZE = 64
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 64



EMBED_SIZE = 512 
VOCAB_SIZE = 39270
NUM_HEADS = 8  
NUM_LAYERS = 4  
DROPOUT = 0.15
BLOCK_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


UNK_IDX, PAD_IDX, EOS_IDX = 0, 1, 2
special_symbols = ['<unk>', '<pad>', '<|endoftext|>']

END_OF_TEXT_TOKEN = '<|endoftext|>'
tokenizer = get_tokenizer("basic_english")
PROCESSED_DATA_PATH = "data/processed_data.pkl"
VOCAB_PATH = "artifacts/vocab.pth"
