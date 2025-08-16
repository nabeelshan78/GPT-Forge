import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import *


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device
):
    """
    Trains the GPT-style language model for one epoch.
    
    Returns:
        Tuple[float, float, float]: Average loss, accuracy, and perplexity for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc="Training Epoch")

    for batch_idx, (src, tgt) in enumerate(progress_bar, start=1):
        src, tgt = src.to(device), tgt.to(device)

        # --- Forward pass and Loss ---
        logits = model(src)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        tgt_flat = tgt.reshape(-1)
        loss = criterion(logits_flat, tgt_flat)

        # --- Backward pass and Optimization ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        # --- Update Metrics ---
        total_loss += loss.item()
        
        preds = torch.argmax(logits_flat, dim=1)
        non_pad_mask = (tgt_flat != PAD_IDX)
        total_correct += (preds[non_pad_mask] == tgt_flat[non_pad_mask]).sum().item()
        total_tokens += non_pad_mask.sum().item()
        
        running_loss = total_loss / batch_idx
        running_acc = total_correct / total_tokens if total_tokens > 0 else 0

        # Live update in tqdm bar
        progress_bar.set_postfix({
            "loss": f"{running_loss:.4f}",
            "acc": f"{running_acc:.4f}"
        })

    # --- Calculate final epoch metrics ---
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss)
    
    return avg_loss, accuracy, perplexity


def evaluate(
    model,
    dataloader,
    criterion,
    device
):
    """
    Evaluates the GPT-style language model.
    
    Returns:
        Tuple[float, float, float]: Average loss, accuracy, and perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for src, tgt in progress_bar:
            src, tgt = src.to(device), tgt.to(device)

            logits = model(src)
            
            # --- Loss Calculation ---
            logits_flat = logits.reshape(-1, logits.shape[-1])
            tgt_flat = tgt.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)
            total_loss += loss.item()

            # --- Accuracy Calculation (Per-Token) ---
            preds = torch.argmax(logits_flat, dim=1)
            non_pad_mask = (tgt_flat != PAD_IDX)
            total_correct += (preds[non_pad_mask] == tgt_flat[non_pad_mask]).sum().item()
            total_tokens += non_pad_mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
    
    return avg_loss, accuracy, perplexity