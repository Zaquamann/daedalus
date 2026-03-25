#!/usr/bin/env python3
"""Train a CNN for spoken word recognition on log mel spectrogram data."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "processed", "dataset.pt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "processed", "model.pt")

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.33
PATIENCE = 15
RANDOM_SEED = 42


# ── SpecAugment ─────────────────────────────────────────────────────────────
def spec_augment(x, freq_mask_max=10, time_mask_max=10):
    """Apply frequency and time masking to a spectrogram tensor (1, F, T)."""
    x = x.clone()
    _, n_freq, n_time = x.shape

    # Frequency mask
    f = torch.randint(0, freq_mask_max + 1, (1,)).item()
    if f > 0 and n_freq > f:
        f0 = torch.randint(0, n_freq - f, (1,)).item()
        x[:, f0:f0 + f, :] = 0.0

    # Time mask
    t = torch.randint(0, time_mask_max + 1, (1,)).item()
    if t > 0 and n_time > t:
        t0 = torch.randint(0, n_time - t, (1,)).item()
        x[:, :, t0:t0 + t] = 0.0

    return x


# ── Dataset ─────────────────────────────────────────────────────────────────
class WordDataset(Dataset):
    def __init__(self, spectrograms, labels, augment=False):
        self.X = spectrograms.unsqueeze(1)  # (N, 80, T) -> (N, 1, 80, T)
        self.y = labels
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = spec_augment(x)
        return x, self.y[idx]


# ── Model ───────────────────────────────────────────────────────────────────
class WordCNN(nn.Module):
    def __init__(self, num_classes, input_shape=(1, 80, 99)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Compute flat_size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.conv2(self.conv1(dummy))
            flat_size = dummy.numel()

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ── Stratified split ────────────────────────────────────────────────────────
def stratified_split(labels, test_size, random_state):
    """Split indices into train/val with stratification by label."""
    rng = np.random.RandomState(random_state)
    labels_np = labels.numpy()

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels_np):
        class_indices[label].append(idx)

    train_idx, val_idx = [], []
    for cls, indices in class_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * test_size))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    return np.array(train_idx), np.array(val_idx)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    data = torch.load(DATA_PATH, weights_only=False)
    spectrograms = data["spectrograms"]
    labels = data["labels"]
    label_to_idx = data["label_to_idx"]
    idx_to_label = data["idx_to_label"]
    config = data["config"]
    num_classes = len(label_to_idx)

    print(f"Loaded {len(labels)} samples, {num_classes} classes")
    print(f"Spectrogram shape: {spectrograms.shape}")

    # Stratified train/val split
    train_idx, val_idx = stratified_split(labels, TEST_SIZE, RANDOM_SEED)

    train_dataset = WordDataset(spectrograms[train_idx], labels[train_idx], augment=True)
    val_dataset = WordDataset(spectrograms[val_idx], labels[val_idx], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    input_shape = (1, spectrograms.shape[1], spectrograms.shape[2])
    model = WordCNN(num_classes, input_shape).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.1%} | {val_loss:8.4f} | {val_acc:6.1%}")

        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "config": config,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
            }, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no val loss improvement for {PATIENCE} epochs)")
                break

    print(f"\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val accuracy: {best_val_acc:.1%}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
