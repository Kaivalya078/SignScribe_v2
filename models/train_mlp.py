"""
models/train_mlp.py — Train a PyTorch MLP on hand-landmark features.

Architecture:  63 → 128 (ReLU+Dropout) → 64 (ReLU+Dropout) → 29 (softmax)

Usage (from project root):
    python models/train_mlp.py
    python models/train_mlp.py --csv dataset/asl_landmarks.csv --epochs 50 --lr 0.001

Reads the same CSV produced by utils/extract_landmark.py.
Saves a checkpoint dict to checkpoints/asl_mlp_model.pth containing:
    - model_state_dict
    - classes          (ordered list of label strings)
    - input_size       (63)
    - hidden_sizes     ([128, 64])
    - num_classes      (29)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------------------
# Resolve project root so paths work when invoked from any directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# MLP Definition (must match core/inference.py)
# ---------------------------------------------------------------------------

class ASLNet(nn.Module):
    """Simple 3-layer MLP for ASL landmark classification."""

    def __init__(self, input_size: int = 63, hidden_sizes: list = None,
                 num_classes: int = 29, dropout: float = 0.3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_sizes[1], num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------

def train_mlp(
    csv_path: str,
    output_path: str,
    results_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    test_size: float = 0.2,
    seed: int = 42,
):
    # -- reproducibility --
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -- load data --
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    X = df.drop("label", axis=1).values.astype(np.float32)
    y_raw = df["label"].values

    # -- drop classes with too few samples for stratified split --
    min_samples = max(2, int(1 / test_size) + 1)  # need at least 1 sample in each split
    counts = pd.Series(y_raw).value_counts()
    valid_classes = counts[counts >= min_samples].index.tolist()
    dropped = [c for c in counts.index if c not in valid_classes]
    if dropped:
        print(f"WARNING: Dropping {len(dropped)} class(es) with < {min_samples} samples: {dropped}")
        mask = np.isin(y_raw, valid_classes)
        X = X[mask]
        y_raw = y_raw[mask]
        print(f"Remaining samples: {len(X)}")

    # -- encode labels --
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    classes = le.classes_.tolist()       # ordered list of label strings
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # -- train / test split --
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=seed, stratify=y_encoded
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # -- model / optimizer / loss --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ASLNet(input_size=X.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel architecture:\n{model}\n")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # -- training loop --
    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(train_ds)
        scheduler.step(avg_loss)

        # -- validation --
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        acc = accuracy_score(all_labels, all_preds)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"val_acc={acc:.4f}  lr={current_lr:.6f}")

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()

    print(f"\nBest validation accuracy: {best_acc:.4f}")

    # -- save checkpoint --
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "classes": classes,
        "input_size": X.shape[1],
        "hidden_sizes": [128, 64],
        "num_classes": num_classes,
    }
    torch.save(checkpoint, output_path)
    print(f"Model saved to {output_path}")

    # -- final evaluation with best model --
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    print(f"Final test accuracy: {final_acc:.4f}")

    # -- confusion matrix --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(all_labels, all_preds)
        os.makedirs(results_path, exist_ok=True)

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix — MLP (acc={final_acc:.2%})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plot_file = os.path.join(results_path, "confusion_matrix_mlp.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"Confusion matrix saved to {plot_file}")
    except ImportError:
        print("matplotlib/seaborn not available — skipping confusion matrix plot")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ASL MLP classifier")
    parser.add_argument("--csv", type=str,
                        default=str(PROJECT_ROOT / "dataset" / "asl_landmarks.csv"),
                        help="Path to landmark CSV")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "checkpoints" / "asl_mlp_model.pth"),
                        help="Path to save .pth checkpoint")
    parser.add_argument("--results", type=str,
                        default=str(PROJECT_ROOT / "results"),
                        help="Directory for evaluation plots")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_mlp(
        csv_path=args.csv,
        output_path=args.output,
        results_path=args.results,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
