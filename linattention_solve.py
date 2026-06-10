# /// script
# dependencies = [
#   "torch",
#   "matplotlib",
# ]
# ///
"""Simple student/teacher example with a single-layer linear attention.

Teacher and student are both single-layer linear self-attentions whose
W_q, W_k, W_v are initialized with independent random orthogonal matrices.
The student is trained with plain gradient descent to match the teacher,
and the loss curve over time is saved to reports/single_layer_loss.png.
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)

DIM = 32
SEQ_LEN = 8
BATCH_SIZE = 32
NUM_STEPS = 600
LR = 1e-2

def generate_orthogonal_matrix(dim):
    """Generates a random orthogonal matrix using QR decomposition."""
    H = torch.randn(dim, dim)
    Q, R = torch.linalg.qr(H)
    return Q * torch.diagonal(R).sign()

def generate_orthogonal_rows_batch(batch_size, num_rows, dim):
    """Generates a batch of row-orthogonal inputs with shape (B, N, D)."""
    return torch.stack([generate_orthogonal_matrix(dim)[:num_rows] for _ in range(batch_size)])

class LinearSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_q = nn.Parameter(generate_orthogonal_matrix(dim))
        self.W_k = nn.Parameter(generate_orthogonal_matrix(dim))
        self.W_v = nn.Parameter(generate_orthogonal_matrix(dim))

    def forward(self, X):
        Q, K, V = X @ self.W_q, X @ self.W_k, X @ self.W_v
        # Pure polynomial causal mixing
        kv_state = torch.cumsum(torch.einsum("bnd,bne->bnde", K, V), dim=1)
        return torch.einsum("bnd,bnde->bne", Q, kv_state)

def train():
    teacher = LinearSelfAttention(DIM).requires_grad_(False)
    student = LinearSelfAttention(DIM)
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)

    # Fixed evaluation set
    eval_x = generate_orthogonal_rows_batch(BATCH_SIZE, SEQ_LEN, DIM)
    with torch.no_grad():
        eval_y = teacher(eval_x)

    train_losses, eval_losses = [], []
    for step in range(1, NUM_STEPS + 1):
        X = generate_orthogonal_rows_batch(BATCH_SIZE, SEQ_LEN, DIM)
        with torch.no_grad():
            Y = teacher(X)

        loss = torch.mean((student(X) - Y) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            eval_loss = torch.mean((student(eval_x) - eval_y) ** 2).item()
        train_losses.append(loss.item())
        eval_losses.append(eval_loss)

        if step % 100 == 0 or step == 1:
            print(f"Step {step:4d} | train loss: {loss.item():.2e} | eval loss: {eval_loss:.2e}")

    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.semilogy(train_losses, label="train loss", alpha=0.6)
    plt.semilogy(eval_losses, label="eval loss")
    plt.xlabel("step")
    plt.ylabel("MSE loss")
    plt.title(f"Single-layer linear attention student/teacher (dim={DIM}, lr={LR})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out_path = "reports/single_layer_loss.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved loss plot to {out_path}")

if __name__ == "__main__":
    train()
