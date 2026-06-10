# /// script
# dependencies = [
#   "torch",
#   "matplotlib",
#   "numpy",
# ]
# ///

"""Single-layer linear-attention teacher/student prototype.

Generate x,y from a fixed teacher linear-attention layer and train student
layers to recover the teacher's function.  The teacher and students are
initialized with random orthogonal Q/K/V matrices, and several update rules
make explicit use of that orthogonal structure.
"""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def random_orthogonal_matrix(dim: int) -> torch.Tensor:
    """Sample a Haar-ish orthogonal matrix with deterministic QR sign cleanup."""
    q, r = torch.linalg.qr(torch.randn(dim, dim))
    return q * torch.diagonal(r).sign()


def random_orthogonal_batch(batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
    """Generate row-orthogonal sequences with shape (batch, seq, dim)."""
    if seq_len > dim:
        raise ValueError("seq_len must be <= dim for row-orthogonal inputs")
    return torch.stack([random_orthogonal_matrix(dim)[:seq_len] for _ in range(batch_size)])


class SingleLayerLinearAttention(nn.Module):
    """Causal polynomial linear attention with orthogonal Q/K/V matrices."""

    def __init__(self, dim: int):
        super().__init__()
        self.W_q = nn.Parameter(random_orthogonal_matrix(dim))
        self.W_k = nn.Parameter(random_orthogonal_matrix(dim))
        self.W_v = nn.Parameter(random_orthogonal_matrix(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, w_k, w_v = self.orthogonal_matrices()
        q = x @ w_q
        k = x @ w_k
        v = x @ w_v
        kv_prefix = torch.cumsum(torch.einsum("btd,bte->btde", k, v), dim=1)
        return torch.einsum("btd,btde->bte", q, kv_prefix)

    def orthogonal_matrices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.W_q, self.W_k, self.W_v


class SkewExpLinearAttention(nn.Module):
    """Orthogonal layer with W = W_initial exp(A - A.T)."""

    def __init__(self, initial_state: dict[str, torch.Tensor]):
        super().__init__()
        dim = initial_state["W_q"].shape[0]
        for name in ("W_q", "W_k", "W_v"):
            self.register_buffer(f"base_{name}", initial_state[name].clone())
        self.A_q = nn.Parameter(torch.zeros(dim, dim))
        self.A_k = nn.Parameter(torch.zeros(dim, dim))
        self.A_v = nn.Parameter(torch.zeros(dim, dim))

    @staticmethod
    def skew(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * (matrix - matrix.T)

    def orthogonal_matrices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_q = self.base_W_q @ torch.matrix_exp(self.skew(self.A_q))
        w_k = self.base_W_k @ torch.matrix_exp(self.skew(self.A_k))
        w_v = self.base_W_v @ torch.matrix_exp(self.skew(self.A_v))
        return w_q, w_k, w_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, w_k, w_v = self.orthogonal_matrices()
        q = x @ w_q
        k = x @ w_k
        v = x @ w_v
        kv_prefix = torch.cumsum(torch.einsum("btd,bte->btde", k, v), dim=1)
        return torch.einsum("btd,btde->bte", q, kv_prefix)


class GaugeFixedLinearAttention(nn.Module):
    """Quotient model using the identifiable A = W_q W_k.T and V = W_v."""

    def __init__(self, initial_state: dict[str, torch.Tensor]):
        super().__init__()
        with torch.no_grad():
            score_matrix = initial_state["W_q"] @ initial_state["W_k"].T
            value_matrix = initial_state["W_v"]
        self.A = nn.Parameter(score_matrix.clone())
        self.V = nn.Parameter(value_matrix.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x @ self.A
        v = x @ self.V
        kv_prefix = torch.cumsum(torch.einsum("btd,bte->btde", x, v), dim=1)
        return torch.einsum("btd,btde->bte", q, kv_prefix)

    def orthogonal_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.A, self.V


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    color: str
    lr: float
    kind: str
    orthogonal_penalty: float = 0.0


@dataclass(frozen=True)
class RunConfig:
    dim: int = 32
    seq_len: int = 8
    batch_size: int = 16
    steps: int = 500
    seed: int = 42
    eval_batches: int = 8
    eval_every: int = 10
    target_reduction: float = 100.0
    adam_lr: float = 3e-3
    projected_lr: float = 3e-2
    penalty_lr: float = 5e-3
    orthogonal_penalty: float = 5e-2
    skew_lr: float = 3e-2
    expmap_lr: float = 200.0
    methods: tuple[str, ...] = (
        "adam",
        "orthogonal_penalty_adam",
        "penalty_then_project",
        "projected_adam",
        "tangent_projected_adam",
        "relative_projected_adam",
        "relative_tangent_projected_adam",
        "relative_penalty_then_project",
        "skew_exp_adam",
        "expmap_sgd",
    )
    plot_path: Path = REPORTS_DIR / "single_layer_linattention_loss.png"


@dataclass
class MethodHistory:
    train_loss: list[float]
    train_steps: list[int]
    eval_loss: list[float]
    eval_steps: list[int]
    stop_step: int | None = None


@dataclass
class MethodRun:
    spec: MethodSpec
    model: SingleLayerLinearAttention
    history: MethodHistory
    optimizer: torch.optim.Optimizer | None = None


@dataclass
class ExperimentResult:
    methods: list[MethodRun]
    target_eval_loss: float
    initial_eval_loss: float


PENALTY_KINDS = {
    "orthogonal_penalty_adam",
    "penalty_then_project",
    "relative_penalty_then_project",
}

POLAR_PROJECTED_KINDS = {
    "projected_adam",
    "tangent_projected_adam",
    "relative_projected_adam",
    "relative_tangent_projected_adam",
}

TANGENT_PROJECTED_KINDS = {
    "tangent_projected_adam",
    "relative_tangent_projected_adam",
}

POLAR_EVAL_KINDS = {
    "penalty_then_project",
    "relative_penalty_then_project",
}

GAUGE_FIXED_KINDS = {
    "relative_projected_adam",
    "relative_tangent_projected_adam",
    "relative_penalty_then_project",
}


def available_method_specs(config: RunConfig) -> dict[str, MethodSpec]:
    return {
        "adam": MethodSpec("adam", "Adam", "#2563eb", config.adam_lr, "adam"),
        "projected_adam": MethodSpec(
            "projected_adam",
            "Adam + polar projection",
            "#dc2626",
            config.projected_lr,
            "projected_adam",
        ),
        "orthogonal_penalty_adam": MethodSpec(
            "orthogonal_penalty_adam",
            "Adam + orthogonality penalty",
            "#0891b2",
            config.penalty_lr,
            "orthogonal_penalty_adam",
            config.orthogonal_penalty,
        ),
        "penalty_then_project": MethodSpec(
            "penalty_then_project",
            "Penalty Adam -> polar",
            "#0f766e",
            config.penalty_lr,
            "penalty_then_project",
            config.orthogonal_penalty,
        ),
        "tangent_projected_adam": MethodSpec(
            "tangent_projected_adam",
            "Tangent Adam + projection",
            "#16a34a",
            config.projected_lr,
            "tangent_projected_adam",
        ),
        "relative_projected_adam": MethodSpec(
            "relative_projected_adam",
            "Gauge-fixed Adam + polar",
            "#be123c",
            config.projected_lr,
            "relative_projected_adam",
        ),
        "relative_tangent_projected_adam": MethodSpec(
            "relative_tangent_projected_adam",
            "Gauge-fixed tangent Adam",
            "#65a30d",
            config.projected_lr,
            "relative_tangent_projected_adam",
        ),
        "relative_penalty_then_project": MethodSpec(
            "relative_penalty_then_project",
            "Gauge-fixed penalty -> polar",
            "#14b8a6",
            config.penalty_lr,
            "relative_penalty_then_project",
            config.orthogonal_penalty,
        ),
        "expmap_sgd": MethodSpec(
            "expmap_sgd",
            "Exp-map orthogonal SGD",
            "#9333ea",
            config.expmap_lr,
            "expmap_sgd",
        ),
        "skew_exp_adam": MethodSpec(
            "skew_exp_adam",
            "Skew-exp Adam",
            "#ea580c",
            config.skew_lr,
            "skew_exp_adam",
        ),
    }


def make_eval_set(config: RunConfig, teacher: SingleLayerLinearAttention) -> tuple[torch.Tensor, torch.Tensor]:
    eval_xs = [
        random_orthogonal_batch(config.batch_size, config.seq_len, config.dim)
        for _ in range(config.eval_batches)
    ]
    eval_x = torch.cat(eval_xs, dim=0)
    with torch.no_grad():
        eval_y = teacher(eval_x)
    return eval_x, eval_y


def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return torch.mean((model(x) - y) ** 2).item()


def polar_project_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Return the nearest orthogonal matrix in Frobenius norm."""
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def project_model_to_orthogonal_(model: SingleLayerLinearAttention) -> None:
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(polar_project_matrix(param))


def tangent_project_gradients_(model: SingleLayerLinearAttention) -> None:
    """Project Euclidean gradients onto the tangent space of O(n)."""
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None:
                continue
            gram = param.T @ param.grad
            sym_gram = 0.5 * (gram + gram.T)
            param.grad.copy_(param.grad - param @ sym_gram)


def orthogonality_error(model: SingleLayerLinearAttention) -> float:
    with torch.no_grad():
        errors = []
        for matrix in model.orthogonal_matrices():
            ident = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
            errors.append(torch.linalg.matrix_norm(matrix.T @ matrix - ident).item())
    return max(errors)


def orthogonality_penalty(model: SingleLayerLinearAttention) -> torch.Tensor:
    penalty = None
    for matrix in model.orthogonal_matrices():
        ident = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        term = torch.mean((matrix.T @ matrix - ident) ** 2)
        penalty = term if penalty is None else penalty + term
    assert penalty is not None
    return penalty


def score_and_value_matrices(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    matrices = model.orthogonal_matrices()
    if len(matrices) == 2:
        return matrices[0], matrices[1]
    w_q, w_k, w_v = matrices
    return w_q @ w_k.T, w_v


def alignment_stats(model: nn.Module, teacher: SingleLayerLinearAttention) -> tuple[float, float]:
    """Return Frobenius cosine similarities for score product and value matrix."""
    with torch.no_grad():
        score_product, value_matrix = score_and_value_matrices(model)
        teacher_score_product, teacher_value_matrix = score_and_value_matrices(teacher)
        score_cos = torch.sum(score_product * teacher_score_product) / (
            torch.linalg.matrix_norm(score_product) * torch.linalg.matrix_norm(teacher_score_product)
        )
        value_cos = torch.sum(value_matrix * teacher_value_matrix) / (
            torch.linalg.matrix_norm(value_matrix) * torch.linalg.matrix_norm(teacher_value_matrix)
        )
    return score_cos.item(), value_cos.item()


def train_adam_like(run: MethodRun, x: torch.Tensor, y: torch.Tensor) -> float:
    assert run.optimizer is not None
    run.optimizer.zero_grad(set_to_none=True)
    fit_loss = torch.mean((run.model(x) - y) ** 2)
    loss = fit_loss
    if run.spec.kind in PENALTY_KINDS:
        loss = loss + run.spec.orthogonal_penalty * orthogonality_penalty(run.model)
    loss.backward()
    if run.spec.kind in TANGENT_PROJECTED_KINDS:
        tangent_project_gradients_(run.model)
    run.optimizer.step()
    if run.spec.kind in POLAR_PROJECTED_KINDS:
        project_model_to_orthogonal_(run.model)
    return fit_loss.item()


def eval_loss_for_run(run: MethodRun, x: torch.Tensor, y: torch.Tensor) -> float:
    if run.spec.kind not in POLAR_EVAL_KINDS:
        return mse(run.model, x, y)
    projected_model = copy.deepcopy(run.model)
    project_model_to_orthogonal_(projected_model)
    return mse(projected_model, x, y)


def train_expmap_sgd(run: MethodRun, x: torch.Tensor, y: torch.Tensor) -> float:
    loss = torch.mean((run.model(x) - y) ** 2)
    params = list(run.model.parameters())
    grads = torch.autograd.grad(loss, params)
    with torch.no_grad():
        for param, grad in zip(params, grads):
            tangent_generator = 0.5 * (param.T @ grad - grad.T @ param)
            update = torch.matrix_exp(-run.spec.lr * tangent_generator)
            param.copy_(param @ update)
    return loss.item()


def train_one_step(run: MethodRun, x: torch.Tensor, y: torch.Tensor) -> float:
    if run.spec.kind == "expmap_sgd":
        return train_expmap_sgd(run, x, y)
    return train_adam_like(run, x, y)


def make_method_runs(
    config: RunConfig,
    initial_state: dict[str, torch.Tensor],
    initial_eval_loss: float,
) -> list[MethodRun]:
    specs = available_method_specs(config)
    runs = []
    for method_name in config.methods:
        spec = specs[method_name]
        if spec.kind in GAUGE_FIXED_KINDS:
            model = GaugeFixedLinearAttention(initial_state)
        elif spec.kind == "skew_exp_adam":
            model = SkewExpLinearAttention(initial_state)
        else:
            model = SingleLayerLinearAttention(config.dim)
            model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = None
        if spec.kind != "expmap_sgd":
            optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=0.0)
        runs.append(
            MethodRun(
                spec=spec,
                model=model,
                optimizer=optimizer,
                history=MethodHistory(
                    train_loss=[],
                    train_steps=[],
                    eval_loss=[initial_eval_loss],
                    eval_steps=[0],
                ),
            )
        )
    return runs


def train_students(config: RunConfig) -> ExperimentResult:
    set_seed(config.seed)
    teacher = SingleLayerLinearAttention(config.dim).requires_grad_(False)

    # Use a different seed so the student is independently orthogonal, not a
    # copy of the teacher.
    torch.manual_seed(config.seed + 1)
    initial_student = SingleLayerLinearAttention(config.dim)
    initial_state = copy.deepcopy(initial_student.state_dict())

    eval_x, eval_y = make_eval_set(config, teacher)
    initial_eval_loss = mse(initial_student, eval_x, eval_y)
    target_eval_loss = initial_eval_loss / config.target_reduction
    runs = make_method_runs(config, initial_state, initial_eval_loss)
    torch.manual_seed(config.seed + 2)

    print(
        f"target eval loss <= {target_eval_loss:.4e} "
        f"({config.target_reduction:g}x below initial {initial_eval_loss:.4e})"
    )
    print(f"dim={config.dim}, seq_len={config.seq_len}, batch_size={config.batch_size}")

    for step in range(1, config.steps + 1):
        x = random_orthogonal_batch(config.batch_size, config.seq_len, config.dim)
        with torch.no_grad():
            y = teacher(x)

        for run in runs:
            if run.history.stop_step is not None:
                continue
            train_loss = train_one_step(run, x, y)
            run.history.train_loss.append(train_loss)
            run.history.train_steps.append(step)

        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            for run in runs:
                if run.history.stop_step is not None:
                    continue
                eval_loss = eval_loss_for_run(run, eval_x, eval_y)
                run.history.eval_loss.append(eval_loss)
                run.history.eval_steps.append(step)
                if eval_loss <= target_eval_loss:
                    run.history.stop_step = step
                    if run.spec.kind in POLAR_EVAL_KINDS:
                        project_model_to_orthogonal_(run.model)

        if step == 1 or step % 100 == 0:
            losses = ", ".join(
                f"{run.spec.name}: {run.history.eval_loss[-1]:.3e}"
                for run in runs
            )
            print(f"step {step:4d} | {losses}")

        if all(run.history.stop_step is not None for run in runs):
            print(f"all methods reached target by step {step}")
            break

    for run in runs:
        status = f"hit target at step {run.history.stop_step}" if run.history.stop_step else "did not hit target"
        score_cos, value_cos = alignment_stats(run.model, teacher)
        print(
            f"{run.spec.label}: {status}; final eval={run.history.eval_loss[-1]:.4e}; "
            f"orthogonality error={orthogonality_error(run.model):.2e}; "
            f"score cos={score_cos:.3f}; value cos={value_cos:.3f}"
        )

    return ExperimentResult(
        methods=runs,
        target_eval_loss=target_eval_loss,
        initial_eval_loss=initial_eval_loss,
    )


def plot_losses(result: ExperimentResult, config: RunConfig) -> None:
    os.makedirs(config.plot_path.parent, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11.0, 5.8), dpi=150)
    for run in result.methods:
        stop_label = f"step {run.history.stop_step}" if run.history.stop_step else "not hit"
        ax.plot(
            run.history.eval_steps,
            run.history.eval_loss,
            color=run.spec.color,
            linewidth=2.1,
            label=f"{run.spec.label} ({stop_label})",
        )
    ax.axhline(
        result.target_eval_loss,
        color="#334155",
        linewidth=1.2,
        linestyle="--",
        label=f"{config.target_reduction:g}x target",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Fixed eval MSE loss")
    ax.set_title("Single-Layer Linear Attention: Orthogonal Update Prototypes")
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    ax.text(
        0.02,
        0.03,
        (
            f"dim={config.dim}, seq={config.seq_len}, batch={config.batch_size}, "
            f"Adam lr={config.adam_lr:g}, proj lr={config.projected_lr:g}, penalty={config.orthogonal_penalty:g}, "
            f"skew lr={config.skew_lr:g}, exp lr={config.expmap_lr:g}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0, 0.75, 1))
    fig.savefig(config.plot_path)
    plt.close(fig)


def parse_args() -> RunConfig:
    method_choices = tuple(available_method_specs(RunConfig()).keys())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=RunConfig.dim)
    parser.add_argument("--seq-len", type=int, default=RunConfig.seq_len)
    parser.add_argument("--batch-size", type=int, default=RunConfig.batch_size)
    parser.add_argument("--steps", type=int, default=RunConfig.steps)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--eval-batches", type=int, default=RunConfig.eval_batches)
    parser.add_argument("--eval-every", type=int, default=RunConfig.eval_every)
    parser.add_argument("--target-reduction", type=float, default=RunConfig.target_reduction)
    parser.add_argument("--adam-lr", type=float, default=RunConfig.adam_lr)
    parser.add_argument("--projected-lr", type=float, default=RunConfig.projected_lr)
    parser.add_argument("--penalty-lr", type=float, default=RunConfig.penalty_lr)
    parser.add_argument("--orthogonal-penalty", type=float, default=RunConfig.orthogonal_penalty)
    parser.add_argument("--skew-lr", type=float, default=RunConfig.skew_lr)
    parser.add_argument("--expmap-lr", type=float, default=RunConfig.expmap_lr)
    parser.add_argument("--methods", nargs="+", choices=method_choices, default=list(RunConfig.methods))
    parser.add_argument("--plot-path", type=Path, default=RunConfig.plot_path)
    args = parser.parse_args()
    return RunConfig(
        dim=args.dim,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        eval_batches=args.eval_batches,
        eval_every=args.eval_every,
        target_reduction=args.target_reduction,
        adam_lr=args.adam_lr,
        projected_lr=args.projected_lr,
        penalty_lr=args.penalty_lr,
        orthogonal_penalty=args.orthogonal_penalty,
        skew_lr=args.skew_lr,
        expmap_lr=args.expmap_lr,
        methods=tuple(args.methods),
        plot_path=args.plot_path,
    )


def main() -> None:
    config = parse_args()
    if config.seq_len > config.dim:
        raise ValueError("--seq-len must be <= --dim for orthogonal-row inputs")
    if config.eval_every < 1:
        raise ValueError("--eval-every must be >= 1")
    if config.target_reduction <= 1:
        raise ValueError("--target-reduction must be > 1")
    if (
        config.adam_lr <= 0
        or config.projected_lr <= 0
        or config.penalty_lr <= 0
        or config.skew_lr <= 0
        or config.expmap_lr <= 0
    ):
        raise ValueError("learning rates must be positive")
    if config.orthogonal_penalty < 0:
        raise ValueError("--orthogonal-penalty must be nonnegative")
    result = train_students(config)
    plot_losses(result, config)
    print(f"Saved loss plot to {config.plot_path}")
    print(f"Initial eval loss: {result.initial_eval_loss:.4e}")
    print(f"Target eval loss:  {result.target_eval_loss:.4e}")


if __name__ == "__main__":
    main()
