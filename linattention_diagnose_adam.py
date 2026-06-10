# /// script
# dependencies = [
#   "torch",
#   "matplotlib",
#   "numpy",
# ]
# ///

"""Diagnose the S-shaped plain-Adam curve in linattention_solve.py.

The plain Adam run initially learns a cheap zero-predictor-like solution before
it learns the teacher.  This script makes that visible by comparing normal Adam
against an identical student trained on y=0, then decomposing

    MSE(pred, target) = E[pred^2] + E[target^2] - 2 E[pred * target].
"""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import linattention_solve as la


REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@dataclass
class AdamDiagnosticTrace:
    steps: list[int]
    true_eval_loss: list[float]
    zero_target_eval_loss: list[float]
    true_pred_energy: list[float]
    zero_pred_energy: list[float]
    target_energy: list[float]
    two_cross: list[float]
    output_cos: list[float]
    score_cos: list[float]
    value_cos: list[float]
    q_norm: list[float]
    k_norm: list[float]
    v_norm: list[float]


def matrix_norms(model: la.SingleLayerLinearAttention) -> tuple[float, float, float]:
    w_q, w_k, w_v = model.orthogonal_matrices()
    return (
        torch.linalg.matrix_norm(w_q).item(),
        torch.linalg.matrix_norm(w_k).item(),
        torch.linalg.matrix_norm(w_v).item(),
    )


def record_metrics(
    trace: AdamDiagnosticTrace,
    step: int,
    true_run: la.MethodRun,
    zero_run: la.MethodRun,
    teacher: la.SingleLayerLinearAttention,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
) -> None:
    with torch.no_grad():
        true_pred = true_run.model(eval_x)
        zero_pred = zero_run.model(eval_x)
        true_eval_loss = torch.mean((true_pred - eval_y) ** 2).item()
        zero_target_eval_loss = torch.mean((zero_pred - eval_y) ** 2).item()
        true_pred_energy = torch.mean(true_pred ** 2).item()
        zero_pred_energy = torch.mean(zero_pred ** 2).item()
        target_energy = torch.mean(eval_y ** 2).item()
        two_cross = (2.0 * torch.mean(true_pred * eval_y)).item()
        output_cos = (
            torch.sum(true_pred * eval_y)
            / (torch.linalg.vector_norm(true_pred) * torch.linalg.vector_norm(eval_y))
        ).item()
    score_cos, value_cos = la.alignment_stats(true_run.model, teacher)
    q_norm, k_norm, v_norm = matrix_norms(true_run.model)

    trace.steps.append(step)
    trace.true_eval_loss.append(true_eval_loss)
    trace.zero_target_eval_loss.append(zero_target_eval_loss)
    trace.true_pred_energy.append(true_pred_energy)
    trace.zero_pred_energy.append(zero_pred_energy)
    trace.target_energy.append(target_energy)
    trace.two_cross.append(two_cross)
    trace.output_cos.append(output_cos)
    trace.score_cos.append(score_cos)
    trace.value_cos.append(value_cos)
    trace.q_norm.append(q_norm)
    trace.k_norm.append(k_norm)
    trace.v_norm.append(v_norm)


def run_diagnostic(config: la.RunConfig) -> AdamDiagnosticTrace:
    la.set_seed(config.seed)
    teacher = la.SingleLayerLinearAttention(config.dim).requires_grad_(False)

    torch.manual_seed(config.seed + 1)
    initial_student = la.SingleLayerLinearAttention(config.dim)
    initial_state = copy.deepcopy(initial_student.state_dict())

    eval_x, eval_y = la.make_eval_set(config, teacher)
    initial_eval_loss = la.mse(initial_student, eval_x, eval_y)
    true_run = la.make_method_runs(config, initial_state, initial_eval_loss)[0]
    zero_run = la.make_method_runs(config, initial_state, initial_eval_loss)[0]
    torch.manual_seed(config.seed + 2)

    trace = AdamDiagnosticTrace(
        steps=[],
        true_eval_loss=[],
        zero_target_eval_loss=[],
        true_pred_energy=[],
        zero_pred_energy=[],
        target_energy=[],
        two_cross=[],
        output_cos=[],
        score_cos=[],
        value_cos=[],
        q_norm=[],
        k_norm=[],
        v_norm=[],
    )
    record_metrics(trace, 0, true_run, zero_run, teacher, eval_x, eval_y)

    for step in range(1, config.steps + 1):
        x = la.random_orthogonal_batch(config.batch_size, config.seq_len, config.dim)
        with torch.no_grad():
            y = teacher(x)
            y_zero = torch.zeros_like(y)
        la.train_one_step(true_run, x, y)
        la.train_one_step(zero_run, x, y_zero)
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            record_metrics(trace, step, true_run, zero_run, teacher, eval_x, eval_y)

    return trace


def first_crossing(steps: list[int], values: list[float], threshold: float) -> int | None:
    for step, value in zip(steps, values):
        if value <= threshold:
            return step
    return None


def plot_trace(trace: AdamDiagnosticTrace, config: la.RunConfig) -> None:
    os.makedirs(config.plot_path.parent, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), dpi=150)
    ax_loss, ax_decomp, ax_align, ax_norm = axes.ravel()

    ax_loss.plot(trace.steps, trace.true_eval_loss, label="Adam on teacher", linewidth=2.0)
    ax_loss.plot(
        trace.steps,
        trace.zero_target_eval_loss,
        label="Adam trained on y=0, eval vs teacher",
        linewidth=2.0,
        linestyle="--",
    )
    ax_loss.axhline(trace.target_energy[0], color="#475569", linestyle=":", label="zero-predictor loss")
    ax_loss.axhline(trace.true_eval_loss[0] / config.target_reduction, color="#334155", linestyle="--", label="100x target")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Loss: early Adam behaves like zero-target training")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("fixed eval MSE")
    ax_loss.legend(fontsize=8)

    ax_decomp.plot(trace.steps, trace.true_pred_energy, label="E[pred^2]", linewidth=2.0)
    ax_decomp.plot(trace.steps, trace.target_energy, label="E[target^2]", linewidth=1.6)
    ax_decomp.plot(trace.steps, trace.two_cross, label="2 E[pred target]", linewidth=2.0)
    ax_decomp.set_yscale("log")
    ax_decomp.set_title("MSE decomposition")
    ax_decomp.set_xlabel("step")
    ax_decomp.legend(fontsize=8)

    ax_align.plot(trace.steps, trace.output_cos, label="output cosine", linewidth=2.0)
    ax_align.plot(trace.steps, trace.score_cos, label="score-product cosine", linewidth=2.0)
    ax_align.plot(trace.steps, trace.value_cos, label="value cosine", linewidth=2.0)
    ax_align.set_title("Teacher alignment turns on after amplitude collapse")
    ax_align.set_xlabel("step")
    ax_align.set_ylim(-0.05, 1.05)
    ax_align.legend(fontsize=8)

    ax_norm.plot(trace.steps, trace.q_norm, label="||W_q||_F", linewidth=2.0)
    ax_norm.plot(trace.steps, trace.k_norm, label="||W_k||_F", linewidth=2.0)
    ax_norm.plot(trace.steps, trace.v_norm, label="||W_v||_F", linewidth=2.0)
    ax_norm.axhline(config.dim ** 0.5, color="#475569", linestyle=":", label="orthogonal norm")
    ax_norm.set_title("Plain Adam leaves the orthogonal manifold")
    ax_norm.set_xlabel("step")
    ax_norm.legend(fontsize=8)

    fig.suptitle(
        f"Plain Adam S-curve diagnostic: dim={config.dim}, batch={config.batch_size}, lr={config.adam_lr:g}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(config.plot_path)
    plt.close(fig)


def parse_args() -> la.RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=la.RunConfig.dim)
    parser.add_argument("--seq-len", type=int, default=la.RunConfig.seq_len)
    parser.add_argument("--batch-size", type=int, default=la.RunConfig.batch_size)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=la.RunConfig.seed)
    parser.add_argument("--eval-batches", type=int, default=la.RunConfig.eval_batches)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--target-reduction", type=float, default=la.RunConfig.target_reduction)
    parser.add_argument("--adam-lr", type=float, default=la.RunConfig.adam_lr)
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=REPORTS_DIR / "adam_s_curve_diagnostic.png",
    )
    args = parser.parse_args()
    return la.RunConfig(
        dim=args.dim,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        eval_batches=args.eval_batches,
        eval_every=args.eval_every,
        target_reduction=args.target_reduction,
        adam_lr=args.adam_lr,
        methods=("adam",),
        plot_path=args.plot_path,
    )


def main() -> None:
    config = parse_args()
    trace = run_diagnostic(config)
    plot_trace(trace, config)

    zero_step = first_crossing(trace.steps, trace.true_eval_loss, trace.target_energy[0])
    pred_min = min(zip(trace.true_pred_energy, trace.steps))
    print(f"initial eval loss:       {trace.true_eval_loss[0]:.6e}")
    print(f"zero-predictor loss:    {trace.target_energy[0]:.6e}")
    print(f"first below zero loss:  {zero_step}")
    print(f"minimum pred energy:    {pred_min[0]:.6e} at step {pred_min[1]}")
    print(f"final output cosine:    {trace.output_cos[-1]:.4f}")
    print(f"final score/value cos:  {trace.score_cos[-1]:.4f} / {trace.value_cos[-1]:.4f}")
    print(f"saved diagnostic plot:  {config.plot_path}")


if __name__ == "__main__":
    main()
