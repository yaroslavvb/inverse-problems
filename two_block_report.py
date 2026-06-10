# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
# ]
# ///
"""Instrumentation for the two-block exact solver report (reports/two_block_report.html).

Produces solver-specific diagnostics beyond what ortho_updates.py records:
  - half-step convergence: eval MSE after each block-V solve and each block-A solve,
    plus gauge-invariant distances to the teacher per alternation
  - batch-size ablation against the rank bound rank(Phi) <= B*N*(N+1)/2 (B >= 29 at D=32, N=8)
  - measured rank(Phi) vs B
  - determinant-mismatch robustness (doubly det-mismatched init)
  - label-noise robustness (noisy teacher outputs, clean eval)

Outputs: reports/two_block_convergence.png, two_block_batch_ablation.png,
two_block_noise.png, two_block_results.json.
"""

import json
import math
import time
import copy
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ortho_updates import (DIM, SEQ_LEN, polar, gen_batch, mse, make_problem,
                           ortho_error, det_sign, identifiable_distances)
from linattention_solve import LinearSelfAttention

RANK_BOUND_B = math.ceil(DIM * DIM / (SEQ_LEN * (SEQ_LEN + 1) / 2))  # = 29 at D=32, N=8


def build_phi(X, V):
    """Design matrix of the block-A subproblem: out = Phi @ vec(A).
    Phi[(b,n,e),(d,f)] = X[b,n,d] * C[b,n,e,f], C[b,n,e,f] = sum_{m<=n} V[b,m,e] X[b,m,f]."""
    C = torch.cumsum(V.unsqueeze(-1) * X.unsqueeze(2), dim=1)          # (B,N,E,D)
    return torch.einsum("bnd,bnef->bnedf", X, C).reshape(-1, DIM * DIM)


def solve_block_v(student, X, Y):
    """Exact orthogonal Procrustes for W_v given W_q, W_k."""
    Q, K = X @ student.W_q, X @ student.W_k
    scores = torch.tril(torch.einsum("bnd,bmd->bnm", Q, K))
    M = torch.einsum("bnm,bmd->bnd", scores, X).reshape(-1, DIM)
    student.W_v.copy_(polar(M.T @ Y.reshape(-1, DIM)))


def solve_block_a(student, X, Y):
    """Least squares for A given W_v, polar-projected to O(D); W_q <- A W_k."""
    V = X @ student.W_v
    Phi = build_phi(X, V)
    sol = torch.linalg.lstsq(Phi, Y.reshape(-1)).solution
    A = polar(sol.reshape(DIM, DIM))
    student.W_q.copy_(A @ student.W_k)


def run_two_block(teacher, init_state, eval_x, eval_y, n_alt=15, batch_size=64,
                  data_seed=777, noise_std=0.0):
    """Two-block solver with half-step instrumentation."""
    student = LinearSelfAttention(DIM)
    student.load_state_dict(copy.deepcopy(init_state))
    gen = torch.Generator().manual_seed(data_seed)
    rec = {"after_v": [], "after_a": [], "dist": [], "ortho": [], "alt_seconds": []}
    with torch.no_grad():
        rec["init_loss"] = mse(student(eval_x), eval_y).item()
        rec["init_dist"] = identifiable_distances(student, teacher)
    for _ in range(n_alt):
        t0 = time.time()
        X = gen_batch(gen, batch_size=batch_size)
        with torch.no_grad():
            Y = teacher(X)
            if noise_std > 0:
                Y = Y + noise_std * torch.randn(Y.shape, generator=gen)
            solve_block_v(student, X, Y)
            rec["after_v"].append(mse(student(eval_x), eval_y).item())
            solve_block_a(student, X, Y)
            rec["after_a"].append(mse(student(eval_x), eval_y).item())
            rec["dist"].append(identifiable_distances(student, teacher))
            rec["ortho"].append(max(ortho_error(getattr(student, n))
                                    for n in ["W_q", "W_k", "W_v"]))
        rec["alt_seconds"].append(time.time() - t0)
        if rec["after_a"][-1] <= 1e-17 and noise_std == 0:
            break
    return rec, student


def alternations_to(rec, thresh):
    for i, el in enumerate(rec["after_a"], start=1):
        if el <= thresh:
            return i
    return None


def measure_phi_rank(teacher, init_state, batch_size, data_seed=4242):
    student = LinearSelfAttention(DIM)
    student.load_state_dict(copy.deepcopy(init_state))
    gen = torch.Generator().manual_seed(data_seed)
    X = gen_batch(gen, batch_size=batch_size)
    with torch.no_grad():
        V = X @ student.W_v
        return int(torch.linalg.matrix_rank(build_phi(X, V)).item())


def det_diagnostics(init_state, teacher, student_final):
    """Det signs demonstrating component crossing. Note the solver's only init
    dependence is (W_q, W_k) in the first block-V solve: block V overwrites W_v
    without reading it, and block A overwrites W_q, so a 'det-mismatched init'
    experiment on W_v would be a no-op by construction."""
    return {
        "teacher": {n: det_sign(getattr(teacher, n)) for n in ["W_q", "W_k", "W_v"]},
        "student_init": {n: det_sign(init_state[n]) for n in ["W_q", "W_k", "W_v"]},
        "student_final": {n: det_sign(getattr(student_final, n)) for n in ["W_q", "W_k", "W_v"]},
        "det_A_teacher": det_sign(teacher.W_q @ teacher.W_k.T),
        "det_A_init": det_sign(init_state["W_q"] @ init_state["W_k"].T),
        "det_A_final": det_sign(student_final.W_q @ student_final.W_k.T),
    }


def main():
    results = {"config": {"dim": DIM, "seq_len": SEQ_LEN, "rank_bound_B": RANK_BOUND_B}}
    teacher, inits, eval_x, eval_y = make_problem(42)

    # ---- 1. half-step convergence (default config, unaligned init)
    rec, student = run_two_block(teacher, inits["unaligned"], eval_x, eval_y)
    results["main_run"] = {k: rec[k] for k in
                           ["after_v", "after_a", "dist", "ortho", "alt_seconds",
                            "init_loss", "init_dist"]}
    print(f"main run: {len(rec['after_a'])} alternations, final {rec['after_a'][-1]:.2e}, "
          f"mean alt time {np.mean(rec['alt_seconds']):.2f}s")

    # det component crossing: the unaligned run itself is the evidence —
    # record the det signs it crossed
    results["det_info"] = det_diagnostics(inits["unaligned"], teacher, student)
    print("det info:", results["det_info"])

    # extra seeds
    for seed in (43, 44):
        t2, i2, ex2, ey2 = make_problem(seed)
        r2, _ = run_two_block(t2, i2["unaligned"], ex2, ey2)
        results[f"seed_{seed}"] = {"after_a": r2["after_a"]}
        print(f"seed {seed}: {len(r2['after_a'])} alternations, final {r2['after_a'][-1]:.2e}")

    plt.figure(figsize=(7.5, 4.8))
    n = len(rec["after_a"])
    xs = np.arange(1, n + 1)
    half = np.empty(2 * n)
    half[0::2], half[1::2] = rec["after_v"], rec["after_a"]
    hx = np.empty(2 * n)
    hx[0::2], hx[1::2] = xs - 0.5, xs
    plt.semilogy(hx, half, "-", color="#888", lw=1, zorder=1)
    plt.semilogy(xs - 0.5, rec["after_v"], "o", label="after block V (Procrustes $W_v$)", zorder=2)
    plt.semilogy(xs, rec["after_a"], "s", label="after block A (lstsq+polar $A$)", zorder=2)
    plt.axhline(rec["init_loss"], color="k", ls=":", lw=1, label="init loss")
    plt.xlabel("alternation")
    plt.ylabel("eval MSE")
    plt.title("Half-step convergence (seed 42, B=64)")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/two_block_convergence.png", dpi=150)
    plt.close()

    # ---- 2. batch-size ablation vs the rank bound
    b_grid = [8, 16, 24, 28, 29, 32, 40, 48, 64, 96]
    ablation = []
    for B in b_grid:
        r, _ = run_two_block(teacher, inits["unaligned"], eval_x, eval_y,
                             n_alt=40, batch_size=B)
        rank = measure_phi_rank(teacher, inits["unaligned"], B)
        a12 = alternations_to(r, 1e-12)
        ablation.append({"B": B, "rank_phi": rank, "rank_bound": B * SEQ_LEN * (SEQ_LEN + 1) // 2,
                         "alt_to_1e-12": a12, "final_loss": r["after_a"][-1]})
        print(f"B={B:3d} rank(Phi)={rank:4d} (bound {min(B*36, DIM*DIM)}) "
              f"alt to 1e-12: {a12} final {r['after_a'][-1]:.2e}")
    results["batch_ablation"] = ablation

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    Bs = [a["B"] for a in ablation]
    alts = [a["alt_to_1e-12"] if a["alt_to_1e-12"] else np.nan for a in ablation]
    finals = [a["final_loss"] for a in ablation]
    ax1.plot(Bs, alts, "o-", color="tab:blue", label="alternations to 1e-12")
    ax1.set_xlabel("batch size B (fresh sequences per alternation)")
    ax1.set_ylabel("alternations to eval MSE ≤ 1e-12", color="tab:blue")
    ax1.axvline(DIM * DIM / 36, color="k", ls="--", lw=1,
                label=f"rank bound B = D²/36 ≈ {DIM*DIM/36:.1f}")
    ax2 = ax1.twinx()
    ax2.semilogy(Bs, finals, "s--", color="tab:red", alpha=0.7,
                 label="final eval MSE (≤40 alt., early stop below 1e-17)")
    ax2.set_ylabel("final eval MSE", color="tab:red")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    plt.title("Batch-size ablation: identifiability phase transition at the rank bound")
    plt.tight_layout()
    plt.savefig("reports/two_block_batch_ablation.png", dpi=150)
    plt.close()

    # ---- 3. label-noise robustness (noisy teacher outputs, clean eval)
    sigmas = [1e-8, 1e-6, 1e-4, 1e-2]
    noise = []
    for s in sigmas:
        r, _ = run_two_block(teacher, inits["unaligned"], eval_x, eval_y,
                             n_alt=20, batch_size=64, noise_std=s)
        noise.append({"sigma": s, "final_loss": r["after_a"][-1],
                      "min_loss": min(r["after_a"])})
        print(f"sigma={s:.0e} final {r['after_a'][-1]:.2e}")
    results["noise"] = noise

    plt.figure(figsize=(6.5, 4.4))
    plt.loglog([n["sigma"] for n in noise], [n["final_loss"] for n in noise], "o-",
               label="final eval MSE (clean eval, 20 alt.)")
    sg = np.array(sigmas)
    plt.loglog(sg, sg ** 2, "k--", lw=1, label=r"$\sigma^2$ reference")
    plt.xlabel(r"label noise std $\sigma$ on teacher outputs")
    plt.ylabel("eval MSE")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    plt.title("Noise robustness of the two-block solver")
    plt.tight_layout()
    plt.savefig("reports/two_block_noise.png", dpi=150)
    plt.close()

    with open("reports/two_block_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("Saved reports/two_block_convergence.png, two_block_batch_ablation.png, "
          "two_block_noise.png, two_block_results.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {time.time() - t0:.1f}s")
