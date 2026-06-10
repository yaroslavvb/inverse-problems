# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
# ]
# ///
"""Learning without Forgetting: streaming student/teacher on linear attention.

Setting: every round a SMALL batch (B=2 sequences) arrives from the teacher and
the student fits it aggressively (K inner steps). With 2*D^2 parameters and only
B*N*D equations per round the per-round fit is underdetermined, so fitting the
current batch can destroy the fit on earlier batches. Methods differ in how
they constrain the round update to avoid that.

The spectral-descent analogy (verified numerically by --verify):
the model out_n = sum_{m<=n} (x_n^T A x_m) x_m^T W_v with A = W_q W_k^T is
bilinear in (A, W_v). For ANY input sequence with orthonormal rows,

    ||Delta out_n||_2 <= ||DA||_2 * ||W_v + DW_v||_2 + ||A||_2 * ||DW_v||_2

i.e. the SPECTRAL norms of the parameter changes bound the output change on
every possible input — worst-case forgetting. Frobenius norms bound only the
AVERAGE output change over random inputs (a rank-1 update with the same
Frobenius norm changes worst-case outputs ~sqrt(D) more than an isotropic one).
Hence: Muon's msign direction = steepest descent per unit of worst-case
forgetting budget; GD = steepest descent per unit of average-case budget.

Methods (each fits the current batch with K Adam steps unless noted):
  naive        unconstrained per-round fit (lr swept)
  frob_anchor  + lam * ||theta - theta_round_start||_F^2  (average-case trust region)
  func_anchor  + lam * ||f(X_t; theta) - f(X_t; theta_round_start)||^2 (LwF-style)
  spec_clip    unconstrained fit, then round delta projected onto spectral ball
               ||Delta||_2 <= tau per matrix (worst-case trust region, projection)
  muon         K steps theta -= eta * msign(momentum); total spectral change <= K*eta
               (worst-case trust region, steepest-descent version)
  replay       reservoir buffer of past sequences (cap swept); each inner step
               trains on current batch + 8 sampled buffer sequences
  ewc          online diagonal-Fisher penalty lam * sum F * (theta - anchor)^2
  null_proj    GPM-style: per-step updates projected off the span of past-batch
               feature rows (exact null-space of past outputs per block)

Protocol: D=32, N=8, ROUNDS=40, B=2, K=20, float64, 3 seeds, shared teacher
(make_problem(42)) and held-out eval set; per-round metrics: eval MSE (overall
error), past-seen MSE (forgetting), current-batch MSE (fit quality).
Runs execute in parallel via ProcessPoolExecutor (one torch thread per worker).

Outputs: reports/forgetting_*.png, reports/forgetting_results.json.
"""

import os
import sys
import json
import math
import time
import itertools
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ortho_updates import DIM, gen_batch, mse, make_problem

SEQ_LEN = 8
ROUNDS = 40
BATCH = 2
INNER_STEPS = 20
BASE_LR = 1e-2
REPLAY_SAMPLE = 8
SEEDS = (0, 1, 2)
EVAL_BATCH = 128


# ---------------------------------------------------------------- model

def forward(X, A, Wv):
    """Linear attention in the identifiable parameterization (A, W_v)."""
    scores = torch.tril(torch.einsum("bnd,bmd->bnm", X @ A, X))
    return torch.einsum("bnm,bme->bne", scores, X @ Wv)


def rand_orth(gen, dim=DIM):
    H = torch.randn(dim, dim, generator=gen)
    Q, R = torch.linalg.qr(H)
    return Q * torch.diagonal(R).sign()


def spec_project(delta, tau):
    """Project onto the spectral-norm ball {||D||_2 <= tau} (clip singular values)."""
    U, S, Vh = torch.linalg.svd(delta)
    return U @ torch.diag(S.clamp(max=tau)) @ Vh


def msign_safe(Z, rel_eps=1e-12):
    """msign(Z) = U_r V_r^T, zeroing the null space. Batch gradients here are
    rank-deficient (rank <= B*N = 16 of 32), and a polar map with an absolute
    eigenvalue clamp amplifies the numerically-zero directions into unit-norm
    noise — that variant diverged at every step size."""
    w, Q = torch.linalg.eigh(Z.T @ Z)
    inv = torch.where(w > rel_eps * w.max(), w.clamp_min(1e-300).rsqrt(),
                      torch.zeros_like(w))
    return Z @ (Q * inv) @ Q.T


def eigh_inv_sqrt(S):
    """(S)^(-1/2) for symmetric positive-definite S (Gram + damping)."""
    w, Q = torch.linalg.eigh(S)
    return Q @ torch.diag(w.clamp_min(1e-12).rsqrt()) @ Q.T


# Anisotropic gauge for the factored (W_q, W_k) student: W_q0 = A0 D^-1,
# W_k0 = D gives the SAME function as the (A, W_v) init (W_q0 W_k0^T = A0)
# but condition-10 unbalanced factor spectra — the regime compositional Muon
# targets (trained transformers have unbalanced QK factor spectra).
ANISO = torch.logspace(-0.5, 0.5, DIM)
CM_DAMPING = 1e-2   # lambda in C = (W^T W + lambda I)^(1/2), Tilde's default


# ---------------------------------------------------------------- per-round methods
# Each takes (state, Xb, Yb, mem, hyper); state = {'A','Wv'} leaf tensors.


def _adam_fit(state, K, lr, loss_fn):
    opt = torch.optim.Adam([state["A"], state["Wv"]], lr=lr)
    for _ in range(K):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()


def round_naive(state, Xb, Yb, mem, hyper):
    _adam_fit(state, INNER_STEPS, hyper["lr"],
              lambda: mse(forward(Xb, state["A"], state["Wv"]), Yb))


def round_frob_anchor(state, Xb, Yb, mem, hyper):
    A0, W0 = state["A"].detach().clone(), state["Wv"].detach().clone()
    lam = hyper["lam"]

    def loss_fn():
        return (mse(forward(Xb, state["A"], state["Wv"]), Yb)
                + lam * (((state["A"] - A0) ** 2).mean()
                         + ((state["Wv"] - W0) ** 2).mean()))
    _adam_fit(state, INNER_STEPS, BASE_LR, loss_fn)


def round_func_anchor(state, Xb, Yb, mem, hyper):
    with torch.no_grad():
        Y0 = forward(Xb, state["A"], state["Wv"])
    lam = hyper["lam"]

    def loss_fn():
        pred = forward(Xb, state["A"], state["Wv"])
        return mse(pred, Yb) + lam * mse(pred, Y0)
    _adam_fit(state, INNER_STEPS, BASE_LR, loss_fn)


def round_spec_clip(state, Xb, Yb, mem, hyper):
    A0, W0 = state["A"].detach().clone(), state["Wv"].detach().clone()
    round_naive(state, Xb, Yb, mem, {"lr": BASE_LR})
    with torch.no_grad():
        state["A"].copy_(A0 + spec_project(state["A"] - A0, hyper["tau"]))
        state["Wv"].copy_(W0 + spec_project(state["Wv"] - W0, hyper["tau"]))


def round_muon(state, Xb, Yb, mem, hyper):
    """Per-round Muon: momentum is reset each round (unlike streaming Muon)
    so the K*eta spectral budget is per-round clean. Note msign of a tiny
    momentum still has unit spectral norm, so the method spends its FULL
    worst-case budget every round and floors instead of converging."""
    eta = hyper["eta"]
    M = {n: torch.zeros(DIM, DIM) for n in ("A", "Wv")}
    for _ in range(INNER_STEPS):
        loss = mse(forward(Xb, state["A"], state["Wv"]), Yb)
        gA, gW = torch.autograd.grad(loss, [state["A"], state["Wv"]])
        with torch.no_grad():
            M["A"] = 0.9 * M["A"] + gA
            M["Wv"] = 0.9 * M["Wv"] + gW
            state["A"].sub_(eta * msign_safe(M["A"]))
            state["Wv"].sub_(eta * msign_safe(M["Wv"]))


def _factored_init(state, mem):
    """Lazily split state's A into unbalanced factors (same function, cond-10 spectra)."""
    if "Wq" not in mem:
        with torch.no_grad():
            mem["Wq"] = (state["A"].detach() @ torch.diag(1.0 / ANISO)).requires_grad_(True)
            mem["Wk"] = torch.diag(ANISO).clone().requires_grad_(True)
    return mem["Wq"], mem["Wk"]


def _factored_round(state, Xb, Yb, mem, eta, compositional):
    """Shared loop for the two factored Muons. Per inner step the QK pair gets a
    combined update; W_v gets a plain msign step of spectral norm eta.

    compositional=True (Tilde's partner-whitened half-split rule):
        dWq = -(eta/2) msign(Mq C_k^-1) C_k^-1,  C_k = (W_k^T W_k + lam I)^(1/2)
        dWk = -(eta/2) msign(Mk C_q^-1) C_q^-1,  C_q symmetric
    which bounds the COMPOSED update ||d(W_q W_k^T)||_2 <= eta to first order —
    the quantity the layer's forgetting bound actually charges.
    compositional=False: naive per-factor msign with the same eta/2 budgets,
    which bounds the factors but lets the composed change blow up with the
    partner's spectral norm when factors are unbalanced."""
    Wq, Wk = _factored_init(state, mem)
    Wv = state["Wv"]
    I = torch.eye(DIM)
    M = {k: torch.zeros(DIM, DIM) for k in ("q", "k", "v")}
    for _ in range(INNER_STEPS):
        loss = mse(forward(Xb, Wq @ Wk.T, Wv), Yb)
        gq, gk, gv = torch.autograd.grad(loss, [Wq, Wk, Wv])
        with torch.no_grad():
            M["q"] = 0.9 * M["q"] + gq
            M["k"] = 0.9 * M["k"] + gk
            M["v"] = 0.9 * M["v"] + gv
            if compositional:
                Ck_inv = eigh_inv_sqrt(Wk.T @ Wk + CM_DAMPING * I)
                Cq_inv = eigh_inv_sqrt(Wq.T @ Wq + CM_DAMPING * I)
                Wq.sub_((eta / 2) * msign_safe(M["q"] @ Ck_inv) @ Ck_inv)
                Wk.sub_((eta / 2) * msign_safe(M["k"] @ Cq_inv) @ Cq_inv)
            else:
                Wq.sub_((eta / 2) * msign_safe(M["q"]))
                Wk.sub_((eta / 2) * msign_safe(M["k"]))
            Wv.sub_(eta * msign_safe(M["v"]))
    with torch.no_grad():
        state["A"].copy_(Wq @ Wk.T)   # keep the shared metrics in sync


def round_comp_muon(state, Xb, Yb, mem, hyper):
    _factored_round(state, Xb, Yb, mem, hyper["eta"], compositional=True)


def round_factored_muon(state, Xb, Yb, mem, hyper):
    _factored_round(state, Xb, Yb, mem, hyper["eta"], compositional=False)


def round_replay(state, Xb, Yb, mem, hyper):
    cap = hyper["cap"]
    bufX, bufY = mem.get("bufX"), mem.get("bufY")

    def loss_fn():
        if bufX is None:
            X, Y = Xb, Yb
        else:
            k = min(REPLAY_SAMPLE, bufX.shape[0])
            idx = torch.randint(bufX.shape[0], (k,), generator=mem["gen"])
            X = torch.cat([Xb, bufX[idx]])
            Y = torch.cat([Yb, bufY[idx]])
        return mse(forward(X, state["A"], state["Wv"]), Y)
    _adam_fit(state, INNER_STEPS, BASE_LR, loss_fn)

    # reservoir update with the round's sequences
    with torch.no_grad():
        newX, newY = Xb, Yb
        if bufX is None:
            mem["bufX"], mem["bufY"] = newX.clone(), newY.clone()
        elif bufX.shape[0] + newX.shape[0] <= cap:
            mem["bufX"] = torch.cat([bufX, newX])
            mem["bufY"] = torch.cat([bufY, newY])
        else:
            mem["seen"] = mem.get("seen", bufX.shape[0])
            for i in range(newX.shape[0]):
                mem["seen"] += 1
                j = int(torch.randint(mem["seen"], (1,), generator=mem["gen"]))
                if j < cap:
                    mem["bufX"][j] = newX[i]
                    mem["bufY"][j] = newY[i]


def round_ewc(state, Xb, Yb, mem, hyper):
    """Online EWC with the Gauss-Newton (true Fisher) diagonal. The textbook
    'squared gradient' proxy is the squared batch-mean gradient of an
    already-fitted batch — residual-scale, ~6 orders below the GN diagonal
    here, which made the penalty inert; the GN diagonal comes free from the
    same Jacobian factors the model is linear in per block."""
    lam = hyper["lam"]
    if "F_A" not in mem:
        mem["F_A"] = torch.zeros(DIM, DIM)
        mem["F_W"] = torch.zeros(DIM, 1)
    anchA, anchW = state["A"].detach().clone(), state["Wv"].detach().clone()
    F_A, F_W = mem["F_A"], mem["F_W"]

    def loss_fn():
        return (mse(forward(Xb, state["A"], state["Wv"]), Yb)
                + lam * ((F_A * (state["A"] - anchA) ** 2).sum()
                         + (F_W * (state["Wv"] - anchW) ** 2).sum()))
    _adam_fit(state, INNER_STEPS, BASE_LR, loss_fn)

    # accumulate GN/Fisher diagonal at round-end params
    with torch.no_grad():
        A, Wv = state["A"], state["Wv"]
        M = Yb.numel()
        scores = torch.tril(torch.einsum("bnd,bmd->bnm", Xb @ A, Xb))
        feats_W = torch.einsum("bnm,bmd->bnd", scores, Xb).reshape(-1, DIM)
        V = Xb @ Wv
        C = torch.cumsum(V.unsqueeze(-1) * Xb.unsqueeze(2), dim=1)
        Phi = torch.einsum("bnd,bnef->bnedf", Xb, C).reshape(-1, DIM * DIM)
        mem["F_A"] += (2.0 / M) * (Phi ** 2).sum(0).reshape(DIM, DIM)
        mem["F_W"] += (2.0 / M) * (feats_W ** 2).sum(0)[:, None]  # same for every column


def _proj_off(vec, P):
    """vec minus its projection onto the column space of P (or vec if P empty)."""
    if P is None or P.shape[1] == 0:
        return vec
    return vec - P @ (P.T @ vec)


def round_null_proj(state, Xb, Yb, mem, hyper):
    tol = hyper["tol"]
    PA, PW = mem.get("PA"), mem.get("PW")
    opt = torch.optim.Adam([state["A"], state["Wv"]], lr=BASE_LR)
    for _ in range(INNER_STEPS):
        A_pre = state["A"].detach().clone()
        W_pre = state["Wv"].detach().clone()
        opt.zero_grad()
        mse(forward(Xb, state["A"], state["Wv"]), Yb).backward()
        opt.step()
        with torch.no_grad():
            dA = (state["A"] - A_pre).reshape(-1, 1)
            state["A"].copy_(A_pre + _proj_off(dA, PA).reshape(DIM, DIM))
            dW = state["Wv"] - W_pre
            if PW is not None and PW.shape[1] > 0:
                dW = dW - PW @ (PW.T @ dW)      # left-project rows of the update
            state["Wv"].copy_(W_pre + dW)

    # Extend bases with this round's feature rows (computed at round-end params).
    # HONESTY NOTE (verified by adversarial review): protection is exact only
    # while the bases have room. Each round contributes rank-16 W_v features in
    # a 32-dim space and rank-72 A features in a 1024-dim space, so the W_v cap
    # saturates by round ~3 and the A cap by round ~14 — after that, later
    # batches are only approximately protected and updates live in a shrinking
    # free subspace. This is structural for exact null-space methods at this
    # data/parameter ratio, not a tuning issue (the tol knob barely moves
    # results, and removing the caps diverges: the relative threshold then
    # ratchets in noise directions past full rank).
    with torch.no_grad():
        A, Wv = state["A"], state["Wv"]
        # W_v block: faithful GPM — accumulate the feature covariance (32x32)
        # and recompress to the top eigendirections each round (cheap at D=32)
        scores = torch.tril(torch.einsum("bnd,bmd->bnm", Xb @ A, Xb))
        feats_W = torch.einsum("bnm,bmd->bnd", scores, Xb).reshape(-1, DIM)   # (B*N, D)
        mem["S_W"] = mem.get("S_W", torch.zeros(DIM, DIM)) + feats_W.T @ feats_W
        w, Q = torch.linalg.eigh(mem["S_W"])
        sel = Q[:, w > (tol ** 2) * w.max()]
        mem["PW"] = sel[:, -(DIM - 1):]
        # A block: outputs linear in vec(A) through Phi rows; covariance
        # recompression would need a 1024^3 eigh per round (breaks the <1s
        # budget), so append new residual directions and truncate at the cap
        V = Xb @ Wv
        C = torch.cumsum(V.unsqueeze(-1) * Xb.unsqueeze(2), dim=1)
        Phi = torch.einsum("bnd,bnef->bnedf", Xb, C).reshape(-1, DIM * DIM)   # (B*N*D, D^2)
        P = mem.get("PA")
        R = Phi.T if P is None else Phi.T - P @ (P.T @ Phi.T)
        # round features have rank <= B*N(N+1)/2 = 72, so a randomized
        # rank-96 SVD captures them exactly at ~10x less cost than full SVD
        q = min(96, R.shape[0], R.shape[1])
        U, S, _ = torch.svd_lowrank(R, q=q)
        keep = U[:, S > tol * (S.max() + 1e-30)]
        P = keep if P is None else torch.cat([P, keep], dim=1)
        mem["PA"] = P[:, :DIM * DIM - 32]


METHODS = {
    # name: (round_fn, hyper grid)
    "naive":       (round_naive,       [{"lr": v} for v in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)]),
    "frob_anchor": (round_frob_anchor, [{"lam": v} for v in (0.1, 1.0, 10.0, 100.0)]),
    "func_anchor": (round_func_anchor, [{"lam": v} for v in (0.1, 1.0, 10.0, 100.0)]),
    "spec_clip":   (round_spec_clip,   [{"tau": v} for v in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]),
    "muon":        (round_muon,        [{"eta": v} for v in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)]),
    "comp_muon":   (round_comp_muon,   [{"eta": v} for v in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)]),
    "factored_muon": (round_factored_muon, [{"eta": v} for v in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)]),
    "replay":      (round_replay,      [{"cap": v} for v in (8, 32, 1000)]),
    "ewc":         (round_ewc,         [{"lam": v} for v in (0.01, 0.03, 0.1, 0.3, 1.0)]),
    "null_proj":   (round_null_proj,   [{"tol": v} for v in (0.3, 0.1, 0.03)]),
}


# ---------------------------------------------------------------- streaming run

def run_stream(method, hyper, seed):
    torch.set_num_threads(1)
    t0 = time.time()
    teacher, _, eval_x, eval_y = make_problem(42)        # shared teacher + eval set
    gen_init = torch.Generator().manual_seed(7 * seed + 5)
    state = {"A": rand_orth(gen_init).requires_grad_(True),
             "Wv": rand_orth(gen_init).requires_grad_(True)}
    gen_stream = torch.Generator().manual_seed(1000 + seed)
    mem = {"gen": torch.Generator().manual_seed(500 + seed)}
    round_fn, _ = METHODS[method]

    pastX, pastY = [], []
    eval_curve, past_curve, fit_curve = [], [], []
    for t in range(ROUNDS):
        Xb = gen_batch(gen_stream, batch_size=BATCH, num_rows=SEQ_LEN)
        with torch.no_grad():
            Yb = teacher(Xb)
        round_fn(state, Xb, Yb, mem, hyper)
        with torch.no_grad():
            el = mse(forward(eval_x, state["A"], state["Wv"]), eval_y).item()
            eval_curve.append(el)
            fit_curve.append(mse(forward(Xb, state["A"], state["Wv"]), Yb).item())
            if pastX:
                PX, PY = torch.cat(pastX), torch.cat(pastY)
                past_curve.append(mse(forward(PX, state["A"], state["Wv"]), PY).item())
            else:
                past_curve.append(float("nan"))
        pastX.append(Xb)
        pastY.append(Yb)
        if not math.isfinite(el):     # diverged: freeze the record and stop
            pad = ROUNDS - t - 1
            eval_curve += [float("inf")] * pad
            past_curve += [float("inf")] * pad
            fit_curve += [float("inf")] * pad
            break

    finite_past = [x for x in past_curve if math.isfinite(x)]
    finite_fit = [x for x in fit_curve if math.isfinite(x)]
    late_past = [x for x in past_curve[-10:] if math.isfinite(x)]
    return {
        "method": method, "hyper": hyper, "seed": seed,
        "eval_curve": eval_curve, "past_curve": past_curve, "fit_curve": fit_curve,
        "final_eval": eval_curve[-1],
        "mean_past": float(np.mean(finite_past)) if finite_past else float("inf"),
        # late-window forgetting: the all-round mean is dominated by the early
        # not-yet-learned transient, which compresses steady-state differences
        "late_past": float(np.mean(late_past)) if late_past else float("inf"),
        "mean_fit": float(np.mean(finite_fit)) if finite_fit else float("inf"),
        "seconds": time.time() - t0,
    }


def _worker(args):
    return run_stream(*args)


# ---------------------------------------------------------------- verification

def verify():
    print("Learning-without-forgetting pre-flight checks (float64)")
    teacher, _, eval_x, eval_y = make_problem(42)
    A_star = (teacher.W_q @ teacher.W_k.T).detach()
    W_star = teacher.W_v.detach()

    # 1. forward equivalence with the (W_q, W_k, W_v) teacher
    with torch.no_grad():
        err = (forward(eval_x, A_star, W_star) - eval_y).abs().max().item()
    print(f"1. forward equivalence (A, W_v) vs teacher: max err {err:.2e} "
          f"{'OK' if err < 1e-12 else 'FAIL'}")

    # 2. spectral forgetting bound: ||Dout_n|| <= ||DA||_2 ||Wv'||_2 + ||A||_2 ||DWv||_2
    gen = torch.Generator().manual_seed(3)
    tauA, tauW = 0.07, 0.05
    dA = spec_project(torch.randn(DIM, DIM, generator=gen), tauA)
    dA *= tauA / torch.linalg.matrix_norm(dA, ord=2)
    dW = spec_project(torch.randn(DIM, DIM, generator=gen), tauW)
    dW *= tauW / torch.linalg.matrix_norm(dW, ord=2)
    bound = (tauA * torch.linalg.matrix_norm(W_star + dW, ord=2)
             + torch.linalg.matrix_norm(A_star, ord=2) * tauW).item()
    sup = 0.0
    for _ in range(50):
        X = gen_batch(gen, batch_size=40)
        with torch.no_grad():
            d = forward(X, A_star + dA, W_star + dW) - forward(X, A_star, W_star)
        sup = max(sup, d.norm(dim=-1).max().item())
    print(f"2. spectral bound: sup ||Dout_n|| over 2000 sequences {sup:.4f} <= bound {bound:.4f} "
          f"{'OK' if sup <= bound * (1 + 1e-9) else 'FAIL'}")

    # 3. worst-case vs average-case: rank-1 vs isotropic DA at equal Frobenius norm
    f = 0.1
    rank1 = torch.zeros(DIM, DIM)
    rank1[0, 0] = f
    iso = spec_project(torch.randn(DIM, DIM, generator=gen), 1.0)
    iso *= f / torch.linalg.norm(iso)
    aligned = torch.eye(DIM)[:SEQ_LEN].unsqueeze(0)   # rows e_1..e_N attain the rank-1 sup
    sups = []
    for dAx in (rank1, iso):
        s = 0.0
        for _ in range(50):
            X = torch.cat([gen_batch(gen, batch_size=40), aligned])
            with torch.no_grad():
                d = forward(X, A_star + dAx, W_star) - forward(X, A_star, W_star)
            s = max(s, d.norm(dim=-1).max().item())
        sups.append(s)
    print(f"3. equal-Frobenius worst case: rank-1 {sups[0]:.4f} vs isotropic {sups[1]:.4f} "
          f"(ratio {sups[0]/sups[1]:.1f}x; spectral norms {f:.2f} vs "
          f"{torch.linalg.matrix_norm(iso, ord=2).item():.3f})")

    # 4. null_proj invariance: projected update leaves past outputs unchanged
    gen2 = torch.Generator().manual_seed(11)
    Xb = gen_batch(gen2, batch_size=2)
    with torch.no_grad():
        Yb = teacher(Xb)
        scores = torch.tril(torch.einsum("bnd,bmd->bnm", Xb @ A_star, Xb))
        feats_W = torch.einsum("bnm,bmd->bnd", scores, Xb).reshape(-1, DIM)
        PW = torch.linalg.qr(feats_W.T)[0]
        dW = torch.randn(DIM, DIM, generator=gen2)
        dW = dW - PW @ (PW.T @ dW)
        change = (forward(Xb, A_star, W_star + dW) - forward(Xb, A_star, W_star)).abs().max()
        V = Xb @ W_star
        C = torch.cumsum(V.unsqueeze(-1) * Xb.unsqueeze(2), dim=1)
        Phi = torch.einsum("bnd,bnef->bnedf", Xb, C).reshape(-1, DIM * DIM)
        PA = torch.linalg.qr(Phi.T)[0]
        dA2 = torch.randn(DIM * DIM, 1, generator=gen2)
        dA2 = (dA2 - PA @ (PA.T @ dA2)).reshape(DIM, DIM)
        change_A = (forward(Xb, A_star + dA2, W_star) - forward(Xb, A_star, W_star)).abs().max()
    print(f"4. null_proj invariance: past-output change W_v-block {change.item():.2e}, "
          f"A-block {change_A.item():.2e} {'OK' if max(change.item(), change_A.item()) < 1e-9 else 'FAIL'}")

    # 5. underdetermination: equations per round vs parameters
    print(f"5. round equations B*N*D = {BATCH * SEQ_LEN * DIM} vs parameters 2*D^2 = {2 * DIM * DIM} "
          f"-> per-round fit underdetermined: {'OK' if BATCH * SEQ_LEN * DIM < 2 * DIM * DIM else 'FAIL'}")

    # 6. compositional budget: from the cond-10 unbalanced factored init, one
    #    round at eta=0.01 — comp_muon's realized composed change ||D(Wq Wk^T)||_2
    #    should track its K*eta budget, naive per-factor msign should overshoot it
    eta = 0.01
    realized = {}
    for name, fn in (("comp_muon", round_comp_muon), ("factored_muon", round_factored_muon)):
        gen3 = torch.Generator().manual_seed(5)
        st = {"A": rand_orth(torch.Generator().manual_seed(12)).requires_grad_(True),
              "Wv": rand_orth(torch.Generator().manual_seed(13)).requires_grad_(True)}
        m = {}
        Xb = gen_batch(gen3, batch_size=BATCH)
        with torch.no_grad():
            Yb = teacher(Xb)
            A0 = st["A"].detach().clone()
        fn(st, Xb, Yb, m, {"eta": eta})
        with torch.no_grad():
            realized[name] = torch.linalg.matrix_norm(st["A"] - A0, ord=2).item()
    budget = INNER_STEPS * eta
    print(f"6. composed-update budget K*eta = {budget:.3f} | realized ||D(WqWk^T)||_2: "
          f"comp_muon {realized['comp_muon']:.3f}, factored_muon {realized['factored_muon']:.3f} "
          f"{'OK' if realized['comp_muon'] <= budget * 1.25 < realized['factored_muon'] else 'CHECK'}")


# ---------------------------------------------------------------- sweep + outputs

def geo_mean(xs):
    """Geometric mean; non-finite entries (diverged runs) count as 1e9."""
    xs = [x if math.isfinite(x) else 1e9 for x in xs]
    xs = [max(x, 1e-300) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def hyper_str(h):
    return ", ".join(f"{k}={v:g}" for k, v in h.items())


def sweep():
    tasks = [(m, h, s) for m, (_, grid) in METHODS.items() for h in grid for s in SEEDS]
    workers = max(1, (os.cpu_count() or 8) - 2)
    print(f"{len(tasks)} runs on {workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        runs = list(ex.map(_worker, tasks))
    secs = sorted(r["seconds"] for r in runs)
    # the slowest few runs are first-task-per-worker torch warmup, not run cost
    print(f"sweep wall-clock {time.time()-t0:.1f}s | median run {secs[len(secs)//2]:.2f}s | "
          f"warm max {secs[-workers-1] if len(secs) > workers else secs[-1]:.2f}s "
          f"(absolute max {secs[-1]:.2f}s incl. per-worker warmup)")

    results = {}
    for m in METHODS:
        per_hyper = {}
        for r in [r for r in runs if r["method"] == m]:
            per_hyper.setdefault(json.dumps(r["hyper"]), []).append(r)
        configs = []
        for hjson, rs in per_hyper.items():
            configs.append({
                "hyper": json.loads(hjson),
                "final_eval_geo": geo_mean([r["final_eval"] for r in rs]),
                "final_eval_per_seed": [r["final_eval"] for r in rs],
                "mean_past_geo": geo_mean([r["mean_past"] for r in rs]),
                "late_past_geo": geo_mean([r["late_past"] for r in rs]),
                "mean_fit_geo": geo_mean([r["mean_fit"] for r in rs]),
                "eval_curves": [r["eval_curve"] for r in rs],
                "past_curves": [r["past_curve"] for r in rs],
                "fit_curves": [r["fit_curve"] for r in rs],
                "seconds": [r["seconds"] for r in rs],
            })
        configs.sort(key=lambda c: c["final_eval_geo"])
        results[m] = {"configs": configs, "best": configs[0]}
        b = configs[0]
        print(f"{m:12s} best {hyper_str(b['hyper']):14s} | final eval {b['final_eval_geo']:.2e} "
              f"| late past {b['late_past_geo']:.2e} | mean fit {b['mean_fit_geo']:.2e}")
    return results


def save_outputs(results):
    os.makedirs("reports", exist_ok=True)
    xs = np.arange(1, ROUNDS + 1)

    def geo_curve(curves):
        a = np.array(curves, dtype=float)
        a = np.nan_to_num(a, nan=np.nan, posinf=1e9)   # keep diverged runs visible
        a = np.maximum(a, 1e-300)
        with np.errstate(all="ignore"):
            out = np.exp(np.nanmean(np.log(a), axis=0))
        return out

    # 1. overall error over rounds (best config per method)
    plt.figure(figsize=(10, 6))
    for m, res in results.items():
        b = res["best"]
        plt.semilogy(xs, geo_curve(b["eval_curves"]),
                     label=f"{m} ({hyper_str(b['hyper'])})", alpha=0.9)
    plt.xlabel("round (2 fresh sequences per round)")
    plt.ylabel("eval MSE (held-out)")
    plt.title("Overall error in the streaming setting (best config per method, geo-mean of 3 seeds)")
    plt.legend(fontsize=7.5)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/forgetting_eval_curves.png", dpi=150)
    plt.close()

    # 2. forgetting and fit dynamics (past curve starts at round 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    for m, res in results.items():
        b = res["best"]
        ax1.semilogy(xs[1:], geo_curve(b["past_curves"])[1:], label=m, alpha=0.9)
        ax2.semilogy(xs, geo_curve(b["fit_curves"]), label=m, alpha=0.9)
    ax1.set_xlabel("round")
    ax1.set_ylabel("MSE on all previously seen batches")
    ax1.set_title("Forgetting: error on past observations")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", alpha=0.3)
    ax2.set_xlabel("round")
    ax2.set_ylabel("MSE on current batch after its round")
    ax2.set_title("Fit: error on the just-fitted batch")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/forgetting_dynamics.png", dpi=150)
    plt.close()

    # 3. strength-vs-error per method (small multiples)
    names = list(METHODS)
    ncols = math.ceil(len(names) / 2)
    fig, axes = plt.subplots(2, ncols, figsize=(13, 6))
    for ax, m in zip(axes.flat, names):
        cfgs = sorted(results[m]["configs"], key=lambda c: list(c["hyper"].values())[0])
        hv = [list(c["hyper"].values())[0] for c in cfgs]
        ax.plot(hv, [c["final_eval_geo"] for c in cfgs], "o-", label="final eval")
        ax.plot(hv, [c["mean_past_geo"] for c in cfgs], "s--", alpha=0.7, label="mean past")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel(list(cfgs[0]["hyper"].keys())[0], fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Strength of the forgetting-avoidance knob vs overall error", fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/forgetting_strength.png", dpi=150)
    plt.close()

    # 4. fit-vs-forget trade-off (every config; late-window forgetting)
    plt.figure(figsize=(8.5, 6))
    markers = dict(zip(names, "osd^v<>pXP*"))
    for m, res in results.items():
        f = [c["mean_fit_geo"] for c in res["configs"]]
        p = [c["late_past_geo"] for c in res["configs"]]
        e = [c["final_eval_geo"] for c in res["configs"]]
        sc = plt.scatter(f, p, c=np.log10(e), cmap="viridis", vmin=-8, vmax=-2,
                         marker=markers[m], s=70, label=m, edgecolors="k", linewidths=0.4)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("mean MSE on the just-fitted batch (lower = fits harder)")
    plt.ylabel("MSE on previously seen data, last 10 rounds (lower = forgets less)")
    plt.colorbar(sc, label="log10 final eval MSE")
    plt.title("Fit-vs-forget trade-off (every config; color = overall error)")
    plt.legend(fontsize=7.5)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/forgetting_tradeoff.png", dpi=150)
    plt.close()

    def jsonable(obj):
        """Strict JSON: non-finite floats become None."""
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [jsonable(v) for v in obj]
        return obj

    summary = {"config": {"dim": DIM, "seq_len": SEQ_LEN, "rounds": ROUNDS,
                          "batch": BATCH, "inner_steps": INNER_STEPS,
                          "base_lr": BASE_LR, "seeds": list(SEEDS),
                          "replay_sample": REPLAY_SAMPLE}}
    for m, res in results.items():
        summary[m] = {"best_hyper": res["best"]["hyper"],
                      "configs": res["configs"]}
    with open("reports/forgetting_results.json", "w") as f:
        json.dump(jsonable(summary), f, indent=1)
    print("Saved reports/forgetting_eval_curves.png, forgetting_dynamics.png, "
          "forgetting_strength.png, forgetting_tradeoff.png, forgetting_results.json")


if __name__ == "__main__":
    t0 = time.time()
    if "--verify" in sys.argv:
        verify()
    elif "--smoke" in sys.argv:
        for m in METHODS:
            r = run_stream(m, METHODS[m][1][len(METHODS[m][1]) // 2], 0)
            print(f"{m:12s} final eval {r['final_eval']:.2e} | mean past {r['mean_past']:.2e} "
                  f"| mean fit {r['mean_fit']:.2e} | {r['seconds']:.2f}s")
    else:
        save_outputs(sweep())
    print(f"Total time: {time.time() - t0:.1f}s")
