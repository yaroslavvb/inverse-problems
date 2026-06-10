# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
# ]
# ///
"""Orthogonality-exploiting updates for the single-layer linear attention student/teacher task.

Both teacher and student weights W_q, W_k, W_v live on the orthogonal group O(D).
This script compares update rules that exploit that structure (multiplicative
rotation updates, Lie-algebra optimizers, retractions, landing, Procrustes /
two-block exact solver) against Euclidean baselines (SGD, Adam), with a
per-method learning-rate sweep.

Conventions (verified by first-order expansion; see --verify):
  skew(A) = (A - A^T) / 2
  Riemannian gradient at W in O(n), right chart: Omega = skew(W^T G)
  Multiplicative update: W <- W @ expm(-lr * Omega)
  First-order check: <G, dW> = -lr <skew(W^T G), Omega> = -lr ||Omega||_F^2 <= 0.
  Landing uses the left form skew(G W^T) @ W, valid off-manifold (Ablin & Peyre).

Determinant obstruction: multiplicative/retraction updates preserve det(W) = +-1.
Zero loss needs det(W_v) = det(W_v*) and det(W_q) det(W_k) = det(W_q* W_k*^T);
a random init violates this with probability 3/4 and then floors at ~1e-3.
We therefore run an "aligned" arm (student init det-matched to the teacher by
column sign flips) and an "unaligned" arm to document the floor.

Everything runs in float64: the <=1e-15 loss regime is below float32 resolution.

Outputs: reports/ortho_updates_results.json plus PNG plots in reports/.
"""

import os
import json
import math
import time
import copy
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

from linattention_solve import LinearSelfAttention

DIM = 32
SEQ_LEN = 8
BATCH_SIZE = 32
NUM_STEPS = 800
THRESHOLDS = [1e-12, 1e-15]   # steps-to-threshold metrics (headline: 1e-12)
CONVERGED_MSE = 1e-16         # stop a run early below this
DIVERGED_MSE = 1e3

PARAM_NAMES = ["W_q", "W_k", "W_v"]


# ---------------------------------------------------------------- utilities

def skew(A):
    return (A - A.T) / 2.0


def ortho_error(W):
    """||W^T W - I||_F"""
    return torch.linalg.norm(W.T @ W - torch.eye(W.shape[0])).item()


def qf(A):
    """Q factor of QR with sign-fixed diagonal of R (canonical representative)."""
    Q, R = torch.linalg.qr(A)
    return Q * torch.diagonal(R).sign()


def polar(Z, eps=1e-30):
    """Polar factor (closest orthogonal matrix / orthogonal Procrustes solution).

    eigh-based: gesdd SVD crashes on matrices with paired singular values
    (skew or near-orthogonal inputs), which these methods produce routinely.
    """
    w, Q = torch.linalg.eigh(Z.T @ Z)
    return Z @ (Q * w.clamp_min(eps).rsqrt()) @ Q.T


def det_sign(W):
    return torch.linalg.slogdet(W)[0].item()


def gen_batch(gen, batch_size=BATCH_SIZE, num_rows=SEQ_LEN, dim=DIM):
    """Batch of row-orthogonal inputs drawn from an explicit torch.Generator."""
    xs = []
    for _ in range(batch_size):
        H = torch.randn(dim, dim, generator=gen)
        Q, R = torch.linalg.qr(H)
        xs.append((Q * torch.diagonal(R).sign())[:num_rows])
    return torch.stack(xs)


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


def loss_and_grads(model, X, Y):
    """Returns (loss value, {param_name: euclidean grad})."""
    loss = mse(model(X), Y)
    params = [getattr(model, n) for n in PARAM_NAMES]
    grads = torch.autograd.grad(loss, params)
    return loss.item(), dict(zip(PARAM_NAMES, grads))


def align_determinants(student_state, teacher):
    """Column sign flips so the student init can reach the teacher under
    det-preserving updates: match det(W_v) and det(W_q) * det(W_k)."""
    state = copy.deepcopy(student_state)
    if det_sign(state["W_v"]) != det_sign(teacher.W_v):
        state["W_v"][:, 0] *= -1
    prod_s = det_sign(state["W_q"]) * det_sign(state["W_k"])
    prod_t = det_sign(teacher.W_q) * det_sign(teacher.W_k)
    if prod_s != prod_t:
        state["W_q"][:, 0] *= -1
    return state


# ---------------------------------------------------------------- optimizers
# Each optimizer is a class with .step(model, X, Y) -> float (train loss).
# All mutate model weights in-place under no_grad.


class EuclideanSGD:
    def __init__(self, model, lr):
        self.lr = lr

    def step(self, model, X, Y):
        loss, grads = loss_and_grads(model, X, Y)
        with torch.no_grad():
            for n in PARAM_NAMES:
                getattr(model, n).sub_(self.lr * grads[n])
        return loss


class EuclideanAdam:
    """Plain Adam on raw matrix entries (the baseline from linattention_solve)."""

    def __init__(self, model, lr):
        self.opt = torch.optim.Adam([getattr(model, n) for n in PARAM_NAMES], lr=lr)

    def step(self, model, X, Y):
        loss = mse(model(X), Y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


class ExpSGD:
    """Multiplicative Riemannian SGD: W <- W expm(-lr skew(W^T G))."""

    def __init__(self, model, lr):
        self.lr = lr

    def step(self, model, X, Y):
        loss, grads = loss_and_grads(model, X, Y)
        with torch.no_grad():
            for n in PARAM_NAMES:
                W = getattr(model, n)
                Om = skew(W.T @ grads[n])
                W.copy_(W @ torch.linalg.matrix_exp(-self.lr * Om))
        return loss


class CayleySGD:
    """Multiplicative update via Cayley retraction: W <- W (I + a/2)^{-1} (I - a/2), a = lr*Omega."""

    def __init__(self, model, lr):
        self.lr = lr

    def step(self, model, X, Y):
        loss, grads = loss_and_grads(model, X, Y)
        with torch.no_grad():
            I = torch.eye(DIM)
            for n in PARAM_NAMES:
                W = getattr(model, n)
                A = self.lr * skew(W.T @ grads[n])
                W.copy_(W @ torch.linalg.solve(I + A / 2, I - A / 2))
        return loss


class ExpMomentum:
    """Heavy-ball momentum in so(n) with a max-rotation-angle trust region:
    M <- mu M + Omega; step = lr M clipped to spectral norm theta_max;
    W <- W expm(-step)."""

    def __init__(self, model, lr, beta=0.9, theta_max=0.3):
        self.lr, self.beta, self.theta_max = lr, beta, theta_max
        self.M = {n: torch.zeros(DIM, DIM) for n in PARAM_NAMES}

    def step(self, model, X, Y):
        loss, grads = loss_and_grads(model, X, Y)
        with torch.no_grad():
            for n in PARAM_NAMES:
                W = getattr(model, n)
                Om = skew(W.T @ grads[n])
                self.M[n] = self.beta * self.M[n] + Om
                step = self.lr * self.M[n]
                ang = torch.linalg.matrix_norm(step, ord=2)
                if ang > self.theta_max:
                    step = step * (self.theta_max / ang)
                W.copy_(W @ torch.linalg.matrix_exp(-step))
        return loss


class SOnAdam:
    """Adam whose moments live on the skew-projected gradient ('multiplicative Adam').

    m is skew, v is symmetric (elementwise square of a skew matrix), so
    m_hat / (sqrt(v_hat) + eps) stays skew elementwise; re-skew for safety.
    """

    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {n: torch.zeros(DIM, DIM) for n in PARAM_NAMES}
        self.v = {n: torch.zeros(DIM, DIM) for n in PARAM_NAMES}
        self.t = 0

    def step(self, model, X, Y):
        loss, grads = loss_and_grads(model, X, Y)
        self.t += 1
        with torch.no_grad():
            for n in PARAM_NAMES:
                W = getattr(model, n)
                Om = skew(W.T @ grads[n])
                self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * Om
                self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * Om * Om
                m_hat = self.m[n] / (1 - self.b1 ** self.t)
                v_hat = self.v[n] / (1 - self.b2 ** self.t)
                upd = skew(m_hat / (v_hat.sqrt() + self.eps))
                W.copy_(W @ torch.linalg.matrix_exp(-self.lr * upd))
        return loss


class QRRetraction:
    """Riemannian step in the tangent space, retracted via QR: W <- qf(W - lr * W Omega)."""

    def __init__(self, model, lr):
        self.lr = lr

    def step(self, model, X, Y):
        loss, grads = loss_and_grads(model, X, Y)
        with torch.no_grad():
            for n in PARAM_NAMES:
                W = getattr(model, n)
                xi = W @ skew(W.T @ grads[n])
                W.copy_(qf(W - self.lr * xi))
        return loss


class PolarRetraction:
    """Riemannian step retracted via the polar decomposition (closest orthogonal)."""

    def __init__(self, model, lr):
        self.lr = lr

    def step(self, model, X, Y):
        loss, grads = loss_and_grads(model, X, Y)
        with torch.no_grad():
            for n in PARAM_NAMES:
                W = getattr(model, n)
                xi = W @ skew(W.T @ grads[n])
                W.copy_(polar(W - self.lr * xi))
        return loss


class Landing:
    """Ablin & Peyre landing field: no retraction, attraction term pulls back to O(n).

    W <- W - lr * skew(G W^T) W - gamma * W (W^T W - I)
    The left form skew(G W^T) W is exactly orthogonal to the pull term for all W.
    The pull stepsize is DECOUPLED from lr (equivalent to Ablin-Peyre lam = gamma/lr):
    the descent term needs lr ~ 300 on this task while pull stability needs
    (pull stepsize) < ~1, so the coupled form fails at every lr.
    """

    def __init__(self, model, lr, gamma=0.3):
        self.lr, self.gamma = lr, gamma

    def step(self, model, X, Y):
        loss, grads = loss_and_grads(model, X, Y)
        with torch.no_grad():
            I = torch.eye(DIM)
            for n in PARAM_NAMES:
                W = getattr(model, n)
                W.sub_(self.lr * skew(grads[n] @ W.T) @ W
                       + self.gamma * W @ (W.T @ W - I))
        return loss


class PenaltyAdam:
    """Plain Adam on loss + mu * sum ||W^T W - I||_F^2 (soft constraint, coupled).
    mu is swept jointly with lr: at mu=1.0 the penalty gradient is ~450x the data
    gradient and Adam spends the whole update on constraint oscillation."""

    def __init__(self, model, lr, mu=0.01):
        self.mu = mu
        self.opt = torch.optim.Adam([getattr(model, n) for n in PARAM_NAMES], lr=lr)

    def step(self, model, X, Y):
        I = torch.eye(DIM)
        data_loss = mse(model(X), Y)
        penalty = sum(torch.linalg.norm(getattr(model, n).T @ getattr(model, n) - I) ** 2
                      for n in PARAM_NAMES)
        loss = data_loss + self.mu * penalty
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return data_loss.item()


class AdamDecoupledOrth:
    """Adam on the data loss; orthogonality pull applied OUTSIDE Adam
    (decoupled, like decoupled weight decay): W <- W - gamma W (W^T W - I).
    Folding the pull into the Adam gradient makes drift gamma-independent."""

    def __init__(self, model, lr, gamma=0.1):
        self.gamma = gamma
        self.opt = torch.optim.Adam([getattr(model, n) for n in PARAM_NAMES], lr=lr)

    def step(self, model, X, Y):
        loss = mse(model(X), Y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            I = torch.eye(DIM)
            for n in PARAM_NAMES:
                W = getattr(model, n)
                W.sub_(self.gamma * W @ (W.T @ W - I))
        return loss.item()


class MuonPolar:
    """Muon-style: orthogonalize the momentum (msign) as direction, retract via polar.
    Muon itself does NOT keep weights orthogonal; the polar retraction variant does.
    Note: polar(M) has constant norm sqrt(D) regardless of gradient magnitude, so
    with constant lr this orbits at an MSE floor ~ lr^2 instead of converging —
    a structural property of gradient-normalized updates, kept as a documented arm."""

    def __init__(self, model, lr, beta=0.9):
        self.lr, self.beta = lr, beta
        self.M = {n: torch.zeros(DIM, DIM) for n in PARAM_NAMES}

    def step(self, model, X, Y):
        loss, grads = loss_and_grads(model, X, Y)
        with torch.no_grad():
            for n in PARAM_NAMES:
                W = getattr(model, n)
                self.M[n] = self.beta * self.M[n] + grads[n]
                W.copy_(polar(W - self.lr * polar(self.M[n])))
        return loss


class ProcrustesAlternating:
    """Structure-exploiting: with W_q, W_k fixed, the map is linear in W_v
    (out = A @ W_v with A_n = sum_{m<=n} (Q_n . K_m) X_m), so the optimal
    orthogonal W_v has a closed form (orthogonal Procrustes). Alternate:
    multiplicative gradient steps on W_q, W_k; closed-form solve for W_v.
    """

    def __init__(self, model, lr):
        self.lr = lr

    def step(self, model, X, Y):
        loss, grads = loss_and_grads(model, X, Y)
        with torch.no_grad():
            for n in ["W_q", "W_k"]:
                W = getattr(model, n)
                Om = skew(W.T @ grads[n])
                W.copy_(W @ torch.linalg.matrix_exp(-self.lr * Om))
            # closed-form Procrustes for W_v given updated W_q, W_k
            Q, K = X @ model.W_q, X @ model.W_k
            scores = torch.tril(torch.einsum("bnd,bmd->bnm", Q, K))    # causal Q_n . K_m
            A = torch.einsum("bnm,bmd->bnd", scores, X)                # (B,N,D)
            M = A.reshape(-1, DIM).T @ Y.reshape(-1, DIM)              # D x D
            model.W_v.copy_(polar(M))
        return loss


class TrivializationAdam:
    """Lezcano-Casado trivialization: W = W0 @ expm(skew(A)), plain Adam on the
    unconstrained matrix A. Exactly on-manifold by construction.
    DYNAMIC REBASING is required: when ||skew(A)||_2 grows past theta the chart
    distorts and the static variant stalls around 1e-6; rebase W0 and reset A."""

    def __init__(self, model, lr, rebase_thresh=0.7):
        self.rebase_thresh = rebase_thresh
        self.W0 = {n: getattr(model, n).detach().clone() for n in PARAM_NAMES}
        self.A = {n: torch.zeros(DIM, DIM, requires_grad=True) for n in PARAM_NAMES}
        self.opt = torch.optim.Adam(list(self.A.values()), lr=lr)

    def step(self, model, X, Y):
        Ws = {n: self.W0[n] @ torch.linalg.matrix_exp(skew(self.A[n])) for n in PARAM_NAMES}
        Q, K, V = X @ Ws["W_q"], X @ Ws["W_k"], X @ Ws["W_v"]
        kv_state = torch.cumsum(torch.einsum("bnd,bne->bnde", K, V), dim=1)
        loss = mse(torch.einsum("bnd,bnde->bne", Q, kv_state), Y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for n in PARAM_NAMES:
                B = skew(self.A[n])
                if torch.linalg.matrix_norm(B, ord=2) > self.rebase_thresh:
                    self.W0[n] = self.W0[n] @ torch.linalg.matrix_exp(B)
                    self.A[n].zero_()  # Adam moments kept (fixed-chart approximation)
                    B = torch.zeros_like(B)
                getattr(model, n).copy_(self.W0[n] @ torch.linalg.matrix_exp(B))
        return loss.item()


METHODS = {
    # name: (factory, lr sweep, extra hyperparameter grid or None)
    "sgd":            (EuclideanSGD,           np.logspace(-3, 3, 13), None),
    "adam":           (EuclideanAdam,          np.logspace(-4, 0, 9),  None),
    "exp_sgd":        (ExpSGD,                 np.logspace(-3, 3, 13), None),
    "cayley_sgd":     (CayleySGD,              np.logspace(-3, 3, 13), None),
    "exp_momentum":   (ExpMomentum,            np.logspace(-2, 3, 11), None),
    "so_adam":        (SOnAdam,                np.logspace(-4, 0, 9),  None),
    "qr_retraction":  (QRRetraction,           np.logspace(-3, 3, 13), None),
    "polar_retraction": (PolarRetraction,      np.logspace(-3, 3, 13), None),
    "landing":        (Landing,                np.logspace(-2, 3, 11),
                       [{"gamma": 0.1}, {"gamma": 0.3}]),
    "penalty_adam":   (PenaltyAdam,            np.logspace(-4, 0, 9),
                       [{"mu": 1e-3}, {"mu": 1e-2}, {"mu": 1e-1}]),
    "adam_decoupled_orth": (AdamDecoupledOrth, np.logspace(-4, 0, 9),
                       [{"gamma": 0.03}, {"gamma": 0.1}, {"gamma": 0.3}]),
    "muon_polar":     (MuonPolar,              np.logspace(-4, 0, 9),  None),
    "procrustes_alt": (ProcrustesAlternating,  np.logspace(-2, 3, 11), None),
    "triv_adam":      (TrivializationAdam,     np.logspace(-4, 0, 9),  None),
}

# Methods whose updates preserve det(W) (and the det(W_q) det(W_k) product) and
# therefore cannot escape the wrong connected component of O(D)^3 from an
# unaligned init. procrustes_alt is only PARTIALLY det-preserving: its W_v polar
# solve crosses components freely, but the multiplicative W_q/W_k steps preserve
# the product, so it still floors when the q*k product is mismatched (as it is
# for the seed-42 unaligned init); its floor probability over random seeds is
# 1/2 rather than the 3/4 of the fully det-preserving family.
DET_PRESERVING = {"exp_sgd", "cayley_sgd", "exp_momentum", "so_adam",
                  "qr_retraction", "polar_retraction", "landing",
                  "procrustes_alt", "triv_adam"}


# ---------------------------------------------------------------- two-block exact solver

def two_block_exact(teacher, init_state, eval_x, eval_y, n_alt=15,
                    batch_size=64, data_seed=777):
    """Alternating exact solver exploiting bilinearity in (A, W_v), A = W_q W_k^T.

    Block V: orthogonal Procrustes (closed form) for W_v.
    Block A: least squares for A (the map is linear in A given V), polar-projected
    to O(D); then W_q <- A @ W_k with W_k frozen (gauge fix).
    Not an optimizer — a solver; reported in alternations, not steps.
    """
    student = LinearSelfAttention(DIM)
    student.load_state_dict(copy.deepcopy(init_state))
    gen = torch.Generator().manual_seed(data_seed)
    eval_losses, ortho_errs = [], []
    for _ in range(n_alt):
        X = gen_batch(gen, batch_size=batch_size)
        with torch.no_grad():
            Y = teacher(X)
            # ---- Block V: Procrustes
            Q, K = X @ student.W_q, X @ student.W_k
            scores = torch.tril(torch.einsum("bnd,bmd->bnm", Q, K))
            Afeat = torch.einsum("bnm,bmd->bnd", scores, X).reshape(-1, DIM)
            student.W_v.copy_(polar(Afeat.T @ Y.reshape(-1, DIM)))
            # ---- Block A: least squares + polar
            V = X @ student.W_v
            C = torch.cumsum(V.unsqueeze(-1) * X.unsqueeze(2), dim=1)   # (B,N,E,D): sum_m V_me X_mf
            Phi = torch.einsum("bnd,bnef->bnedf", X, C).reshape(-1, DIM * DIM)
            sol = torch.linalg.lstsq(Phi, Y.reshape(-1)).solution
            A = polar(sol.reshape(DIM, DIM))
            student.W_q.copy_(A @ student.W_k)
            el = mse(student(eval_x), eval_y).item()
        eval_losses.append(el)
        ortho_errs.append(max(ortho_error(getattr(student, n)) for n in PARAM_NAMES))
        if el <= CONVERGED_MSE:
            break
    return {"eval_losses": eval_losses, "ortho_errs": ortho_errs, "student": student}


# ---------------------------------------------------------------- harness

def make_problem(seed=42):
    """Fixed teacher, fixed student init (aligned + unaligned arms), fixed eval set."""
    torch.manual_seed(seed)
    teacher = LinearSelfAttention(DIM).requires_grad_(False)
    student_init = LinearSelfAttention(DIM)
    gen = torch.Generator().manual_seed(seed + 1)
    eval_x = gen_batch(gen, batch_size=128)
    with torch.no_grad():
        eval_y = teacher(eval_x)
    unaligned = copy.deepcopy(student_init.state_dict())
    aligned = align_determinants(unaligned, teacher)
    return teacher, {"aligned": aligned, "unaligned": unaligned}, eval_x, eval_y


def identifiable_distances(student, teacher):
    """Gauge-invariant distance: min over c != 0 of
    ||c A_s - A_t||_F^2 + ||W_v_s / c - W_v_t||_F^2,
    since the map is bilinear in (A, W_v) with A = W_q W_k^T, so
    (A, W_v) ~ (cA, W_v/c) is an exact model symmetry for any c != 0
    (on the manifold only c = +-1 survives, but off-manifold methods exploit
    the full scale gauge). Stationary c solves a c^4 - b c^3 + g c - f = 0."""
    with torch.no_grad():
        A_s, A_t = student.W_q @ student.W_k.T, teacher.W_q @ teacher.W_k.T
        V_s, V_t = student.W_v, teacher.W_v
        a, b = (A_s * A_s).sum().item(), (A_s * A_t).sum().item()
        f, g = (V_s * V_s).sum().item(), (V_s * V_t).sum().item()
        cst = (A_t * A_t).sum().item() + (V_t * V_t).sum().item()
    best = float("inf")
    for r in np.roots([a, -b, 0.0, g, -f]):
        if abs(r.imag) < 1e-9 and abs(r.real) > 1e-12:
            c = r.real
            best = min(best, a * c * c - 2 * b * c + f / (c * c) - 2 * g / c + cst)
    return math.sqrt(max(best, 0.0))


def train_run(method_name, lr, teacher, init_state, eval_x, eval_y,
              num_steps=NUM_STEPS, data_seed=1234, hyper=None, stop_at=None):
    """One training run. Every (method, lr) sees the same init and data stream.
    stop_at overrides the early-stop loss (default CONVERGED_MSE) — used by the
    early-phase tuner, which only needs the run up to its target threshold."""
    student = LinearSelfAttention(DIM)
    student.load_state_dict(copy.deepcopy(init_state))
    factory, _, _ = METHODS[method_name]
    opt = factory(student, lr, **(hyper or {}))
    gen = torch.Generator().manual_seed(data_seed)

    eval_losses, ortho_errs = [], []
    steps_to = {t: None for t in THRESHOLDS}
    status = "ok"
    for step in range(1, num_steps + 1):
        X = gen_batch(gen)
        with torch.no_grad():
            Y = teacher(X)
        try:
            opt.step(student, X, Y)
        except Exception:
            status = "error"
            break
        with torch.no_grad():
            el = mse(student(eval_x), eval_y).item()
            oe = max(ortho_error(getattr(student, n)) for n in PARAM_NAMES)
        eval_losses.append(el)
        ortho_errs.append(oe)
        if not math.isfinite(el) or el > DIVERGED_MSE:
            status = "diverged"
            break
        for t in THRESHOLDS:
            if steps_to[t] is None and el <= t:
                steps_to[t] = step
        if el <= (stop_at if stop_at is not None else CONVERGED_MSE):
            break
    return {
        "method": method_name, "lr": lr, "hyper": hyper or {}, "status": status,
        "eval_losses": eval_losses, "ortho_errs": ortho_errs,
        "steps_to": steps_to,
        "final_loss": eval_losses[-1] if eval_losses else float("inf"),
        "final_dist": identifiable_distances(student, teacher),
    }


def run_key(r):
    """Ranking: reached headline threshold fastest; tie-break on final loss."""
    s = r["steps_to"][THRESHOLDS[0]]
    return (s if s is not None else 10 ** 9, r["final_loss"])


def lr_robustness(runs, best):
    """Number of sweep lrs (at the best run's extra hypers) whose
    steps-to-headline-threshold is within 2x of the best."""
    s_best = best["steps_to"][THRESHOLDS[0]]
    if s_best is None:
        return 0
    same_hyper = [r for r in runs if r["hyper"] == best["hyper"]]
    return sum(1 for r in same_hyper
               if r["steps_to"][THRESHOLDS[0]] is not None
               and r["steps_to"][THRESHOLDS[0]] <= 2 * s_best)


def hyper_label(run):
    return "".join(f", {k}={v:g}" for k, v in run["hyper"].items())


def sweep(num_steps=NUM_STEPS, methods=None, arms=("aligned", "unaligned"), seed=42):
    teacher, inits, eval_x, eval_y = make_problem(seed)
    results = {}
    for name in (methods or METHODS):
        _, lrs, extra = METHODS[name]
        results[name] = {}
        for arm in arms:
            runs = []
            t0 = time.time()
            for hyper in (extra or [None]):
                for lr in lrs:
                    runs.append(train_run(name, float(lr), teacher, inits[arm], eval_x, eval_y,
                                          num_steps=num_steps, hyper=hyper))
            best = min(runs, key=run_key)
            n_lrs = len(lrs)
            results[name][arm] = {"runs": runs, "best": best,
                                  "lr_robustness": lr_robustness(runs, best)}
            stt = best["steps_to"][THRESHOLDS[0]]
            print(f"{name:20s} [{arm:9s}] best lr {best['lr']:.3g}{hyper_label(best)} "
                  f"| steps to 1e-12: {stt if stt is not None else '---':>4} "
                  f"| final {best['final_loss']:.2e} "
                  f"| dist {best['final_dist']:.2e} | ortho {max(best['ortho_errs']):.2e} "
                  f"| robust {results[name][arm]['lr_robustness']}/{n_lrs} | {time.time()-t0:.1f}s",
                  flush=True)
    # two-block exact solver (aligned arm not needed: polar steps cross components)
    t0 = time.time()
    tb = two_block_exact(teacher, inits["unaligned"], eval_x, eval_y)
    print(f"{'two_block_exact':20s} [solver   ] alternations: {len(tb['eval_losses'])} "
          f"| final {tb['eval_losses'][-1]:.2e} | {time.time()-t0:.1f}s", flush=True)
    results["two_block_exact"] = {"solver": {
        "eval_losses": tb["eval_losses"], "ortho_errs": tb["ortho_errs"],
        "final_dist": identifiable_distances(tb["student"], teacher)}}
    return results


def seed_check(results, seeds=(43, 44), num_steps=NUM_STEPS):
    """Re-run each method's best (lr, aligned) config on extra seeds."""
    out = {}
    for seed in seeds:
        teacher, inits, eval_x, eval_y = make_problem(seed)
        out[seed] = {}
        for name, res in results.items():
            if "aligned" not in res:
                continue
            best = res["aligned"]["best"]
            lr, hyper = best["lr"], best["hyper"]
            r = train_run(name, lr, teacher, inits["aligned"], eval_x, eval_y,
                          num_steps=num_steps, data_seed=1234 + seed, hyper=hyper)
            out[seed][name] = {"lr": lr, "hyper": hyper, "steps_to": r["steps_to"],
                               "final_loss": r["final_loss"], "status": r["status"]}
            stt = r["steps_to"][THRESHOLDS[0]]
            print(f"seed {seed} {name:20s} steps to 1e-12: {stt if stt is not None else '---':>4} "
                  f"| final {r['final_loss']:.2e}", flush=True)
    return out


# ---------------------------------------------------------------- pre-flight verification

def verify():
    """Math checks for the conventions; run before trusting any sweep."""
    print("Pre-flight verification (float64)")
    torch.manual_seed(0)
    teacher, inits, eval_x, eval_y = make_problem(0)
    student = LinearSelfAttention(DIM)
    student.load_state_dict(inits["aligned"])
    gen = torch.Generator().manual_seed(5)
    X = gen_batch(gen)
    with torch.no_grad():
        Y = teacher(X)
    loss0, grads = loss_and_grads(student, X, Y)
    W = student.W_q.detach().clone()
    G = grads["W_q"]
    Om = skew(W.T @ G)

    # 1. descent identity <G, W Omega> = ||Omega||^2
    lhs = (G * (W @ Om)).sum().item()
    rhs = (Om * Om).sum().item()
    print(f"1. descent identity: <G, W Om> = {lhs:.6e}, ||Om||^2 = {rhs:.6e} "
          f"{'OK' if abs(lhs - rhs) < 1e-12 * max(1, abs(rhs)) else 'FAIL'}")

    # 2. slope check: (L(W expm(-eta Om)) - L) / eta -> -||Om||^2 over all three params
    def loss_with(Wq):
        s2 = LinearSelfAttention(DIM)
        s2.load_state_dict(student.state_dict())
        with torch.no_grad():
            s2.W_q.copy_(Wq)
        return mse(s2(X), Y).item()
    total_sq = rhs
    for eta in [1e-4, 1e-5, 1e-6]:
        slope = (loss_with(W @ torch.linalg.matrix_exp(-eta * Om)) - loss0) / eta
        print(f"2. slope at eta={eta:.0e}: {slope:.6e} (target {-total_sq:.6e})")

    # 3. negative control: mismatched pairing W @ expm(-eta skew(G W^T)) must NOT
    #    match the descent slope (generally ascends or moves arbitrarily)
    Om_bad = skew(G @ W.T)
    eta = 1e-5
    slope_bad = (loss_with(W @ torch.linalg.matrix_exp(-eta * Om_bad)) - loss0) / eta
    print(f"3. mismatched-pairing slope: {slope_bad:.6e} vs correct {-total_sq:.6e} "
          f"{'OK (differs)' if abs(slope_bad + total_sq) > 1e-3 * abs(total_sq) else 'SUSPICIOUS'}")

    # 4. landing decomposition: descent component orthogonal to pull component for off-manifold W
    W_off = W + 0.1 * torch.randn(DIM, DIM, generator=torch.Generator().manual_seed(7))
    ip = ((skew(G @ W_off.T) @ W_off) * (W_off @ (W_off.T @ W_off - torch.eye(DIM)))).sum().item()
    print(f"4. landing orthogonal decomposition (off-manifold): {ip:.2e} "
          f"{'OK' if abs(ip) < 1e-10 else 'FAIL'}")

    # 5. orthogonality soak: 200 random multiplicative steps keep ||W^T W - I|| ~ 0
    Ws = W.clone()
    g2 = torch.Generator().manual_seed(11)
    for _ in range(200):
        R = torch.randn(DIM, DIM, generator=g2)
        Ws = Ws @ torch.linalg.matrix_exp(-0.05 * skew(R))
    print(f"5. orthogonality soak (200 expm steps): err {ortho_error(Ws):.2e} "
          f"{'OK' if ortho_error(Ws) < 1e-12 else 'FAIL'}")

    # 6. block-A unit test: with W_v = W_v* and W_k arbitrary, block A recovers A*
    s3 = LinearSelfAttention(DIM)
    s3.load_state_dict(inits["unaligned"])
    with torch.no_grad():
        s3.W_v.copy_(teacher.W_v)
        Xb = gen_batch(gen, batch_size=64)
        Yb = teacher(Xb)
        V = Xb @ s3.W_v
        C = torch.cumsum(V.unsqueeze(-1) * Xb.unsqueeze(2), dim=1)
        Phi = torch.einsum("bnd,bnef->bnedf", Xb, C).reshape(-1, DIM * DIM)
        sol = torch.linalg.lstsq(Phi, Yb.reshape(-1)).solution
        A_hat = sol.reshape(DIM, DIM)
        A_star = teacher.W_q @ teacher.W_k.T
        err = torch.linalg.norm(A_hat - A_star).item()
    print(f"6. block-A unit test: ||A_hat - A*|| = {err:.2e} {'OK' if err < 1e-9 else 'FAIL'}")

    # 7. polar sanity: polar(W) == W for orthogonal W
    perr = torch.linalg.norm(polar(W) - W).item()
    print(f"7. polar(W) = W for orthogonal W: err {perr:.2e} {'OK' if perr < 1e-12 else 'FAIL'}")

    # 8. determinant obstruction demo: dets at init
    print(f"8. det signs teacher (q,k,v): "
          f"({det_sign(teacher.W_q):+.0f},{det_sign(teacher.W_k):+.0f},{det_sign(teacher.W_v):+.0f}) | "
          f"student unaligned: ({det_sign(inits['unaligned']['W_q']):+.0f},"
          f"{det_sign(inits['unaligned']['W_k']):+.0f},{det_sign(inits['unaligned']['W_v']):+.0f}) | "
          f"student aligned: ({det_sign(inits['aligned']['W_q']):+.0f},"
          f"{det_sign(inits['aligned']['W_k']):+.0f},{det_sign(inits['aligned']['W_v']):+.0f})")


# ---------------------------------------------------------------- plots & report data

def save_outputs(results, seed_results=None):
    os.makedirs("reports", exist_ok=True)

    def best_of(name, arm="aligned"):
        return results[name][arm]["best"]

    method_names = [n for n in results if n != "two_block_exact"]

    # 1. loss curves at best lr (aligned arm)
    plt.figure(figsize=(11, 6.5))
    for name in method_names:
        best = best_of(name)
        plt.semilogy(range(1, len(best["eval_losses"]) + 1), best["eval_losses"],
                     label=f"{name} (lr={best['lr']:.3g}{hyper_label(best)})", alpha=0.85)
    tb = results["two_block_exact"]["solver"]
    plt.semilogy(range(1, len(tb["eval_losses"]) + 1), tb["eval_losses"], "k*--",
                 label="two_block_exact (per alternation)", markersize=10)
    plt.xlabel("step (alternation for two_block_exact)")
    plt.ylabel("eval MSE")
    plt.title("Best-LR eval loss per method, det-aligned init (D=32, float64)")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/ortho_loss_curves.png", dpi=150)
    plt.close()

    # 2. LR robustness: steps-to-threshold vs lr (aligned arm, best extra hypers)
    plt.figure(figsize=(11, 6.5))
    for name in method_names:
        best = best_of(name)
        runs = [r for r in results[name]["aligned"]["runs"] if r["hyper"] == best["hyper"]]
        lrs = [r["lr"] for r in runs]
        stt = [r["steps_to"][THRESHOLDS[0]] if r["steps_to"][THRESHOLDS[0]] is not None else np.nan
               for r in runs]
        plt.plot(lrs, stt, "o-", label=f"{name}{hyper_label(best)}", alpha=0.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("learning rate")
    plt.ylabel(f"steps to eval MSE <= {THRESHOLDS[0]:.0e}")
    plt.title("LR robustness, det-aligned init (missing point = never reached)")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/ortho_lr_robustness.png", dpi=150)
    plt.close()

    # 3. orthogonality drift at best lr (aligned arm)
    plt.figure(figsize=(11, 6.5))
    for name in method_names:
        best = best_of(name)
        plt.semilogy(range(1, len(best["ortho_errs"]) + 1),
                     [max(e, 1e-18) for e in best["ortho_errs"]], label=name, alpha=0.85)
    plt.xlabel("step")
    plt.ylabel("max over W of ||W^T W - I||_F")
    plt.title("Orthogonality drift at best LR (det-aligned init)")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/ortho_drift.png", dpi=150)
    plt.close()

    # 4. determinant obstruction: aligned vs unaligned best curves for det-preserving methods
    plt.figure(figsize=(11, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    shown = [n for n in method_names if n in DET_PRESERVING][:10]
    for i, name in enumerate(shown):
        for arm, ls in [("aligned", "-"), ("unaligned", "--")]:
            best = results[name][arm]["best"]
            plt.semilogy(range(1, len(best["eval_losses"]) + 1), best["eval_losses"],
                         ls, color=colors[i % 10],
                         label=f"{name} ({arm})", alpha=0.8)
    plt.xlabel("step")
    plt.ylabel("eval MSE")
    plt.title("Determinant obstruction: det-preserving methods, aligned (solid) vs unaligned (dashed) init")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/ortho_det_obstruction.png", dpi=150)
    plt.close()

    # JSON summary
    summary = {"config": {"dim": DIM, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
                          "num_steps": NUM_STEPS, "thresholds": THRESHOLDS,
                          "dtype": "float64"}}
    for name in method_names:
        summary[name] = {}
        for arm in ("aligned", "unaligned"):
            if arm not in results[name]:
                continue
            res = results[name][arm]
            best = res["best"]
            summary[name][arm] = {
                "best_lr": best["lr"],
                "best_hyper": best["hyper"],
                "steps_to": {f"{t:.0e}": best["steps_to"][t] for t in THRESHOLDS},
                "final_loss": best["final_loss"],
                "final_dist": best["final_dist"],
                "max_ortho_err": max(best["ortho_errs"]) if best["ortho_errs"] else None,
                "final_ortho_err": best["ortho_errs"][-1] if best["ortho_errs"] else None,
                "lr_robustness": res["lr_robustness"],
                "n_lrs_tried": len(res["runs"]),
                "lr_table": [
                    {"lr": r["lr"], "hyper": r["hyper"],
                     "steps_to": {f"{t:.0e}": r["steps_to"][t] for t in THRESHOLDS},
                     "final_loss": r["final_loss"], "status": r["status"]}
                    for r in res["runs"]
                ],
                "best_curve": best["eval_losses"],
                "best_ortho_curve": best["ortho_errs"],
            }
    summary["two_block_exact"] = results["two_block_exact"]["solver"]
    if seed_results:
        summary["seed_check"] = {
            str(seed): {name: v for name, v in d.items()} for seed, d in seed_results.items()}
    with open("reports/ortho_updates_results.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("Saved reports/ortho_loss_curves.png, ortho_lr_robustness.png, ortho_drift.png, "
          "ortho_det_obstruction.png, ortho_updates_results.json")


if __name__ == "__main__":
    t0 = time.time()
    if "--verify" in sys.argv:
        verify()
    elif "--smoke" in sys.argv:
        sweep(num_steps=50, arms=("aligned",))
    else:
        results = sweep()
        seed_results = seed_check(results)
        save_outputs(results, seed_results)
    print(f"Total time: {time.time() - t0:.1f}s")
