# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
#   "transformers",
# ]
# ///
"""Learning without Forgetting on REAL pre-trained attention circuits.

Instead of random orthogonal weights, the teacher and student are actual
attention heads of EleutherAI/pythia-14m (d_model=128, head_dim=32):

  teacher  A* = W_Q W_K^T / sqrt(d_h)   (QK circuit, 128x128, rank 32)
           W_v* = W_V W_O               (OV circuit, 128x128, rank 32)
  student  initialized from a DIFFERENT pre-trained head ("that particular
           value"), so the streaming task reads: adapt one real head toward
           another head's function from tiny batches without forgetting.

The weights are transplanted into the linear-attention lab of forgetting_lab.py
(no softmax, no rotary; we study the weight geometry, not pythia's function).
The factored Muons get the student head's REAL tall factors (128x32, conditions
~68 and ~28, mutually unbalanced 4.9x) — the regime compositional Muon targets,
now without synthetic gauges. The sqrt(d_h) score
scaling is split evenly into the factors (W_q0 = W_Q/d_h^(1/4), W_k0 likewise)
so the factored init matches its own head's circuit.

Methods: the forgetting_lab suite minus null_proj (its basis ops scale with
D^2 = 16384 and break the <1s budget; it was also the structural loser at
D=32). EWC uses a memory-light Fisher-diagonal einsum (forming Phi explicitly
at D=128 costs 268 MB/round). Seeds vary the data stream only — the pre-trained
weights are fixed.

Outputs: reports/forgetting_pretrained_curves.png, _strength.png,
forgetting_pretrained_results.json.
"""

import os
import sys
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import forgetting_lab as fl
from forgetting_lab import (forward, mse, msign_safe, geo_mean, hyper_str)
from ortho_updates import gen_batch

D_MODEL = 128
HEAD_DIM = 32
TEACHER_HEAD = (1, 0)   # (layer, head)
STUDENT_HEAD = (4, 2)
ROUNDS = 40
BATCH = 2
INNER_STEPS = 20
SEEDS = (0, 1, 2)
CIRCUITS_CACHE = "reports/pythia_circuits.pt"


def extract_circuits():
    """Per-head QK and OV circuits of pythia-14m, in math convention (y = x W).

    Both heads' circuits are normalized by the TEACHER's spectral norms
    (one scalar per circuit type, split as a square root into each factor).
    This is a units choice only: it preserves every geometric property of the
    pre-trained weights — rank, singular-spectrum shapes, factor imbalance,
    the student/teacher relative scale — while making losses and hyper grids
    comparable to the random-weight study (where ||A*||_2 = ||W_v*||_2 = 1).
    Raw circuits sit at ~0.1 spectral norm with ~1e-5 outputs, which mis-scales
    every method's knob grid."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m", dtype=torch.float64)
    out = {}
    for tag, (layer, h) in (("teacher", TEACHER_HEAD), ("student", STUDENT_HEAD)):
        attn = m.gpt_neox.layers[layer].attention
        w = attn.query_key_value.weight.view(4, 3 * HEAD_DIM, D_MODEL).detach()
        Wq = w[h, :HEAD_DIM].T                                   # (D, hd)
        Wk = w[h, HEAD_DIM:2 * HEAD_DIM].T
        Wv = w[h, 2 * HEAD_DIM:].T
        Wo = attn.dense.weight[:, h * HEAD_DIM:(h + 1) * HEAD_DIM].detach().T   # (hd, D)
        out[tag] = {
            "A": (Wq @ Wk.T / HEAD_DIM ** 0.5).contiguous(),     # QK circuit
            "OV": (Wv @ Wo).contiguous(),                        # OV circuit
            "Wq": (Wq / HEAD_DIM ** 0.25).contiguous(),          # scale split so
            "Wk": (Wk / HEAD_DIM ** 0.25).contiguous(),          # Wq Wk^T = A
        }
    sA = torch.linalg.matrix_norm(out["teacher"]["A"], ord=2)
    sV = torch.linalg.matrix_norm(out["teacher"]["OV"], ord=2)
    for tag in out:
        out[tag]["A"] = out[tag]["A"] / sA
        out[tag]["OV"] = out[tag]["OV"] / sV
        out[tag]["Wq"] = out[tag]["Wq"] / sA ** 0.5
        out[tag]["Wk"] = out[tag]["Wk"] / sA ** 0.5
    return out


def load_circuits():
    if os.path.exists(CIRCUITS_CACHE):
        return torch.load(CIRCUITS_CACHE)
    c = extract_circuits()
    os.makedirs("reports", exist_ok=True)
    torch.save(c, CIRCUITS_CACHE)
    return c


def round_ewc_p(state, Xb, Yb, mem, hyper):
    """forgetting_lab.round_ewc with a memory-light GN diagonal: at D=128 the
    explicit Phi is (B*N*D) x D^2 = 268 MB/round; the diagonal only needs
    sum_rows Phi^2 = einsum(X^2, C^2)."""
    lam = hyper["lam"]
    D = fl.DIM
    if "F_A" not in mem:
        mem["F_A"] = torch.zeros(D, D)
        mem["F_W"] = torch.zeros(D, 1)
    anchA, anchW = state["A"].detach().clone(), state["Wv"].detach().clone()
    F_A, F_W = mem["F_A"], mem["F_W"]

    def loss_fn():
        return (mse(forward(Xb, state["A"], state["Wv"]), Yb)
                + lam * ((F_A * (state["A"] - anchA) ** 2).sum()
                         + (F_W * (state["Wv"] - anchW) ** 2).sum()))
    fl._adam_fit(state, INNER_STEPS, BASE_LR_P, loss_fn)

    with torch.no_grad():
        A, Wv = state["A"], state["Wv"]
        M = Yb.numel()
        scores = torch.tril(torch.einsum("bnd,bmd->bnm", Xb @ A, Xb))
        feats_W = torch.einsum("bnm,bmd->bnd", scores, Xb).reshape(-1, D)
        V = Xb @ Wv
        C = torch.cumsum(V.unsqueeze(-1) * Xb.unsqueeze(2), dim=1)
        mem["F_A"] += (2.0 / M) * torch.einsum("bnd,bnef->df", Xb ** 2, C ** 2)
        mem["F_W"] += (2.0 / M) * (feats_W ** 2).sum(0)[:, None]


# Inner Adam lr for the constrained methods: circuit entries are ~8x smaller
# than the random lab's (unit spectral norm spread over rank 32 of 128 dims),
# so the random lab's 1e-2 damages the weights faster than the penalties can
# protect them; 1e-3 behaves like 1e-2 did there.
BASE_LR_P = 1e-3

# method name -> (round fn, hyper grid); grids centered by a coarse probe
METHODS_P = {
    "naive":       (fl.round_naive,       [{"lr": v} for v in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)]),
    "frob_anchor": (fl.round_frob_anchor, [{"lam": v} for v in (0.1, 0.3, 1.0, 3.0, 10.0)]),
    "func_anchor": (fl.round_func_anchor, [{"lam": v} for v in (0.1, 1.0, 10.0, 100.0)]),
    "spec_clip":   (fl.round_spec_clip,   [{"tau": v} for v in (0.01, 0.03, 0.1, 0.3, 1.0)]),
    "muon":        (fl.round_muon,        [{"eta": v} for v in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)]),
    "comp_muon":   (fl.round_comp_muon,   [{"eta": v} for v in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)]),
    "factored_muon": (fl.round_factored_muon, [{"eta": v} for v in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)]),
    "replay":      (fl.round_replay,      [{"cap": v} for v in (8, 32, 1000)]),
    "ewc":         (round_ewc_p,          [{"lam": v} for v in (0.01, 0.03, 0.1, 0.3, 1.0)]),
}


def run_stream_p(method, hyper, seed, circuits):
    torch.set_num_threads(1)
    fl.DIM = D_MODEL                  # round fns read forgetting_lab.DIM at call time
    fl.BASE_LR = BASE_LR_P            # likewise the constrained methods' inner lr
    t0 = time.time()
    A_star = circuits["teacher"]["A"]
    W_star = circuits["teacher"]["OV"]
    state = {"A": circuits["student"]["A"].clone().requires_grad_(True),
             "Wv": circuits["student"]["OV"].clone().requires_grad_(True)}
    mem = {"gen": torch.Generator().manual_seed(500 + seed)}
    if method in ("comp_muon", "factored_muon"):
        # real tall factors of the student head (skips _factored_init's gauge)
        mem["Wq"] = circuits["student"]["Wq"].clone().requires_grad_(True)
        mem["Wk"] = circuits["student"]["Wk"].clone().requires_grad_(True)
    round_fn, _ = METHODS_P[method]

    gen_eval = torch.Generator().manual_seed(99)
    eval_x = gen_batch(gen_eval, batch_size=128, num_rows=fl.SEQ_LEN, dim=D_MODEL)
    with torch.no_grad():
        eval_y = forward(eval_x, A_star, W_star)
    gen_stream = torch.Generator().manual_seed(1000 + seed)

    pastX, pastY = [], []
    eval_curve, past_curve, fit_curve = [], [], []
    for t in range(ROUNDS):
        Xb = gen_batch(gen_stream, batch_size=BATCH, num_rows=fl.SEQ_LEN, dim=D_MODEL)
        with torch.no_grad():
            Yb = forward(Xb, A_star, W_star)
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
        if not math.isfinite(el):
            pad = ROUNDS - t - 1
            eval_curve += [float("inf")] * pad
            past_curve += [float("inf")] * pad
            fit_curve += [float("inf")] * pad
            break

    finite_past = [x for x in past_curve if math.isfinite(x)]
    late_past = [x for x in past_curve[-10:] if math.isfinite(x)]
    return {
        "method": method, "hyper": hyper, "seed": seed,
        "eval_curve": eval_curve, "past_curve": past_curve, "fit_curve": fit_curve,
        "final_eval": eval_curve[-1],
        "mean_past": float(np.mean(finite_past)) if finite_past else float("inf"),
        "late_past": float(np.mean(late_past)) if late_past else float("inf"),
        "mean_fit": float(np.mean([x for x in fit_curve if math.isfinite(x)] or [float("inf")])),
        "seconds": time.time() - t0,
    }


def _worker(args):
    method, hyper, seed = args
    circuits = torch.load(CIRCUITS_CACHE)
    return run_stream_p(method, hyper, seed, circuits)


def verify():
    print("Pre-trained-circuit checks (pythia-14m, float64)")
    c = load_circuits()
    fl.DIM = D_MODEL
    A, OV = c["teacher"]["A"], c["teacher"]["OV"]
    Wq, Wk = c["student"]["Wq"], c["student"]["Wk"]
    sq, sk = torch.linalg.svdvals(Wq), torch.linalg.svdvals(Wk)
    print(f"1. teacher circuits: ||A*||_2 {torch.linalg.matrix_norm(A, ord=2):.3f} "
          f"(rank {torch.linalg.matrix_rank(A).item()}), ||OV*||_2 "
          f"{torch.linalg.matrix_norm(OV, ord=2):.3f} (rank {torch.linalg.matrix_rank(OV).item()})")
    print(f"2. student factors: ||Wq||_2 {sq[0]:.3f} cond {sq[0]/sq[-1]:.1f} | "
          f"||Wk||_2 {sk[0]:.3f} cond {sk[0]/sk[-1]:.1f} "
          f"| factor imbalance ||Wq||/||Wk|| = {sq[0]/sk[0]:.2f}")
    err = (Wq @ Wk.T - c["student"]["A"]).abs().max().item()
    print(f"3. factored init consistency: ||Wq0 Wk0^T - A0||_max = {err:.2e} "
          f"{'OK' if err < 1e-12 else 'FAIL'}")

    # composed-budget check with REAL factors (analogue of forgetting_lab check 6)
    eta = 3e-4
    realized = {}
    for name in ("comp_muon", "factored_muon"):
        state = {"A": c["student"]["A"].clone().requires_grad_(True),
                 "Wv": c["student"]["OV"].clone().requires_grad_(True)}
        mem = {"Wq": c["student"]["Wq"].clone().requires_grad_(True),
               "Wk": c["student"]["Wk"].clone().requires_grad_(True)}
        gen = torch.Generator().manual_seed(5)
        Xb = gen_batch(gen, batch_size=BATCH, num_rows=fl.SEQ_LEN, dim=D_MODEL)
        with torch.no_grad():
            Yb = forward(Xb, c["teacher"]["A"], c["teacher"]["OV"])
            A0 = state["A"].detach().clone()
        METHODS_P[name][0](state, Xb, Yb, mem, {"eta": eta})
        realized[name] = torch.linalg.matrix_norm(state["A"].detach() - A0, ord=2).item()
    budget = INNER_STEPS * eta
    print(f"4. composed budget K*eta = {budget:.4f} | realized ||D(WqWk^T)||_2: "
          f"comp_muon {realized['comp_muon']:.4f}, factored_muon {realized['factored_muon']:.4f} "
          f"(overshoot {realized['factored_muon']/budget:.1f}x) "
          f"{'OK' if realized['comp_muon'] <= budget * 1.25 < realized['factored_muon'] else 'CHECK'}")

    # init loss scale
    gen = torch.Generator().manual_seed(99)
    ex = gen_batch(gen, batch_size=128, num_rows=fl.SEQ_LEN, dim=D_MODEL)
    with torch.no_grad():
        ey = forward(ex, A, OV)
        il = mse(forward(ex, c["student"]["A"], c["student"]["OV"]), ey).item()
        print(f"5. init eval loss (student head vs teacher head): {il:.3e} | "
              f"teacher output RMS {ey.pow(2).mean().sqrt():.3e}")
    print(f"6. equations/round B*N*D = {BATCH * fl.SEQ_LEN * D_MODEL} vs params 2*D^2 = "
          f"{2 * D_MODEL ** 2} -> {2 * D_MODEL ** 2 / (BATCH * fl.SEQ_LEN * D_MODEL):.0f}x underdetermined")


def sweep():
    # torch.set_num_threads(1) does not govern macOS Accelerate's BLAS pool;
    # without these, the eigh-heavy muon runs at D=128 strangle each other
    # (measured: 541s/run under 16-way contention vs ~1s alone). Spawned
    # workers inherit the environment.
    for var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    load_circuits()
    tasks = [(m, h, s) for m, (_, grid) in METHODS_P.items() for h in grid for s in SEEDS]
    workers = max(1, (os.cpu_count() or 8) - 2)
    print(f"{len(tasks)} runs on {workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        runs = list(ex.map(_worker, tasks))
    secs = sorted(r["seconds"] for r in runs)
    print(f"sweep wall-clock {time.time()-t0:.1f}s | median run {secs[len(secs)//2]:.2f}s | "
          f"warm max {secs[-workers-1] if len(secs) > workers else secs[-1]:.2f}s "
          f"(absolute max {secs[-1]:.2f}s incl. warmup)")

    results = {}
    for m in METHODS_P:
        per_hyper = {}
        for r in [r for r in runs if r["method"] == m]:
            per_hyper.setdefault(json.dumps(r["hyper"]), []).append(r)
        configs = []
        for hjson, rs in per_hyper.items():
            configs.append({
                "hyper": json.loads(hjson),
                "final_eval_geo": geo_mean([r["final_eval"] for r in rs]),
                "final_eval_per_seed": [r["final_eval"] for r in rs],
                "late_past_geo": geo_mean([r["late_past"] for r in rs]),
                "mean_fit_geo": geo_mean([r["mean_fit"] for r in rs]),
                "eval_curves": [r["eval_curve"] for r in rs],
                "seconds": [r["seconds"] for r in rs],
            })
        configs.sort(key=lambda c: c["final_eval_geo"])
        results[m] = {"configs": configs, "best": configs[0]}
        b = configs[0]
        print(f"{m:14s} best {hyper_str(b['hyper']):14s} | final eval {b['final_eval_geo']:.2e} "
              f"| late past {b['late_past_geo']:.2e} | mean fit {b['mean_fit_geo']:.2e}")
    return results


def save_outputs(results):
    xs = np.arange(1, ROUNDS + 1)

    def geo_curve(curves):
        a = np.array(curves, dtype=float)
        a = np.nan_to_num(a, nan=np.nan, posinf=1e9)
        a = np.maximum(a, 1e-300)
        with np.errstate(all="ignore"):
            return np.exp(np.nanmean(np.log(a), axis=0))

    plt.figure(figsize=(10, 6))
    for m, res in results.items():
        b = res["best"]
        plt.semilogy(xs, geo_curve(b["eval_curves"]),
                     label=f"{m} ({hyper_str(b['hyper'])})", alpha=0.9)
    plt.xlabel("round (2 fresh sequences per round)")
    plt.ylabel("eval MSE (held-out)")
    plt.title("Streaming adaptation between two real pythia-14m heads (best config per method)")
    plt.legend(fontsize=7.5)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/forgetting_pretrained_curves.png", dpi=150)
    plt.close()

    names = list(METHODS_P)
    ncols = math.ceil(len(names) / 2)
    fig, axes = plt.subplots(2, ncols, figsize=(13, 6))
    for ax, m in zip(axes.flat, names):
        cfgs = sorted(results[m]["configs"], key=lambda c: list(c["hyper"].values())[0])
        hv = [list(c["hyper"].values())[0] for c in cfgs]
        ax.plot(hv, [c["final_eval_geo"] for c in cfgs], "o-", label="final eval")
        ax.plot(hv, [c["late_past_geo"] for c in cfgs], "s--", alpha=0.7, label="late past")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(m, fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Pre-trained circuits: knob strength vs overall error", fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/forgetting_pretrained_strength.png", dpi=150)
    plt.close()

    def jsonable(obj):
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [jsonable(v) for v in obj]
        return obj

    summary = {"config": {"model": "EleutherAI/pythia-14m", "d_model": D_MODEL,
                          "head_dim": HEAD_DIM, "teacher_head": list(TEACHER_HEAD),
                          "student_head": list(STUDENT_HEAD), "rounds": ROUNDS,
                          "batch": BATCH, "inner_steps": INNER_STEPS,
                          "seeds": list(SEEDS)}}
    for m, res in results.items():
        summary[m] = {"best_hyper": res["best"]["hyper"], "configs": res["configs"]}
    with open("reports/forgetting_pretrained_results.json", "w") as f:
        json.dump(jsonable(summary), f, indent=1)
    print("Saved reports/forgetting_pretrained_curves.png, forgetting_pretrained_strength.png, "
          "forgetting_pretrained_results.json")


if __name__ == "__main__":
    t0 = time.time()
    if "--verify" in sys.argv:
        verify()
    elif "--smoke" in sys.argv:
        circuits = load_circuits()
        for m in METHODS_P:
            grid = METHODS_P[m][1]
            r = run_stream_p(m, grid[len(grid) // 2], 0, circuits)
            print(f"{m:14s} final eval {r['final_eval']:.2e} | late past {r['late_past']:.2e} "
                  f"| fit {r['mean_fit']:.2e} | {r['seconds']:.2f}s")
    else:
        save_outputs(sweep())
    print(f"Total time: {time.time() - t0:.1f}s")
