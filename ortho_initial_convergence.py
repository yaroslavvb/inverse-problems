# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
# ]
# ///
"""Initial-convergence analysis: steps to the first 10x / 100x eval-loss reduction.

Two parts:
1. Post-processing of the deep-convergence sweep: steps-to-10x/100x read off the
   best-run curves stored in reports/ortho_updates_results.json (configs tuned
   for steps-to-1e-12).
2. Early-phase RE-TUNING: each method's full lr (and hyper) grid is swept again
   on the same aligned-arm problem with the same init and data stream, but the
   best config is selected by steps-to-100x. Runs are capped at 500 steps and
   stop as soon as they cross the 100x threshold, so the sweep is cheap
   (~minutes, no deep convergence is run).

Outputs: reports/ortho_initial_convergence.png, ortho_initial_convergence.json.
"""

import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ortho_updates import (DIM, METHODS, make_problem, mse, train_run)
from linattention_solve import LinearSelfAttention

TUNE_STEPS = 500   # cap for the early-phase tuning runs


def steps_to(curve, thresh):
    for i, v in enumerate(curve, start=1):
        if v <= thresh:
            return i
    return None


def hyper_label(hyper):
    return "".join(f", {k}={v:g}" for k, v in hyper.items())


def tune_early(teacher, inits, eval_x, eval_y, thr10, thr100):
    """Re-tune every method for steps-to-100x over its full config grid.
    If the optimum lands on the top of the lr grid, extend upward by
    half-decade steps (the deep sweep never needed that, the early phase might)."""
    tuned = {}
    for name in METHODS:
        _, lrs, extra = METHODS[name]
        best = None
        t0 = time.time()

        def run_cfg(lr, hyper):
            r = train_run(name, float(lr), teacher, inits["aligned"], eval_x, eval_y,
                          num_steps=TUNE_STEPS, hyper=hyper, stop_at=thr100)
            s100 = steps_to(r["eval_losses"], thr100)
            s10 = steps_to(r["eval_losses"], thr10)
            return {"lr": float(lr), "hyper": hyper or {},
                    "steps_to_100x": s100, "steps_to_10x": s10,
                    "curve": r["eval_losses"],
                    "_key": (s100 is None, s100 if s100 is not None else 10 ** 9,
                             s10 if s10 is not None else 10 ** 9)}

        for hyper in (extra or [None]):
            for lr in lrs:
                cand = run_cfg(lr, hyper)
                if best is None or cand["_key"] < best["_key"]:
                    best = cand
        max_lr, extensions = float(max(lrs)), 0
        while best["lr"] == max_lr and extensions < 3:
            max_lr *= 10 ** 0.5
            extensions += 1
            cand = run_cfg(max_lr, best["hyper"] or None)
            if cand["_key"] < best["_key"]:
                best = cand
            else:
                break
        best.pop("_key")
        best["grid_extended"] = extensions > 0
        tuned[name] = best
        print(f"{name:20s} early-tuned lr {best['lr']:.3g}{hyper_label(best['hyper'])} "
              f"| 10x: {best['steps_to_10x']!s:>5} | 100x: {best['steps_to_100x']!s:>5} "
              f"{'(grid extended) ' if best['grid_extended'] else ''}| {time.time()-t0:.0f}s",
              flush=True)
    return tuned


def main():
    results = json.load(open("reports/ortho_updates_results.json"))
    tb = json.load(open("reports/two_block_results.json"))

    # init eval loss of the shared aligned-arm problem (one forward pass)
    teacher, inits, eval_x, eval_y = make_problem(42)
    student = LinearSelfAttention(DIM)
    student.load_state_dict(inits["aligned"])
    with torch.no_grad():
        init_loss = mse(student(eval_x), eval_y).item()
    thr10, thr100 = init_loss / 10, init_loss / 100
    print(f"aligned init eval loss {init_loss:.4e} | 10x thr {thr10:.3e} | 100x thr {thr100:.3e}")

    # ---- part 1: read early-phase crossings off the deep-tuned best curves
    methods = [k for k in results if isinstance(results[k], dict) and "aligned" in results[k]]
    table = {}
    for name in methods:
        b = results[name]["aligned"]
        curve = b["best_curve"]
        table[name] = {
            "lr": b["best_lr"], "hyper": b.get("best_hyper", {}),
            "steps_to_10x": steps_to(curve, thr10),
            "steps_to_100x": steps_to(curve, thr100),
            "steps_to_1e-12": b["steps_to"]["1e-12"],
        }
        print(f"{name:20s} deep-tuned   10x: {table[name]['steps_to_10x']!s:>5} | "
              f"100x: {table[name]['steps_to_100x']!s:>5} | 1e-12: {b['steps_to']['1e-12']}")

    # ---- part 2: re-tune for steps-to-100x
    tuned = tune_early(teacher, inits, eval_x, eval_y, thr10, thr100)

    # two-block solver: alternations to the same factors from its own init
    tb_curve, tb_init = tb["main_run"]["after_a"], tb["main_run"]["init_loss"]
    tb_entry = {"alt_to_10x": steps_to(tb_curve, tb_init / 10),
                "alt_to_100x": steps_to(tb_curve, tb_init / 100), "init_loss": tb_init}
    print(f"two_block_exact      10x: alt {tb_entry['alt_to_10x']} | 100x: alt {tb_entry['alt_to_100x']}")

    out = {"init_loss_aligned": init_loss, "thr_10x": thr10, "thr_100x": thr100,
           "tune_steps_cap": TUNE_STEPS,
           "methods": table,
           "early_tuned": {n: {k: v for k, v in t.items() if k != "curve"}
                           for n, t in tuned.items()},
           "two_block_exact": tb_entry}
    with open("reports/ortho_initial_convergence.json", "w") as f:
        json.dump(out, f, indent=1)

    # ---- plot: early-tuned curves + deep-vs-early bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    reached = sorted(t["steps_to_100x"] for t in tuned.values() if t["steps_to_100x"])
    xmax = 30 + (reached[-2] if len(reached) >= 2 else reached[-1])
    for name in methods:
        t = tuned[name]
        curve = t["curve"][:xmax]
        ax1.semilogy(range(1, len(curve) + 1), curve,
                     label=f"{name} (lr={t['lr']:.3g}{hyper_label(t['hyper'])})", alpha=0.85)
    ax1.axhline(thr100, color="k", ls="--", lw=1, label="100x reduction")
    ax1.axhline(thr10, color="k", ls=":", lw=1, label="10x reduction")
    ax1.set_xlim(0, xmax)
    ax1.set_ylim(thr100 / 30, init_loss * 2)
    ax1.set_xlabel("step")
    ax1.set_ylabel("eval MSE")
    ax1.set_title("Early phase at each method's early-tuned config (curves stop at crossing)")
    ax1.legend(fontsize=6.5, ncol=2)
    ax1.grid(True, which="both", alpha=0.3)

    order = sorted(methods, key=lambda n: (tuned[n]["steps_to_100x"] is None,
                                           tuned[n]["steps_to_100x"] or 0))
    names = list(reversed(order))
    y = np.arange(len(names))
    deep_vals = [table[n]["steps_to_100x"] or 0 for n in names]
    early_vals = [tuned[n]["steps_to_100x"] or 0 for n in names]
    ax2.barh(y + 0.2, deep_vals, height=0.38, color="tab:gray", alpha=0.65,
             label="deep-tuned config (steps to 1e-12)")
    ax2.barh(y - 0.2, early_vals, height=0.38, color="tab:blue", alpha=0.9,
             label="re-tuned for 100x")
    for yi, n in zip(y, names):
        d, e = table[n]["steps_to_100x"], tuned[n]["steps_to_100x"]
        ax2.text((d or 0) + 2, yi + 0.2, str(d) if d else "never", va="center", fontsize=6.5)
        ax2.text((e or 0) + 2, yi - 0.2, str(e) if e else "never", va="center", fontsize=6.5,
                 fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=7)
    ax2.set_xlabel(f"steps to 100x eval-loss reduction (<= {thr100:.1e})")
    ax2.set_title("Steps to the first 100x: config choice matters")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/ortho_initial_convergence.png", dpi=150)
    print("Saved reports/ortho_initial_convergence.png, ortho_initial_convergence.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {time.time() - t0:.1f}s")
