# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
# ]
# ///
"""Initial-convergence analysis: steps to the first 10x / 100x eval-loss reduction.

Pure post-processing of existing results — no training, no tuning. Reads the
best-run curves stored in reports/ortho_updates_results.json (and the two-block
solver curve in reports/two_block_results.json); the only computation is one
forward pass to record the shared init eval loss. Caveat carried into the
report: each method's curve is at the config tuned for fastest deep convergence
(steps to 1e-12), not retuned for the early phase.

Outputs: reports/ortho_initial_convergence.png, ortho_initial_convergence.json.
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ortho_updates import make_problem, mse
from linattention_solve import LinearSelfAttention
from ortho_updates import DIM


def steps_to(curve, thresh):
    for i, v in enumerate(curve, start=1):
        if v <= thresh:
            return i
    return None


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
        print(f"{name:20s} 10x: {table[name]['steps_to_10x']!s:>5} | "
              f"100x: {table[name]['steps_to_100x']!s:>5} | 1e-12: {b['steps_to']['1e-12']}")

    # two-block solver: alternations to the same factors from its own init
    tb_curve, tb_init = tb["main_run"]["after_a"], tb["main_run"]["init_loss"]
    tb_entry = {"alt_to_10x": steps_to(tb_curve, tb_init / 10),
                "alt_to_100x": steps_to(tb_curve, tb_init / 100), "init_loss": tb_init}
    print(f"two_block_exact      10x: alt {tb_entry['alt_to_10x']} | 100x: alt {tb_entry['alt_to_100x']}")

    out = {"init_loss_aligned": init_loss, "thr_10x": thr10, "thr_100x": thr100,
           "methods": table, "two_block_exact": tb_entry}
    with open("reports/ortho_initial_convergence.json", "w") as f:
        json.dump(out, f, indent=1)

    # ---- plot: early-phase curves + steps-to-100x bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    # clamp the curve panel to the second-slowest method so one straggler
    # (muon_polar, 488 steps) doesn't squeeze the early phase; the bar panel
    # still carries the full numbers
    reached = sorted(t["steps_to_100x"] for t in table.values() if t["steps_to_100x"])
    xmax = 30 + (reached[-2] if len(reached) >= 2 else reached[-1])
    for name in methods:
        curve = results[name]["aligned"]["best_curve"][:xmax]
        ax1.semilogy(range(1, len(curve) + 1), curve, label=name, alpha=0.85)
    ax1.axhline(thr100, color="k", ls="--", lw=1, label="100x reduction")
    ax1.axhline(thr10, color="k", ls=":", lw=1, label="10x reduction")
    ax1.set_xlim(0, xmax)
    ax1.set_ylim(thr100 / 30, init_loss * 2)
    ax1.set_xlabel("step")
    ax1.set_ylabel("eval MSE")
    ax1.set_title("Early phase at each method's best deep-convergence config")
    ax1.legend(fontsize=6.5, ncol=2)
    ax1.grid(True, which="both", alpha=0.3)

    order = sorted(table, key=lambda n: (table[n]["steps_to_100x"] is None,
                                         table[n]["steps_to_100x"] or 0))
    names = list(reversed(order))
    vals = [table[n]["steps_to_100x"] or 0 for n in names]
    bars = ax2.barh(names, vals, color="tab:blue", alpha=0.8)
    for bar, n in zip(bars, names):
        v = table[n]["steps_to_100x"]
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 str(v) if v else "never", va="center", fontsize=7)
    ax2.set_xlabel(f"steps to 100x eval-loss reduction (<= {thr100:.1e})")
    ax2.set_title("Steps to the first 100x")
    ax2.tick_params(axis="y", labelsize=7)
    ax2.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/ortho_initial_convergence.png", dpi=150)
    print("Saved reports/ortho_initial_convergence.png, ortho_initial_convergence.json")


if __name__ == "__main__":
    main()
