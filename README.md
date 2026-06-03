# inverse-problems

Backprop vs **altprop** on an easy inverse problem.

Generate `(x, y)` pairs from a fixed "teacher" network `f(x)`, then try to recover that
function from `(x, y)`. The question: **can we get a much better update if we specialize
the update to the structure of `f`?**

**Altprop** uses a linear approximation to take into account the update to layer `n+1`
when computing the update to layer `n` (rather than treating the upstream layers as
frozen, as standard backprop does). See the
[reference colab](https://colab.research.google.com/drive/1t3YD6hQsBcTwnaVPKxgVoMRo-idLjze3#scrollTo=rg9J-sPFR3Gd).

## Experiment

A minimal inverse problem, kept easy enough that the backprop-vs-altprop difference is
visible cleanly.

### Setup

1. **Stripped-down linear Transformer** — drop the softplus (linear attention), `NUM_LAYERS = 3`.
2. **Teacher** is initialized with small rotations (`theta = pi/10`, via `generalized_rotation`).
3. **Student** is initialized with the identity matrix.
4. **Learning rate**: use a learning-rate search for the starting LR, plus adaptive
   LR tuning at each step.

### Run

```bash
uv run linattention_solve.py      # train student to recover the teacher (backprop vs altprop)
uv run linattention_visualize.py  # generate plots + report into reports/
```

Both scripts declare their dependencies inline (PEP 723), so `uv run` handles the
environment automatically.

## Observations

- **Altprop allows a much larger learning rate and converges faster.**
- **Altprop and backprop redistribute updates in opposite directions across layers:**
  altprop changes the *last* layer much more than the first layer, while regular
  backprop does the opposite — it changes the *first* layer much more than the last.

### Modified-attention follow-ups

- Learning rate is important; updating backprop to account for the upstream update
  (as altprop does) lets you use a much larger learning rate.

## Outputs

`linattention_visualize.py` writes to [`reports/`](reports/):

- `linattention_report.html` — combined report
- `reconstruction_error.png` — student → teacher recovery error over training
- `initial_lr_search.png`, `dynamic_lr_over_time.png` — LR search and adaptive schedule
- `target_angles_over_time.png` — recovered rotation angles vs. teacher
- `weight_changes_L*_W_{q,k,v}.png` — per-layer weight movement (shows the
  first-layer vs last-layer asymmetry between backprop and altprop)
