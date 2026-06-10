
# Experiment 03jun26: backprop alternative on an easy inverse problem

<img width="40%" alt="Screenshot 2026-06-03 at 4 27 15 PM" src="https://github.com/user-attachments/assets/cc4ed827-7724-4a12-85d6-b7960a6e1ee1" />

```
uv run altprop_linattention.py
uv run linattention_visualize.py
```


## Easy problem

1. stripped down [Linear Transformer](https://manifestai.com/blogposts/faster-after-all/) (drop the softplus)
2. teacher is initialized with small rotations (Pi/10)
3. student is initialized with identity matrices
4. Initialize LR using line search, apply adaptive LR tuning at each step


## Observations
[from report](https://yaroslavvb.github.io/inverse-problems/reports/linattention_report.html)

- Alternative allows using much larger learning rate and converges faster

<img src="images/image3.png" width="20%">

- Alternative trains last layer much more than the first layer
- regular backprop trains first layer more than last layer

<img src="images/image1.png" width="20%"> <img src="images/image2.png" width="20%">

- automatic tuner results in decreasing learning rate schedule for alternative propagation whereas for classic backprop it is constant 

*Initial prototyping in [colab](https://colab.research.google.com/drive/1t3YD6hQsBcTwnaVPKxgVoMRo-idLjze3#scrollTo=rg9J-sPFR3Gd).*

# Experiment 09jun26: orthogonality-exploiting updates

Single-layer linear attention student/teacher with random **orthogonal** W_q, W_k, W_v
([linattention_solve.py](linattention_solve.py)). Since the solution lies on O(32)^3,
compare updates that exploit that structure (multiplicative expm/Cayley rotations, so(n)-Adam,
retractions, landing, Procrustes/two-block solvers) against Euclidean SGD/Adam.

```
uv run linattention_solve.py        # simple single-layer baseline + loss plot
uv run ortho_updates.py --verify    # pre-flight math checks
uv run ortho_updates.py             # full comparison sweep
```

## Observations
[from report](https://yaroslavvb.github.io/inverse-problems/reports/ortho_updates_report.html)

- Exploiting **bilinearity** beats exploiting the manifold: a two-block Procrustes/least-squares
  solver converges in 10 alternations; closed-form W_v + rotation steps reaches 1e-12 in ~121 steps
  vs 338 for Adam
- Generic manifold methods (expm/Cayley/QR/polar, ~246 steps) beat Adam but only match well-tuned plain SGD (221),
  and deliver an exactly orthogonal solution (~1e-13); the rotation-angle-parameterized optimizers
  (so(n)-Adam, trivialization, clipped momentum) are slower (~290-314) but tolerate a ~10x wider learning-rate window
- **Determinant obstruction**: det-preserving updates can't leave a connected component of O(n) — 3/4 of random
  inits floor at ~5e-4 unless the init's det signs are matched to the teacher (one column flip)
- All converged methods recover the teacher up to the model's gauge symmetries; Euclidean ones end 0.1-1.0
  off-manifold (transient drift up to ~4.5) by absorbing the continuous scale gauge
