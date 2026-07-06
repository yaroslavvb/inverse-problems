# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
#   "transformers",
#   "datasets",
# ]
# ///
"""Does the Zipf/associative-memory picture behind Muon's advantage hold for the
QK circuit of real self-attention, or only for the FFN?

Motivated by Kim, Nichani, Wu, Bietti, Lee (arXiv:2603.26554), which shows Muon
beats SGD at associative-memory learning precisely when the stored associations
have power-law frequencies p_i ~ i^-alpha: the gradient is then a frequency-
weighted sum of rank-one associations G ~ sum_i p_i u_i v_i^T whose singular
spectrum inherits the power law, and Muon's polar map amplifies the bulk (rare
facts). Henry's question: this picture is natural for the FFN (key-value
memories over Zipf-distributed tokens/concepts) — does it also hold for the
QK circuit?

We measure the Muon-relevant quantities on a PRE-TRAINED transformer (GPT-2
124M — chosen because it has no rotary embeddings, so each head's QK circuit is
exactly one bilinear form A_h = W_Q^h W_K^hT / sqrt(d_h)) over real text
(WikiText-2):

  1. GRADIENT SPECTRA: LM-loss gradients accumulated over the corpus for each
     circuit's parameter matrices (W_Q, W_K, W_V, W_O, W_in, W_out), plus the
     exact per-head associative-memory gradient dL/dA_h — recovered from the
     attention-probability gradients via manual softmax backward,
        dL/dS = P * (dL/dP - rowsum(dL/dP * P)),   dL/dA_h = X^T (dL/dS) X / sqrt(d_h),
     self-checked against autograd through dL/dW_Q^h = (dL/dA_h) W_K^h.
     Measured at the TRAINED weights and at a RE-INITIALIZED copy (the paper's
     one-step-from-init setting).
  2. USAGE FREQUENCIES ("how often is each stored association exercised"):
     FFN: E[a_j^2] per hidden neuron (Geva et al.'s key-value memories);
     QK: score energy through each singular direction of the trained A_h,
         usage_k = sigma_k^2 E[(u_k.x)^2] E[(v_k.x)^2] over token representations.
  3. INPUT COVARIANCE SPECTRA at each circuit's input (the paper's Fig. 6
     anisotropy axis, which counteracts Muon), and the corpus unigram Zipf
     curve as reference.

Power-law tails are quantified by the log-log slope of sigma_k (or usage_k)
vs rank k over FIT_RANGE, normalized by sigma_1.

Outputs: reports/zipf_qk_spectra.png, zipf_qk_usage.png, zipf_qk_results.json.
"""

import os
import sys
import json
import math
import time

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "gpt2"
SEQ_LEN = 256
BATCH = 4
N_BATCHES = 48            # ~50k tokens for gradient accumulation
HEAD_LAYERS = (0, 5, 11)  # layers for per-head dL/dA and usage stats
FIT_RANGE = (8, 256)      # rank window for log-log slope fits
DEVICE = ("mps" if torch.backends.mps.is_available() else "cpu")


def load_text_batches():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = (ids.shape[0] // (BATCH * SEQ_LEN))
    batches = ids[: n * BATCH * SEQ_LEN].view(n, BATCH, SEQ_LEN)
    return batches, ids, tok


def powerlaw_slope(vals, lo=FIT_RANGE[0], hi=FIT_RANGE[1]):
    """OLS slope of log(vals_k) vs log(k) over ranks [lo, hi] (1-indexed)."""
    v = np.asarray(vals, dtype=float)
    hi = min(hi, len(v))
    if hi - lo < 8:
        return float("nan")
    k = np.arange(lo, hi + 1)
    y = v[lo - 1:hi]
    mask = y > 0
    if mask.sum() < 8:
        return float("nan")
    return float(np.polyfit(np.log(k[mask]), np.log(y[mask]), 1)[0])


class Probe:
    """Accumulates gradient matrices, activation stats, and attention-score
    gradients for one model over a stream of batches."""

    def __init__(self, model):
        self.model = model
        cfg = model.config
        self.L, self.H = cfg.n_layer, cfg.n_head
        self.d, self.dh = cfg.n_embd, cfg.n_embd // cfg.n_head
        # accumulated parameter grads per layer
        self.G = {}          # (layer, name) -> tensor
        self.G_A = {}        # (layer, head) -> dL/dA_h, only for HEAD_LAYERS
        # structureless null baselines (same error signal, shuffled inputs):
        #   permT: fresh time-permutation of X per side (destroys content+position)
        #   permB: X shuffled across batch rows at fixed positions (destroys content only)
        self.G_A_null = {"permT": {}, "permB": {}}
        self.G_mlp_null = {"permT": {}, "permB": {}}   # (layer,) -> null of dL/dW_in
        # activation second moments, PER LAYER (uncentered: E[x x^T])
        self.cov_attn = {li: torch.zeros(self.d, self.d) for li in HEAD_LAYERS}
        self.cov_mlp = {li: torch.zeros(self.d, self.d) for li in HEAD_LAYERS}
        self.neuron_energy = {}   # layer -> E[a^2] per FFN hidden unit
        self.x_attn = {}          # layer -> stashed ln_1 outputs (per batch)
        self.x_mlp = {}           # layer -> stashed ln_2 outputs (per batch)
        self.h_mlp = {}           # layer -> stashed c_fc outputs (per batch)
        self.attn_probs = {}      # layer -> stashed attention probs (per batch)
        self.n_tokens = 0
        self.gen = torch.Generator().manual_seed(777)   # permutation RNG
        self._hooks = []
        self._install_hooks()

    def _install_hooks(self):
        for li, block in enumerate(self.model.transformer.h):
            def stash_x(mod, inp, out, li=li):
                if out.requires_grad and li in HEAD_LAYERS:
                    self.x_attn[li] = out
                return out
            self._hooks.append(block.ln_1.register_forward_hook(stash_x))

            def stash_cov(mod, inp, out, li=li, block=block):
                if li in HEAD_LAYERS:
                    with torch.no_grad():
                        x = out.detach().reshape(-1, out.shape[-1])
                        tgt = self.cov_attn if mod is block.ln_1 else self.cov_mlp
                        tgt[li].add_((x.T @ x).cpu())
                    if mod is block.ln_2:
                        self.x_mlp[li] = out.detach()
                return out
            self._hooks.append(block.ln_1.register_forward_hook(stash_cov))
            self._hooks.append(block.ln_2.register_forward_hook(stash_cov))

            def stash_hidden(mod, inp, out, li=li):
                if li in HEAD_LAYERS and out.requires_grad:
                    out.retain_grad()
                    self.h_mlp[li] = out
                return out
            self._hooks.append(block.mlp.c_fc.register_forward_hook(stash_hidden))

            def stash_act(mod, inp, out, li=li):
                with torch.no_grad():
                    a2 = (out.detach() ** 2).reshape(-1, out.shape[-1]).mean(0).cpu()
                    if li in self.neuron_energy:
                        self.neuron_energy[li] += a2
                    else:
                        self.neuron_energy[li] = a2.clone()
                return out
            self._hooks.append(block.mlp.act.register_forward_hook(stash_act))

    def step(self, input_ids):
        """One forward+backward; accumulate everything."""
        model = self.model
        model.zero_grad(set_to_none=True)
        self.x_attn.clear()
        self.x_mlp.clear()
        self.h_mlp.clear()
        self.attn_probs.clear()
        out = model(input_ids, labels=input_ids, output_attentions=True)
        # retain grads on attention probs of the selected layers
        for li in HEAD_LAYERS:
            p = out.attentions[li]
            p.retain_grad()
            self.attn_probs[li] = p
        out.loss.backward()

        with torch.no_grad():
            ntok = input_ids.numel()
            self.n_tokens += ntok
            for li, block in enumerate(self.model.transformer.h):
                grabs = {
                    "c_attn": block.attn.c_attn.weight.grad,     # (d, 3d) cols [Q|K|V]
                    "attn_out": block.attn.c_proj.weight.grad,   # (d, d)  = W_O
                    "mlp_in": block.mlp.c_fc.weight.grad,        # (d, 4d) = W_in
                    "mlp_out": block.mlp.c_proj.weight.grad,     # (4d, d) = W_out
                }
                for name, g in grabs.items():
                    key = (li, name)
                    g = g.detach().cpu()
                    if key in self.G:
                        self.G[key] += g
                    else:
                        self.G[key] = g.clone()

            # exact per-head associative-memory gradient dL/dA_h, plus nulls
            def acc(store, key, val):
                val = val.cpu()
                if key in store:
                    store[key] += val
                else:
                    store[key] = val

            def perms_of(X):
                """(permT_left, permT_right, permB_left, permB_right) shuffles of X."""
                B, T, _ = X.shape
                pt1 = torch.stack([torch.randperm(T, generator=self.gen) for _ in range(B)])
                pt2 = torch.stack([torch.randperm(T, generator=self.gen) for _ in range(B)])
                XT1 = torch.take_along_dim(X, pt1.unsqueeze(-1).to(X.device), dim=1)
                XT2 = torch.take_along_dim(X, pt2.unsqueeze(-1).to(X.device), dim=1)
                XB1 = X[torch.randperm(B, generator=self.gen)]
                XB2 = X[torch.randperm(B, generator=self.gen)]
                return XT1, XT2, XB1, XB2

            for li in HEAD_LAYERS:
                P = self.attn_probs[li]
                Pbar = P.grad
                X = self.x_attn[li].detach()
                XT1, XT2, XB1, XB2 = perms_of(X)
                Sbar = P.detach() * (Pbar - (Pbar * P.detach()).sum(-1, keepdim=True))
                for h in range(self.H):
                    S = Sbar[:, h]
                    acc(self.G_A, (li, h),
                        torch.einsum("bti,btj->ij", X, torch.bmm(S, X)) / math.sqrt(self.dh))
                    acc(self.G_A_null["permT"], (li, h),
                        torch.einsum("bti,btj->ij", XT1, torch.bmm(S, XT2)) / math.sqrt(self.dh))
                    acc(self.G_A_null["permB"], (li, h),
                        torch.einsum("bti,btj->ij", XB1, torch.bmm(S, XB2)) / math.sqrt(self.dh))

            # MLP W_in gradient nulls: dL/dW_in = X2^T Delta with shuffled X2
            for li in HEAD_LAYERS:
                X2 = self.x_mlp[li]
                Delta = self.h_mlp[li].grad
                XT1, _, XB1, _ = perms_of(X2)
                acc(self.G_mlp_null["permT"], (li,),
                    torch.einsum("btd,bth->dh", XT1, Delta))
                acc(self.G_mlp_null["permB"], (li,),
                    torch.einsum("btd,bth->dh", XB1, Delta))

    def self_check(self):
        """dL/dW_Q^h recomputed from dL/dA_h must match autograd's c_attn slice."""
        li = HEAD_LAYERS[0]
        block = self.model.transformer.h[li]
        errs = []
        with torch.no_grad():
            for h in range(self.H):
                Wk = block.attn.c_attn.weight[:, self.d + h * self.dh:self.d + (h + 1) * self.dh]
                # S = X (Wq Wk^T) X^T / sqrt(dh), G_A := dL/d(Wq Wk^T)  (the 1/sqrt(dh)
                # is folded into G_A), so dL/dWq = G_A @ Wk exactly
                pred = self.G_A[(li, h)].to(Wk.device) @ Wk
                got = self.G[(li, "c_attn")][:, h * self.dh:(h + 1) * self.dh].to(Wk.device)
                errs.append(((pred - got).norm() / got.norm()).item())
            # MLP capture sanity: X2^T Delta must equal autograd's c_fc grad
            X2, Delta = self.x_mlp[li], self.h_mlp[li].grad
            pred = torch.einsum("btd,bth->dh", X2, Delta)
            got = self.G[(li, "mlp_in")].to(pred.device)
            errs.append(((pred - got).norm() / got.norm()).item())
        return errs

    def close(self):
        for h in self._hooks:
            h.remove()


def split_qkv(G_cattn, d, dh, H):
    """c_attn grad (d, 3d) -> per-matrix grads W_Q, W_K, W_V (d, d)."""
    return G_cattn[:, :d], G_cattn[:, d:2 * d], G_cattn[:, 2 * d:]


def spectra_from_probe(probe):
    """Singular spectra + slopes for every accumulated gradient."""
    out = {}
    for (li, name), g in probe.G.items():
        mats = {}
        if name == "c_attn":
            q, k, v = split_qkv(g, probe.d, probe.dh, probe.H)
            mats = {"W_Q": q, "W_K": k, "W_V": v}
        else:
            mats = {name: g}
        for mname, m in mats.items():
            s = torch.linalg.svdvals(m.float()).numpy()
            s = s / s[0]
            out[f"L{li}/{mname}"] = {"spectrum": s.tolist()[:512],
                                     "slope": powerlaw_slope(s)}
    for (li, h), g in probe.G_A.items():
        s = torch.linalg.svdvals(g.float()).numpy()
        s = s / s[0]
        entry = {"spectrum": s.tolist()[:512], "slope": powerlaw_slope(s)}
        for null, store in probe.G_A_null.items():
            sn = torch.linalg.svdvals(store[(li, h)].float()).numpy()
            entry[f"slope_{null}"] = powerlaw_slope(sn / sn[0])
        out[f"L{li}/A_head{h}"] = entry
    # attach MLP null slopes to the mlp_in entries of the selected layers
    for li in HEAD_LAYERS:
        key = f"L{li}/mlp_in"
        for null, store in probe.G_mlp_null.items():
            sn = torch.linalg.svdvals(store[(li,)].float()).numpy()
            out[key][f"slope_{null}"] = powerlaw_slope(sn / sn[0])
    return out


def usage_spectra(model, probe):
    """Association-usage frequencies for trained circuits (selected layers)."""
    d, dh, H = probe.d, probe.dh, probe.H
    usage = {}
    with torch.no_grad():
        # FFN neurons: E[a^2], ranked
        for li, e in probe.neuron_energy.items():
            u = np.sort(e.numpy())[::-1]
            u = u / u[0]
            usage[f"L{li}/ffn_neurons"] = {"usage": u.tolist()[:1024],
                                           "slope": powerlaw_slope(u, 8, 768)}
        # QK directions: sigma_k^2 E[(u_k.x)^2] E[(v_k.x)^2] using each layer's
        # OWN ln_1 second moment
        for li in HEAD_LAYERS:
            C = probe.cov_attn[li] / probe.n_tokens
            block = model.transformer.h[li]
            W = block.attn.c_attn.weight.detach().cpu()
            for h in range(H):
                Wq = W[:, h * dh:(h + 1) * dh]
                Wk = W[:, d + h * dh:d + (h + 1) * dh]
                A = (Wq @ Wk.T / math.sqrt(dh)).float()
                U, S, Vh = torch.linalg.svd(A)
                eu = torch.einsum("ik,ij,jk->k", U, C.float(), U)
                ev = torch.einsum("ik,ij,jk->k", Vh.T, C.float(), Vh.T)
                u = (S ** 2 * eu * ev).numpy()
                u = np.sort(u)[::-1]
                u = u / u[0]
                usage[f"L{li}/qk_dirs_head{h}"] = {"usage": u.tolist()[:dh],
                                                   "slope": powerlaw_slope(u, 4, dh)}
    return usage


def covariance_spectra(probe):
    """Per-layer input SECOND MOMENTS E[x x^T] (uncentered: the top eigenvalue
    is dominated by the corpus-mean direction; the tail slope is unaffected)."""
    out = {}
    for fam, covs in (("attn_input", probe.cov_attn), ("mlp_input", probe.cov_mlp)):
        for li, C in covs.items():
            w = torch.linalg.eigvalsh(C.float() / max(probe.n_tokens, 1)).numpy()[::-1]
            w = w / w[0]
            out[f"L{li}/{fam}"] = {"spectrum": w.tolist()[:512], "slope": powerlaw_slope(w)}
    return out


def run_probe(model, batches, n_batches, tag):
    probe = Probe(model)
    t0 = time.time()
    for i in range(min(n_batches, batches.shape[0])):
        probe.step(batches[i].to(DEVICE))
        if i == 0:
            errs = probe.self_check()
            print(f"[{tag}] self-check dL/dA vs autograd, rel err per head: "
                  f"max {max(errs):.2e} {'OK' if max(errs) < 1e-4 else 'FAIL'}")
        if (i + 1) % 12 == 0:
            print(f"[{tag}] batch {i+1}, {time.time()-t0:.0f}s", flush=True)
    probe.close()
    return probe


def token_zipf(ids):
    counts = np.bincount(ids.numpy())
    counts = np.sort(counts[counts > 0])[::-1].astype(float)
    counts = counts / counts[0]
    return {"counts": counts.tolist()[:2048], "slope": powerlaw_slope(counts, 8, 1024)}


def main(smoke=False):
    from transformers import AutoModelForCausalLM, AutoConfig
    n_batches = 2 if smoke else N_BATCHES
    batches, ids, tok = load_text_batches()
    print(f"{batches.shape[0]} batches of {BATCH}x{SEQ_LEN} tokens on {DEVICE}")
    results = {"config": {"model": MODEL, "seq_len": SEQ_LEN, "batch": BATCH,
                          "n_batches": n_batches, "head_layers": list(HEAD_LAYERS),
                          "fit_range": list(FIT_RANGE), "device": DEVICE}}
    results["token_zipf"] = token_zipf(ids)
    print(f"corpus unigram Zipf slope: {results['token_zipf']['slope']:.2f}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
    model.to(DEVICE).eval()   # eval: dropout off; grads still flow
    probe = run_probe(model, batches, n_batches, "trained")
    results["trained_grad_spectra"] = spectra_from_probe(probe)
    results["usage"] = usage_spectra(model, probe)
    results["covariance"] = covariance_spectra(probe)

    # re-initialized model: the paper's one-step-from-init setting
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL)
    init_model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    init_model.to(DEVICE).eval()
    probe_i = run_probe(init_model, batches, n_batches, "init")
    results["init_grad_spectra"] = spectra_from_probe(probe_i)
    results["covariance_init"] = covariance_spectra(probe_i)

    os.makedirs("reports", exist_ok=True)
    with open("reports/zipf_qk_results.json", "w") as f:
        json.dump(results, f, indent=1)

    make_plots(results)
    print("Saved reports/zipf_qk_spectra.png, zipf_qk_usage.png, zipf_qk_results.json")


CIRCUIT_COLORS = {"W_Q": "tab:blue", "W_K": "tab:cyan", "A_head": "navy",
                  "W_V": "tab:green", "attn_out": "tab:olive",
                  "mlp_in": "tab:red", "mlp_out": "tab:orange"}


def _group(spectra, key):
    return [v for k, v in spectra.items() if key in k]


def make_plots(results):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    def plot_family(ax, spectra, title):
        for key, color in CIRCUIT_COLORS.items():
            group = _group(spectra, key)
            if not group:
                continue
            arrs = [np.array(g["spectrum"]) for g in group]
            n = min(len(a) for a in arrs)
            med = np.exp(np.median(np.log(np.stack([a[:n] for a in arrs]) + 1e-30), axis=0))
            ax.loglog(np.arange(1, n + 1), med, color=color, label=key, alpha=0.9)
        tz = np.array(results["token_zipf"]["counts"])
        ax.loglog(np.arange(1, len(tz) + 1), tz, "k--", lw=1, alpha=0.6,
                  label=f"corpus Zipf (slope {results['token_zipf']['slope']:.2f})")
        ax.set_xlabel("singular value rank")
        ax.set_ylabel("normalized singular value")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.25)

    plot_family(axes[0, 0], results["trained_grad_spectra"],
                "Gradient spectra at TRAINED weights (median across layers/heads)")
    plot_family(axes[0, 1], results["init_grad_spectra"],
                "Gradient spectra at RE-INITIALIZED weights")

    # true-minus-null slope gaps: the mechanism-specific evidence.
    # gap < 0 means the real gradient's tail is steeper than a null built from
    # the SAME error signal with shuffled inputs (content and/or position destroyed)
    ax = axes[1, 0]
    groups, labels = [], []
    for tag, spectra in (("trained", results["trained_grad_spectra"]),
                         ("init", results["init_grad_spectra"])):
        for fam in ("A_head", "mlp_in"):
            for null in ("permT", "permB"):
                gaps = [v["slope"] - v[f"slope_{null}"] for k, v in spectra.items()
                        if fam in k and f"slope_{null}" in v
                        and math.isfinite(v["slope"]) and math.isfinite(v[f"slope_{null}"])]
                if gaps:
                    groups.append(gaps)
                    labels.append(f"{fam}\n{tag}\nvs {null}")
    ax.boxplot(groups, tick_labels=labels)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("slope(true) - slope(null)")
    ax.set_title("Structure beyond input geometry: true-vs-null tail-slope gaps")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(HEAD_LAYERS)))
    for ci, li in enumerate(HEAD_LAYERS):
        for src, style in (("covariance", "-"), ("covariance_init", ":")):
            key = f"L{li}/attn_input"
            if key in results.get(src, {}):
                c = results[src][key]
                s = np.array(c["spectrum"])
                ax.loglog(np.arange(1, len(s) + 1), s, style, color=colors[ci],
                          label=f"L{li} {'trained' if src == 'covariance' else 'init'} "
                                f"({c['slope']:.2f})", alpha=0.85)
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("normalized eigenvalue")
    ax.set_title("Attention-input second moments E[xxᵀ] per layer (anisotropy axis)")
    ax.legend(fontsize=6.5)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig("reports/zipf_qk_spectra.png", dpi=150)
    plt.close()

    # usage plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    for k, v in results["usage"].items():
        if "ffn" in k:
            u = np.array(v["usage"])
            ax1.loglog(np.arange(1, len(u) + 1), u, alpha=0.8,
                       label=f"{k} (slope {v['slope']:.2f})")
    tz = np.array(results["token_zipf"]["counts"])
    ax1.loglog(np.arange(1, len(tz) + 1), tz, "k--", lw=1,
               label=f"corpus Zipf ({results['token_zipf']['slope']:.2f})")
    ax1.set_title("FFN neuron usage E[a^2] (ranked)")
    ax1.set_xlabel("neuron rank")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", alpha=0.25)
    qk_slopes = []
    for k, v in results["usage"].items():
        if "qk_dirs" in k:
            u = np.array(v["usage"])
            ax2.loglog(np.arange(1, len(u) + 1), u, alpha=0.35, color="navy")
            qk_slopes.append(v["slope"])
    ax2.set_title(f"QK singular-direction usage per head "
                  f"(median slope {np.median(qk_slopes):.2f})")
    ax2.set_xlabel("direction rank (of 64)")
    ax2.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig("reports/zipf_qk_usage.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    t0 = time.time()
    main(smoke="--smoke" in sys.argv)
    print(f"Total time: {time.time() - t0:.1f}s")
