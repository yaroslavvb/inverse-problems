#!/usr/bin/env python3
"""Generate Noam-style einsum snippets and TensorGrad-style attention diagrams.

The diagrams are intentionally generated as plain SVG so the script has no
runtime dependency on LaTeX, Graphviz, or TensorGrad.  The notation mirrors
TensorGrad's tensor-network diagrams: tensors are nodes, named dimensions are
wires, and contractions are the repeated wire labels.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "reports" / "attention_variants"


DIMENSION_KEY = {
    "B": "batch",
    "L": "query sequence length",
    "T": "key/value sequence length",
    "D": "model dimension",
    "H": "query heads",
    "K": "per-head key/value channel",
    "G": "key/value groups",
    "C": "query heads inside each key/value group",
    "Q": "query groups",
    "F": "kernel feature dimension for linear attention",
    "R": "latent/compressed KV rank",
}


SOURCES = {
    "shape_suffixes": (
        "Noam Shazeer, Shape Suffixes - Good Coding Style",
        "https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd",
    ),
    "transformer": (
        "Vaswani et al., Attention Is All You Need",
        "https://arxiv.org/abs/1706.03762",
    ),
    "mqa": (
        "Noam Shazeer, Fast Transformer Decoding: One Write-Head is All You Need",
        "https://arxiv.org/abs/1911.02150",
    ),
    "gqa": (
        "Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
        "https://aclanthology.org/2023.emnlp-main.298/",
    ),
    "longformer": (
        "Beltagy, Peters, and Cohan, Longformer: The Long-Document Transformer",
        "https://arxiv.org/abs/2004.05150",
    ),
    "bigbird": (
        "Zaheer et al., Big Bird: Transformers for Longer Sequences",
        "https://arxiv.org/abs/2007.14062",
    ),
    "linear": (
        "Katharopoulos et al., Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
        "https://arxiv.org/abs/2006.16236",
    ),
    "mla": (
        "DeepSeek-AI, DeepSeek-V2",
        "https://arxiv.org/abs/2405.04434",
    ),
}


@dataclass(frozen=True)
class Node:
    node_id: str
    label: str
    suffix: str
    x: float
    y: float
    kind: str = "tensor"


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    label: str
    kind: str = "wire"


@dataclass(frozen=True)
class Cluster:
    x: float
    y: float
    width: float
    height: float
    label: str


@dataclass(frozen=True)
class Variant:
    slug: str
    title: str
    subtitle: str
    source_ids: tuple[str, ...]
    equations: tuple[str, ...]
    tensorgrad_graph: tuple[str, ...]
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    clusters: tuple[Cluster, ...]
    notes: tuple[str, ...] = ()


def n(
    node_id: str,
    label: str,
    suffix: str,
    x: float,
    y: float,
    kind: str = "tensor",
) -> Node:
    return Node(node_id, label, suffix, x, y, kind)


def e(source: str, target: str, label: str, kind: str = "wire") -> Edge:
    return Edge(source, target, label, kind)


def common_softmax_nodes(
    *,
    q_suffix: str,
    k_suffix: str,
    v_suffix: str,
    wq_suffix: str,
    wk_suffix: str,
    wv_suffix: str,
    wo_suffix: str,
    logits_suffix: str,
    weights_suffix: str,
    context_suffix: str,
    q_wire: str,
    k_wire: str,
    v_wire: str,
    logits_shape: str,
    softmax_label: str = "softmax_T",
    mask_node: Node | None = None,
    mask_edge_label: str = "",
) -> tuple[tuple[Node, ...], tuple[Edge, ...], tuple[Cluster, ...]]:
    nodes = [
        n("xq", "input", "BLD", 80, 175, "input"),
        n("xk", "memory", "BTD", 80, 305, "input"),
        n("xv", "memory", "BTD", 80, 435, "input"),
        n("wq", "W^Q", wq_suffix, 235, 175, "weight"),
        n("wk", "W^K", wk_suffix, 235, 305, "weight"),
        n("wv", "W^V", wv_suffix, 235, 435, "weight"),
        n("q", "Q", q_suffix, 380, 175, "activation"),
        n("k", "K", k_suffix, 380, 305, "activation"),
        n("v", "V", v_suffix, 380, 435, "activation"),
        n("logits", "QK^T", logits_suffix, 545, 240, "op"),
        n("softmax", softmax_label, logits_suffix, 695, 240, "op"),
        n("a", "A", weights_suffix, 845, 240, "activation"),
        n("context", "A V", context_suffix, 845, 385, "op"),
        n("wo", "W^O", wo_suffix, 1005, 385, "weight"),
        n("out", "out", "BLD", 1145, 385, "output"),
    ]
    edges = [
        e("xq", "wq", "D"),
        e("wq", "q", q_wire),
        e("xk", "wk", "D"),
        e("wk", "k", k_wire),
        e("xv", "wv", "D"),
        e("wv", "v", v_wire),
        e("q", "logits", q_suffix),
        e("k", "logits", k_suffix),
        e("logits", "softmax", logits_suffix, "attention"),
        e("softmax", "a", weights_suffix, "attention"),
        e("a", "context", weights_suffix, "attention"),
        e("v", "context", v_suffix),
        e("context", "wo", context_suffix),
        e("wo", "out", "D"),
    ]
    if mask_node:
        nodes.append(mask_node)
        edges.append(e(mask_node.node_id, "logits", mask_edge_label or mask_node.suffix, "mask"))
    clusters = (
        Cluster(500, 120, 390, 210, f"attention tensor: {logits_shape}"),
        Cluster(780, 335, 280, 115, "value mixing + output"),
    )
    return tuple(nodes), tuple(edges), clusters


def linear_attention_nodes() -> tuple[tuple[Node, ...], tuple[Edge, ...], tuple[Cluster, ...]]:
    nodes = (
        n("xq", "input", "BLD", 80, 175, "input"),
        n("xk", "memory", "BTD", 80, 305, "input"),
        n("xv", "memory", "BTD", 80, 435, "input"),
        n("wq", "W^Q", "DHF", 235, 175, "weight"),
        n("wk", "W^K", "DHF", 235, 305, "weight"),
        n("wv", "W^V", "DHK", 235, 435, "weight"),
        n("phiq", "phi(Q)", "BLHF", 410, 175, "op"),
        n("phik", "phi(K)", "BTHF", 410, 305, "op"),
        n("v", "V", "BTHK", 410, 435, "activation"),
        n("kv", "K^T V", "BHFK", 615, 360, "op"),
        n("ksum", "sum K", "BHF", 615, 245, "op"),
        n("denom", "norm", "BLH", 795, 245, "op"),
        n("context", "Q(K^T V)", "BLHK", 835, 385, "op"),
        n("wo", "W^O", "HKD", 1005, 385, "weight"),
        n("out", "out", "BLD", 1145, 385, "output"),
    )
    edges = (
        e("xq", "wq", "D"),
        e("wq", "phiq", "H,F"),
        e("xk", "wk", "D"),
        e("wk", "phik", "H,F"),
        e("xv", "wv", "D"),
        e("wv", "v", "H,K"),
        e("phik", "kv", "B,T,H,F"),
        e("v", "kv", "B,T,H,K"),
        e("phik", "ksum", "B,T,H,F"),
        e("phiq", "denom", "BLHF"),
        e("ksum", "denom", "BHF"),
        e("phiq", "context", "BLHF"),
        e("kv", "context", "BHFK"),
        e("denom", "context", "divide"),
        e("context", "wo", "BLHK"),
        e("wo", "out", "D"),
    )
    clusters = (
        Cluster(560, 205, 300, 250, "associative contraction, no LxT tensor"),
    )
    return nodes, edges, clusters


def mla_nodes() -> tuple[tuple[Node, ...], tuple[Edge, ...], tuple[Cluster, ...]]:
    nodes = (
        n("xq", "input", "BLD", 80, 175, "input"),
        n("xkv", "memory", "BTD", 80, 350, "input"),
        n("wqdown", "W^DQ", "DR", 225, 135, "weight"),
        n("wqup", "W^UQ", "RHK", 365, 135, "weight"),
        n("q", "Q", "BLHK", 515, 175, "activation"),
        n("wkvdown", "W^DKV", "DR", 225, 350, "weight"),
        n("latent", "latent KV", "BTR", 365, 350, "latent"),
        n("wkup", "W^UK", "RHK", 515, 305, "weight"),
        n("wvup", "W^UV", "RHK", 515, 435, "weight"),
        n("k", "K", "BTHK", 665, 305, "activation"),
        n("v", "V", "BTHK", 665, 435, "activation"),
        n("logits", "QK^T", "BHLT", 815, 240, "op"),
        n("softmax", "softmax_T", "BHLT", 965, 240, "op"),
        n("context", "A V", "BLHK", 965, 385, "op"),
        n("wo", "W^O", "HKD", 1110, 385, "weight"),
        n("out", "out", "BLD", 1245, 385, "output"),
    )
    edges = (
        e("xq", "wqdown", "D"),
        e("wqdown", "wqup", "R"),
        e("wqup", "q", "H,K"),
        e("xkv", "wkvdown", "D"),
        e("wkvdown", "latent", "R"),
        e("latent", "wkup", "R"),
        e("latent", "wvup", "R"),
        e("wkup", "k", "H,K"),
        e("wvup", "v", "H,K"),
        e("q", "logits", "BLHK"),
        e("k", "logits", "BTHK"),
        e("logits", "softmax", "BHLT", "attention"),
        e("softmax", "context", "BHLT", "attention"),
        e("v", "context", "BTHK"),
        e("context", "wo", "BLHK"),
        e("wo", "out", "D"),
    )
    clusters = (
        Cluster(185, 255, 530, 245, "KV cache stored as latent BTR"),
        Cluster(775, 120, 230, 195, "attention tensor: BHLT"),
    )
    return nodes, edges, clusters


def variants() -> tuple[Variant, ...]:
    mha_nodes, mha_edges, mha_clusters = common_softmax_nodes(
        q_suffix="BLHK",
        k_suffix="BTHK",
        v_suffix="BTHK",
        wq_suffix="DHK",
        wk_suffix="DHK",
        wv_suffix="DHK",
        wo_suffix="HKD",
        logits_suffix="BHLT",
        weights_suffix="BHLT",
        context_suffix="BLHK",
        q_wire="H,K",
        k_wire="H,K",
        v_wire="H,K",
        logits_shape="B,H,L,T",
    )
    gqa_nodes, gqa_edges, gqa_clusters = common_softmax_nodes(
        q_suffix="BLGCK",
        k_suffix="BTGK",
        v_suffix="BTGK",
        wq_suffix="DGCK",
        wk_suffix="DGK",
        wv_suffix="DGK",
        wo_suffix="GCKD",
        logits_suffix="BLGCT",
        weights_suffix="BLGCT",
        context_suffix="BLGCK",
        q_wire="G,C,K",
        k_wire="G,K",
        v_wire="G,K",
        logits_shape="B,L,G,C,T",
    )
    all_to_all_nodes, all_to_all_edges, all_to_all_clusters = common_softmax_nodes(
        q_suffix="BLQK",
        k_suffix="BTGK",
        v_suffix="BTGK",
        wq_suffix="DQK",
        wk_suffix="DGK",
        wv_suffix="DGK",
        wo_suffix="QKD",
        logits_suffix="BLQGT",
        weights_suffix="BLQGT",
        context_suffix="BLQK",
        q_wire="Q,K",
        k_wire="G,K",
        v_wire="G,K",
        logits_shape="B,L,Q,G,T",
        softmax_label="softmax_GT",
    )
    mqa_nodes, mqa_edges, mqa_clusters = common_softmax_nodes(
        q_suffix="BLHK",
        k_suffix="BTK",
        v_suffix="BTK",
        wq_suffix="DHK",
        wk_suffix="DK",
        wv_suffix="DK",
        wo_suffix="HKD",
        logits_suffix="BHLT",
        weights_suffix="BHLT",
        context_suffix="BLHK",
        q_wire="H,K",
        k_wire="K",
        v_wire="K",
        logits_shape="B,H,L,T",
    )
    sliding_nodes, sliding_edges, sliding_clusters = common_softmax_nodes(
        q_suffix="BLHK",
        k_suffix="BTHK",
        v_suffix="BTHK",
        wq_suffix="DHK",
        wk_suffix="DHK",
        wv_suffix="DHK",
        wo_suffix="HKD",
        logits_suffix="BHLT",
        weights_suffix="BHLT",
        context_suffix="BLHK",
        q_wire="H,K",
        k_wire="H,K",
        v_wire="H,K",
        logits_shape="B,H,L,T masked by window_LT",
        mask_node=n("mask", "window", "LT", 545, 85, "mask"),
        mask_edge_label="LT",
    )
    bigbird_nodes, bigbird_edges, bigbird_clusters = common_softmax_nodes(
        q_suffix="BLHK",
        k_suffix="BTHK",
        v_suffix="BTHK",
        wq_suffix="DHK",
        wk_suffix="DHK",
        wv_suffix="DHK",
        wo_suffix="HKD",
        logits_suffix="BHLT",
        weights_suffix="BHLT",
        context_suffix="BLHK",
        q_wire="H,K",
        k_wire="H,K",
        v_wire="H,K",
        logits_shape="B,H,L,T masked by sparse_LT",
        mask_node=n("mask", "window\nrandom\nglobal", "LT", 545, 85, "mask"),
        mask_edge_label="LT",
    )
    linear_nodes_, linear_edges_, linear_clusters_ = linear_attention_nodes()
    mla_nodes_, mla_edges_, mla_clusters_ = mla_nodes()

    return (
        Variant(
            slug="multi_head_attention",
            title="Multi-Head Attention",
            subtitle="Independent Q, K, V projections per head; dense L x T attention per head.",
            source_ids=("shape_suffixes", "transformer"),
            equations=(
                'query_BLHK = einsum("BLD,DHK->BLHK", input_BLD, w_q_DHK)',
                'key_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_k_DHK)',
                'value_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_v_DHK)',
                'logits_BHLT = einsum("BLHK,BTHK->BHLT", query_BLHK, key_BTHK) / sqrt(K)',
                'weights_BHLT = softmax(logits_BHLT, dim="T")',
                'wtd_values_BLHK = einsum("BHLT,BTHK->BLHK", weights_BHLT, value_BTHK)',
                'out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)',
            ),
            tensorgrad_graph=(
                "# score logits_BHLT",
                "input_q -D- Wq",
                "memory_k -D- Wk",
                "Wq -K- Wk",
                "# H remains a free shared edge in logits_BHLT",
                "# A_BHLT = softmax_T(logits_BHLT)",
                "# output out_BLD",
                "A -T- memory_v",
                "memory_v -D- Wv",
                "A -H- *head",
                "Wv -H- *head",
                "*head -H- Wo",
                "Wv -K- Wo",
                "A -L-",
                "Wo -D-",
            ),
            nodes=mha_nodes,
            edges=mha_edges,
            clusters=mha_clusters,
        ),
        Variant(
            slug="grouped_multiquery_attention",
            title="Grouped Multi-Query / Grouped-Query Attention",
            subtitle="C query heads share one K/V group; H = G*C.",
            source_ids=("shape_suffixes", "mqa", "gqa"),
            equations=(
                "# H = G * C",
                'query_BLGCK = einsum("BLD,DGCK->BLGCK", input_BLD, w_q_DGCK)',
                'key_BTGK = einsum("BTD,DGK->BTGK", memory_BTD, w_k_DGK)',
                'value_BTGK = einsum("BTD,DGK->BTGK", memory_BTD, w_v_DGK)',
                'logits_BLGCT = einsum("BLGCK,BTGK->BLGCT", query_BLGCK, key_BTGK) / sqrt(K)',
                'weights_BLGCT = softmax(logits_BLGCT, dim="T")',
                'wtd_values_BLGCK = einsum("BLGCT,BTGK->BLGCK", weights_BLGCT, value_BTGK)',
                'out_BLD = einsum("BLGCK,GCKD->BLD", wtd_values_BLGCK, w_o_GCKD)',
            ),
            tensorgrad_graph=(
                "# score logits_BLGCT",
                "input_q -D- Wq",
                "memory_k -D- Wk",
                "Wq -K- Wk",
                "# G and C stay free; A_BLGCT = softmax_T(logits_BLGCT)",
                "# output out_BLD",
                "A -T- memory_v",
                "memory_v -D- Wv",
                "A -G- *group",
                "Wv -G- *group",
                "*group -G- Wo",
                "Wv -K- Wo",
                "A -C- Wo",
                "A -L-",
                "Wo -D-",
            ),
            nodes=gqa_nodes,
            edges=gqa_edges,
            clusters=gqa_clusters,
        ),
        Variant(
            slug="all_to_all_attention",
            title="All-to-All Attention",
            subtitle="Every query group attends across every K/V group; softmax normalizes over (G,T).",
            source_ids=("shape_suffixes",),
            equations=(
                'query_BLQK = einsum("BLD,DQK->BLQK", input_BLD, w_q_DQK)',
                'key_BTGK = einsum("BTD,DGK->BTGK", memory_BTD, w_k_DGK)',
                'value_BTGK = einsum("BTD,DGK->BTGK", memory_BTD, w_v_DGK)',
                'logits_BLQGT = einsum("BLQK,BTGK->BLQGT", query_BLQK, key_BTGK) / sqrt(K)',
                'weights_BLQGT = softmax(logits_BLQGT, dim=("G", "T"))',
                'wtd_values_BLQK = einsum("BLQGT,BTGK->BLQK", weights_BLQGT, value_BTGK)',
                'out_BLD = einsum("BLQK,QKD->BLD", wtd_values_BLQK, w_o_QKD)',
            ),
            tensorgrad_graph=(
                "# score logits_BLQGT",
                "input_q -D- Wq",
                "memory_k -D- Wk",
                "Wq -K- Wk",
                "# Q and G both stay visible in the attention tensor",
                "# A_BLQGT = softmax_GT(logits_BLQGT)",
                "# output out_BLD",
                "A -T- memory_v",
                "memory_v -D- Wv",
                "A -G- Wv",
                "Wv -K- Wo",
                "A -Q- Wo",
                "A -L-",
                "Wo -D-",
            ),
            nodes=all_to_all_nodes,
            edges=all_to_all_edges,
            clusters=all_to_all_clusters,
            notes=(
                "This is an exploratory structural variant from the prompt rather than a named paper baseline.",
                "If you want per-KV-group normalization instead, change softmax_GT to softmax_T and keep G until the value contraction.",
            ),
        ),
        Variant(
            slug="multi_query_attention",
            title="Multi-Query Attention",
            subtitle="All query heads share a single K/V head, shrinking the decode-time KV cache.",
            source_ids=("shape_suffixes", "mqa"),
            equations=(
                'query_BLHK = einsum("BLD,DHK->BLHK", input_BLD, w_q_DHK)',
                'key_BTK = einsum("BTD,DK->BTK", memory_BTD, w_k_DK)',
                'value_BTK = einsum("BTD,DK->BTK", memory_BTD, w_v_DK)',
                'logits_BHLT = einsum("BLHK,BTK->BHLT", query_BLHK, key_BTK) / sqrt(K)',
                'weights_BHLT = softmax(logits_BHLT, dim="T")',
                'wtd_values_BLHK = einsum("BHLT,BTK->BLHK", weights_BHLT, value_BTK)',
                'out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)',
            ),
            tensorgrad_graph=(
                "# score logits_BHLT",
                "input_q -D- Wq",
                "memory_k -D- Wk",
                "Wq -K- Wk",
                "# H is free only on Q; K/V are shared across H",
                "# output out_BLD",
                "A -T- memory_v",
                "memory_v -D- Wv",
                "Wv -K- Wo",
                "A -H- Wo",
                "A -L-",
                "Wo -D-",
            ),
            nodes=mqa_nodes,
            edges=mqa_edges,
            clusters=mqa_clusters,
        ),
        Variant(
            slug="sliding_window_attention",
            title="Sliding-Window Attention",
            subtitle="Dense per-head attention, but logits outside the local L x T band are masked.",
            source_ids=("shape_suffixes", "longformer"),
            equations=(
                'query_BLHK = einsum("BLD,DHK->BLHK", input_BLD, w_q_DHK)',
                'key_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_k_DHK)',
                'value_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_v_DHK)',
                'logits_BHLT = einsum("BLHK,BTHK->BHLT", query_BLHK, key_BTHK) / sqrt(K)',
                'logits_BHLT = where(window_mask_LT, logits_BHLT, -inf)',
                'weights_BHLT = softmax(logits_BHLT, dim="T")',
                'wtd_values_BLHK = einsum("BHLT,BTHK->BLHK", weights_BHLT, value_BTHK)',
                'out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)',
            ),
            tensorgrad_graph=(
                "# score logits_BHLT with a structural mask",
                "input_q -D- Wq",
                "memory_k -D- Wk",
                "Wq -K- Wk",
                "# H remains a free shared edge in logits_BHLT",
                "window_mask -L-T- logits",
                "# A_BHLT = softmax_T(masked_logits_BHLT)",
                "# value/output graph is the same as multi-head attention",
            ),
            nodes=sliding_nodes,
            edges=sliding_edges,
            clusters=sliding_clusters,
        ),
        Variant(
            slug="bigbird_sparse_attention",
            title="BigBird-Style Block Sparse Attention",
            subtitle="The L x T mask is a union of sliding-window, random, and global-token blocks.",
            source_ids=("shape_suffixes", "bigbird"),
            equations=(
                'query_BLHK = einsum("BLD,DHK->BLHK", input_BLD, w_q_DHK)',
                'key_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_k_DHK)',
                'value_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_v_DHK)',
                'logits_BHLT = einsum("BLHK,BTHK->BHLT", query_BLHK, key_BTHK) / sqrt(K)',
                'sparse_mask_LT = window_LT | random_LT | global_LT',
                'logits_BHLT = where(sparse_mask_LT, logits_BHLT, -inf)',
                'weights_BHLT = softmax(logits_BHLT, dim="T")',
                'wtd_values_BLHK = einsum("BHLT,BTHK->BLHK", weights_BHLT, value_BTHK)',
                'out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)',
            ),
            tensorgrad_graph=(
                "# score logits_BHLT with sparse adjacency",
                "input_q -D- Wq",
                "memory_k -D- Wk",
                "Wq -K- Wk",
                "# H remains a free shared edge in logits_BHLT",
                "sparse_mask -L-T- logits",
                "# sparse_mask_LT = window_LT | random_LT | global_LT",
                "# value/output graph is the same as multi-head attention",
            ),
            nodes=bigbird_nodes,
            edges=bigbird_edges,
            clusters=bigbird_clusters,
        ),
        Variant(
            slug="linear_attention",
            title="Linear Attention",
            subtitle="Replace softmax attention with kernel features and reassociate K^T V before multiplying by Q.",
            source_ids=("shape_suffixes", "linear"),
            equations=(
                'query_BLHF = phi(einsum("BLD,DHF->BLHF", input_BLD, w_q_DHF))',
                'key_BTHF = phi(einsum("BTD,DHF->BTHF", memory_BTD, w_k_DHF))',
                'value_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_v_DHK)',
                'kv_BHFK = einsum("BTHF,BTHK->BHFK", key_BTHF, value_BTHK)',
                'k_sum_BHF = einsum("BTHF->BHF", key_BTHF)',
                'denom_BLH = einsum("BLHF,BHF->BLH", query_BLHF, k_sum_BHF)',
                'wtd_values_BLHK = einsum("BLHF,BHFK->BLHK", query_BLHF, kv_BHFK) / denom_BLH[..., None]',
                'out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)',
            ),
            tensorgrad_graph=(
                "# reassociated value summary",
                "phi_K -T- V",
                "phi_K -F- KV_summary",
                "V -K- KV_summary",
                "# query reads the summary and normalization",
                "phi_Q -F- KV_summary",
                "phi_Q -F- K_sum",
                "KV_summary -K- Wo",
                "Wo -D-",
            ),
            nodes=linear_nodes_,
            edges=linear_edges_,
            clusters=linear_clusters_,
        ),
        Variant(
            slug="multi_head_latent_attention",
            title="Multi-Head Latent Attention",
            subtitle="A simplified MLA-style low-rank KV cache: store BTR, expand to K/V only when scoring.",
            source_ids=("shape_suffixes", "mla"),
            equations=(
                'q_latent_BLR = einsum("BLD,DR->BLR", input_BLD, w_q_down_DR)',
                'query_BLHK = einsum("BLR,RHK->BLHK", q_latent_BLR, w_q_up_RHK)',
                'kv_latent_BTR = einsum("BTD,DR->BTR", memory_BTD, w_kv_down_DR)',
                'key_BTHK = einsum("BTR,RHK->BTHK", kv_latent_BTR, w_k_up_RHK)',
                'value_BTHK = einsum("BTR,RHK->BTHK", kv_latent_BTR, w_v_up_RHK)',
                'logits_BHLT = einsum("BLHK,BTHK->BHLT", query_BLHK, key_BTHK) / sqrt(K)',
                'weights_BHLT = softmax(logits_BHLT, dim="T")',
                'wtd_values_BLHK = einsum("BHLT,BTHK->BLHK", weights_BHLT, value_BTHK)',
                'out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)',
            ),
            tensorgrad_graph=(
                "# low-rank query path",
                "input_q -D- Wq_down -R- Wq_up",
                "# compressed KV cache path",
                "memory -D- Wkv_down -R- latent_kv",
                "latent_kv -R- Wk_up",
                "latent_kv -R- Wv_up",
                "# then ordinary score/value contractions",
                "Q -K- K",
                "A -T- V",
                "V -K- Wo",
                "Wo -D-",
            ),
            nodes=mla_nodes_,
            edges=mla_edges_,
            clusters=mla_clusters_,
            notes=(
                "The DeepSeek-V2 implementation has extra details such as decoupled RoPE; this diagram keeps only the structural low-rank KV-cache idea.",
            ),
        ),
    )


KIND_STYLE = {
    "input": ("#f8fafc", "#334155"),
    "tensor": ("#f8fafc", "#334155"),
    "activation": ("#ecfeff", "#0e7490"),
    "weight": ("#eef2ff", "#4338ca"),
    "op": ("#fff7ed", "#c2410c"),
    "mask": ("#fef2f2", "#b91c1c"),
    "latent": ("#f0fdf4", "#15803d"),
    "output": ("#f5f3ff", "#7c3aed"),
}


EDGE_STYLE = {
    "wire": ("#334155", "4"),
    "attention": ("#0f766e", "4"),
    "mask": ("#b91c1c", "3"),
}


def text_element(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    weight: str = "400",
    anchor: str = "middle",
    fill: str = "#0f172a",
    family: str = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}">{escape(text)}</text>'
    )


def wrap_label(label: str, width: int = 30) -> list[str]:
    lines: list[str] = []
    for raw_line in label.splitlines() or [label]:
        if len(raw_line) <= width:
            lines.append(raw_line)
        else:
            lines.extend(textwrap.wrap(raw_line, width=width, break_long_words=False) or [raw_line])
    return lines


def render_node(node: Node) -> str:
    fill, stroke = KIND_STYLE.get(node.kind, KIND_STYLE["tensor"])
    width = 96
    height = 56
    if node.kind == "weight":
        width = 78
        height = 50
    if node.kind == "op":
        width = 112
        height = 52
    if node.kind == "mask":
        width = 126
        height = 50
    if node.kind == "latent":
        width = 116
        height = 52
    x = node.x - width / 2
    y = node.y - height / 2
    rx = 10 if node.kind != "mask" else 6
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width}" height="{height}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>'
    ]
    label_lines = wrap_label(node.label, 18)
    line_y = node.y - 4 if node.suffix else node.y + 4
    if len(label_lines) > 1:
        line_y -= 8 * (len(label_lines) - 1)
    for i, line in enumerate(label_lines):
        parts.append(text_element(node.x, line_y + i * 15, line, size=14, weight="700", fill="#111827"))
    if node.suffix:
        parts.append(
            text_element(
                node.x,
                node.y + height / 2 - 10,
                node.suffix,
                size=11,
                fill="#475569",
                family="'SFMono-Regular', Consolas, 'Liberation Mono', monospace",
            )
        )
    return "\n".join(parts)


def render_edge(edge: Edge, nodes_by_id: dict[str, Node]) -> str:
    source = nodes_by_id[edge.source]
    target = nodes_by_id[edge.target]
    color, stroke_width = EDGE_STYLE.get(edge.kind, EDGE_STYLE["wire"])
    mid_x = (source.x + target.x) / 2
    mid_y = (source.y + target.y) / 2
    dx = target.x - source.x
    curve = min(80, abs(dx) * 0.35)
    if abs(source.y - target.y) > 80:
        curve = min(115, abs(dx) * 0.45)
    path = (
        f"M {source.x:.1f},{source.y:.1f} "
        f"C {source.x + curve:.1f},{source.y:.1f} {target.x - curve:.1f},{target.y:.1f} {target.x:.1f},{target.y:.1f}"
    )
    label_width = max(28, len(edge.label) * 7 + 12)
    label = (
        f'<rect x="{mid_x - label_width / 2:.1f}" y="{mid_y - 14:.1f}" width="{label_width}" height="20" rx="5" '
        f'fill="#ffffff" fill-opacity="0.92" stroke="#e2e8f0"/>'
        + text_element(
            mid_x,
            mid_y + 1,
            edge.label,
            size=11,
            fill=color,
            family="'SFMono-Regular', Consolas, 'Liberation Mono', monospace",
        )
    )
    dash = ' stroke-dasharray="7 5"' if edge.kind == "mask" else ""
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke_width}"{dash}/>\n{label}'


def render_cluster(cluster: Cluster) -> str:
    return "\n".join(
        [
            f'<rect x="{cluster.x:.1f}" y="{cluster.y:.1f}" width="{cluster.width:.1f}" height="{cluster.height:.1f}" '
            'rx="12" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="7 7"/>',
            text_element(cluster.x + cluster.width / 2, cluster.y + 22, cluster.label, size=13, weight="700", fill="#475569"),
        ]
    )


def render_svg(variant: Variant) -> str:
    max_x = max(node.x for node in variant.nodes) + 95
    max_y = max(node.y for node in variant.nodes) + 95
    width = max(1240, int(max_x))
    height = max(535, int(max_y))
    nodes_by_id = {node.node_id: node for node in variant.nodes}
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text_element(36, 42, variant.title, size=25, weight="800", anchor="start"),
        text_element(36, 68, variant.subtitle, size=13, anchor="start", fill="#475569"),
    ]
    parts.extend(render_cluster(cluster) for cluster in variant.clusters)
    parts.extend(render_edge(edge, nodes_by_id) for edge in variant.edges)
    parts.extend(render_node(node) for node in variant.nodes)
    parts.append("</svg>")
    return "\n".join(parts)


def render_markdown(out_dir: Path, generated_svg_paths: dict[str, Path]) -> str:
    lines = [
        "# Transformer Attention Variants",
        "",
        "Generated by `attention_variant_diagrams.py`.",
        "",
        "The equations use Noam-style shape suffixes: every tensor name ends with the ordered logical dimensions in that tensor. The diagrams use a TensorGrad-inspired notation: tensors are nodes, repeated named dimensions are contractions, and free dimensions remain visible as suffixes.",
        "",
        "## Dimension key",
        "",
    ]
    for key, desc in DIMENSION_KEY.items():
        lines.append(f"- `{key}`: {desc}")
    lines.extend(["", "## Variants", ""])

    for variant in variants():
        svg_rel = generated_svg_paths[variant.slug].relative_to(out_dir)
        lines.extend(
            [
                f"### {variant.title}",
                "",
                variant.subtitle,
                "",
                f"![{variant.title}]({svg_rel.as_posix()})",
                "",
                "**Noam-style einsum**",
                "",
                "```python",
                *variant.equations,
                "```",
                "",
                "**TensorGrad-style graph spec**",
                "",
                "```text",
                *variant.tensorgrad_graph,
                "```",
                "",
            ]
        )
        if variant.notes:
            lines.append("**Notes**")
            lines.append("")
            lines.extend(f"- {note}" for note in variant.notes)
            lines.append("")
        if variant.source_ids:
            source_links = []
            for source_id in variant.source_ids:
                title, url = SOURCES[source_id]
                source_links.append(f"[{title}]({url})")
            lines.append("Sources: " + "; ".join(source_links))
            lines.append("")

    lines.extend(["## Sources", ""])
    for source_id, (title, url) in SOURCES.items():
        lines.append(f"- `{source_id}`: [{title}]({url})")
    lines.append("")
    return "\n".join(lines)


def render_einsum_snippets() -> str:
    lines = [
        '"""Generated Noam-style einsum snippets for attention variants.',
        "",
        "These snippets are shape documentation. They assume an `einsum`, `softmax`,",
        "`where`, `phi`, and `sqrt` binding in the surrounding framework.",
        '"""',
        "",
    ]
    for variant in variants():
        lines.append(f"# {variant.title}")
        lines.extend(variant.equations)
        lines.append("")
    return "\n".join(lines)


def write_outputs(out_dir: Path, make_png: bool = False) -> None:
    svg_dir = out_dir / "svg"
    png_dir = out_dir / "png"
    graph_dir = out_dir / "tensorgrad_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    if make_png:
        png_dir.mkdir(parents=True, exist_ok=True)

    svg_paths: dict[str, Path] = {}
    for variant in variants():
        svg_path = svg_dir / f"{variant.slug}.svg"
        svg_path.write_text(render_svg(variant), encoding="utf-8")
        # Validate the generated XML while the failure is close to the writer.
        ET.parse(svg_path)
        svg_paths[variant.slug] = svg_path

        graph_path = graph_dir / f"{variant.slug}.tg.txt"
        graph_path.write_text("\n".join(variant.tensorgrad_graph) + "\n", encoding="utf-8")

        if make_png:
            converter = shutil.which("rsvg-convert")
            if converter is None:
                raise RuntimeError("--png requested, but rsvg-convert is not on PATH")
            png_path = png_dir / f"{variant.slug}.png"
            subprocess.run([converter, str(svg_path), "-o", str(png_path)], check=True)

    (out_dir / "attention_variants.md").write_text(render_markdown(out_dir, svg_paths), encoding="utf-8")
    (out_dir / "einsum_snippets.py").write_text(render_einsum_snippets(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--png", action="store_true", help="Also render PNGs with rsvg-convert when available.")
    args = parser.parse_args()
    write_outputs(args.out_dir, make_png=args.png)
    print(f"Wrote attention variant report to {args.out_dir / 'attention_variants.md'}")


if __name__ == "__main__":
    main()
