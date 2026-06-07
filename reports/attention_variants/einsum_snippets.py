"""Generated Noam-style einsum snippets for attention variants.

These snippets are shape documentation. They assume an `einsum`, `softmax`,
`where`, `phi`, and `sqrt` binding in the surrounding framework.
"""

# Multi-Head Attention
query_BLHK = einsum("BLD,DHK->BLHK", input_BLD, w_q_DHK)
key_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_k_DHK)
value_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_v_DHK)
logits_BHLT = einsum("BLHK,BTHK->BHLT", query_BLHK, key_BTHK) / sqrt(K)
weights_BHLT = softmax(logits_BHLT, dim="T")
wtd_values_BLHK = einsum("BHLT,BTHK->BLHK", weights_BHLT, value_BTHK)
out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)

# Grouped Multi-Query / Grouped-Query Attention
# H = G * C
query_BLGCK = einsum("BLD,DGCK->BLGCK", input_BLD, w_q_DGCK)
key_BTGK = einsum("BTD,DGK->BTGK", memory_BTD, w_k_DGK)
value_BTGK = einsum("BTD,DGK->BTGK", memory_BTD, w_v_DGK)
logits_BLGCT = einsum("BLGCK,BTGK->BLGCT", query_BLGCK, key_BTGK) / sqrt(K)
weights_BLGCT = softmax(logits_BLGCT, dim="T")
wtd_values_BLGCK = einsum("BLGCT,BTGK->BLGCK", weights_BLGCT, value_BTGK)
out_BLD = einsum("BLGCK,GCKD->BLD", wtd_values_BLGCK, w_o_GCKD)

# All-to-All Attention
query_BLQK = einsum("BLD,DQK->BLQK", input_BLD, w_q_DQK)
key_BTGK = einsum("BTD,DGK->BTGK", memory_BTD, w_k_DGK)
value_BTGK = einsum("BTD,DGK->BTGK", memory_BTD, w_v_DGK)
logits_BLQGT = einsum("BLQK,BTGK->BLQGT", query_BLQK, key_BTGK) / sqrt(K)
weights_BLQGT = softmax(logits_BLQGT, dim=("G", "T"))
wtd_values_BLQK = einsum("BLQGT,BTGK->BLQK", weights_BLQGT, value_BTGK)
out_BLD = einsum("BLQK,QKD->BLD", wtd_values_BLQK, w_o_QKD)

# Multi-Query Attention
query_BLHK = einsum("BLD,DHK->BLHK", input_BLD, w_q_DHK)
key_BTK = einsum("BTD,DK->BTK", memory_BTD, w_k_DK)
value_BTK = einsum("BTD,DK->BTK", memory_BTD, w_v_DK)
logits_BHLT = einsum("BLHK,BTK->BHLT", query_BLHK, key_BTK) / sqrt(K)
weights_BHLT = softmax(logits_BHLT, dim="T")
wtd_values_BLHK = einsum("BHLT,BTK->BLHK", weights_BHLT, value_BTK)
out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)

# Sliding-Window Attention
query_BLHK = einsum("BLD,DHK->BLHK", input_BLD, w_q_DHK)
key_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_k_DHK)
value_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_v_DHK)
logits_BHLT = einsum("BLHK,BTHK->BHLT", query_BLHK, key_BTHK) / sqrt(K)
logits_BHLT = where(window_mask_LT, logits_BHLT, -inf)
weights_BHLT = softmax(logits_BHLT, dim="T")
wtd_values_BLHK = einsum("BHLT,BTHK->BLHK", weights_BHLT, value_BTHK)
out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)

# BigBird-Style Block Sparse Attention
query_BLHK = einsum("BLD,DHK->BLHK", input_BLD, w_q_DHK)
key_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_k_DHK)
value_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_v_DHK)
logits_BHLT = einsum("BLHK,BTHK->BHLT", query_BLHK, key_BTHK) / sqrt(K)
sparse_mask_LT = window_LT | random_LT | global_LT
logits_BHLT = where(sparse_mask_LT, logits_BHLT, -inf)
weights_BHLT = softmax(logits_BHLT, dim="T")
wtd_values_BLHK = einsum("BHLT,BTHK->BLHK", weights_BHLT, value_BTHK)
out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)

# Linear Attention
query_BLHF = phi(einsum("BLD,DHF->BLHF", input_BLD, w_q_DHF))
key_BTHF = phi(einsum("BTD,DHF->BTHF", memory_BTD, w_k_DHF))
value_BTHK = einsum("BTD,DHK->BTHK", memory_BTD, w_v_DHK)
kv_BHFK = einsum("BTHF,BTHK->BHFK", key_BTHF, value_BTHK)
k_sum_BHF = einsum("BTHF->BHF", key_BTHF)
denom_BLH = einsum("BLHF,BHF->BLH", query_BLHF, k_sum_BHF)
wtd_values_BLHK = einsum("BLHF,BHFK->BLHK", query_BLHF, kv_BHFK) / denom_BLH[..., None]
out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)

# Multi-Head Latent Attention
q_latent_BLR = einsum("BLD,DR->BLR", input_BLD, w_q_down_DR)
query_BLHK = einsum("BLR,RHK->BLHK", q_latent_BLR, w_q_up_RHK)
kv_latent_BTR = einsum("BTD,DR->BTR", memory_BTD, w_kv_down_DR)
key_BTHK = einsum("BTR,RHK->BTHK", kv_latent_BTR, w_k_up_RHK)
value_BTHK = einsum("BTR,RHK->BTHK", kv_latent_BTR, w_v_up_RHK)
logits_BHLT = einsum("BLHK,BTHK->BHLT", query_BLHK, key_BTHK) / sqrt(K)
weights_BHLT = softmax(logits_BHLT, dim="T")
wtd_values_BLHK = einsum("BHLT,BTHK->BLHK", weights_BHLT, value_BTHK)
out_BLD = einsum("BLHK,HKD->BLD", wtd_values_BLHK, w_o_HKD)
