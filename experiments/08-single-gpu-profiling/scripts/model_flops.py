"""Day08 4.2 分子：transformer 训练 FLOPs 的手推实现。

6N 的来历
---------
一个 [in,out] 的线性层，对每个 token 做 y = xW：
  forward            = in*out 次乘加 = 2*in*out FLOPs = 2 * (该层参数量)
  backward dgrad     dL/dx = dL/dy W^T        再 2*参数量
  backward wgrad     dL/dW = x^T dL/dy        再 2*参数量
  => backward 是 forward 的 2 倍，fwd+bwd 合计 6 * 参数量
对所有（非 embedding 的）权重求和，就是每 token 6N。

attention 的两个 batched matmul 不含参数，单列：
  QK^T   per layer per token: 2 * s * d_attn        (s 个 key，每个 d_attn 维)
  A·V    per layer per token: 2 * s * d_attn
  fwd 合计 4*L*s*d_attn，×3 得 fwd+bwd = 12*L*s*d_attn
注意 d_attn = n_heads * head_dim，Qwen3-0.6B 里是 2048，不等于 hidden=1024。
"""

def model_flops(cfg, seq_len, include_lm_head=False):
    L   = cfg["num_hidden_layers"]
    h   = cfg["hidden_size"]
    hd  = cfg["head_dim"]
    nh  = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    ffn = cfg["intermediate_size"]
    V   = cfg["vocab_size"]
    d_attn = nh * hd
    d_kv   = nkv * hd

    per_layer_params = (h*d_attn + h*d_kv + h*d_kv + d_attn*h   # q k v o
                        + 3*h*ffn                                # gate up down
                        + 2*hd + 2*h)                            # q/k norm + 2 rmsnorm
    N_non_emb = L*per_layer_params + h                           # + final norm
    N_emb = V*h

    dense = 6*N_non_emb                     # 每 token
    attn  = 12*L*seq_len*d_attn             # 每 token
    head  = 6*N_emb if include_lm_head else 0
    return dict(d_attn=d_attn, N_non_emb=N_non_emb, N_emb=N_emb,
                per_layer_params=per_layer_params,
                dense_per_token=dense, attn_per_token=attn,
                lm_head_per_token=head,
                total_per_token=dense+attn+head,
                attn_share=attn/(dense+attn+head),
                lm_head_share=(head/(dense+attn+head)) if head else 0.0)

QWEN3_06B = dict(num_hidden_layers=28, hidden_size=1024, head_dim=128,
                 num_attention_heads=16, num_key_value_heads=8,
                 intermediate_size=3072, vocab_size=151936)

if __name__ == "__main__":
    for s in (512, 2048):
        for inc in (False, True):
            f = model_flops(QWEN3_06B, s, inc)
            tag = "含lm_head" if inc else "任务书口径"
            print(f"s={s:5d} {tag:10s} 6N={f['dense_per_token']/1e9:.3f}G "
                  f"attn={f['attn_per_token']/1e9:.3f}G "
                  f"lm_head={f['lm_head_per_token']/1e9:.3f}G "
                  f"total={f['total_per_token']/1e9:.3f}G "
                  f"attn占{f['attn_share']*100:.1f}% lm_head占{f['lm_head_share']*100:.1f}%")
    f = model_flops(QWEN3_06B, 2048)
    print("N_non_emb =", f["N_non_emb"], " N_emb =", f["N_emb"],
          " per_layer =", f["per_layer_params"], " d_attn =", f["d_attn"])
