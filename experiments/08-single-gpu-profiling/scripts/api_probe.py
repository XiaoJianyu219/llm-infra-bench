import torch, transformers, inspect, json
print("transformers", transformers.__version__, "torch", torch.__version__)
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained("/root/autodl-tmp/models/Qwen3-0.6B")
print("cfg type", type(cfg).__name__)
print("from_pretrained sig has attn_implementation:",
      "attn_implementation" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters)
print("cfg attrs:", [a for a in dir(cfg) if "attn" in a.lower() or "impl" in a.lower()])
# model on meta/cpu, no GPU
m = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/models/Qwen3-0.6B", dtype=torch.float32)
print("model class", type(m).__name__)
print("has gradient_checkpointing_enable:", hasattr(m,"gradient_checkpointing_enable"))
print("gce sig:", inspect.signature(m.gradient_checkpointing_enable))
print("config._attn_implementation =", getattr(m.config,"_attn_implementation",None))
tot = sum(p.numel() for p in m.parameters())
emb = m.get_input_embeddings().weight.numel()
print("PARAM total", tot, "embed", emb, "non-embed", tot-emb)
print("tied:", m.config.tie_word_embeddings, "lm_head is emb:", m.lm_head.weight.data_ptr()==m.get_input_embeddings().weight.data_ptr())
# per-layer breakdown
l0 = m.model.layers[0]
print("layer0 params", sum(p.numel() for p in l0.parameters()))
for n,p in l0.named_parameters(): print("   ", n, tuple(p.shape), p.numel())
# sdpa kernel API
import torch.nn.attention as A
print("sdpa_kernel:", hasattr(A,"sdpa_kernel"), "SDPBackend:", [x for x in dir(A.SDPBackend) if not x.startswith('_')])
print("fwd sig params:", list(inspect.signature(m.forward).parameters)[:12])
