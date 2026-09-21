### 全配置汇总（原始文件：`results/runs/<tag>.json`）

| 配置 | step ms ± std | tokens/s | alloc MiB | reserved MiB | nvidia-smi MiB | MFU | HFU |
|---|---|---|---|---|---|---|---|
| s512 b12 (6144 tok) bf16 flash | **OOM** | — | 22314 | 22322 | 22795 | — | — |
| s512 b12 (6144 tok) bf16 flash +ckpt | 557 ± 1 | 11022 | 20334 | 23532 | 24013 | 21.22% | 28.29% |
| s512 b16 (8192 tok) bf16 flash +ckpt | **OOM** | — | 20854 | 21370 | 21851 | — | — |
| s512 b1 (512 tok) bf16 flash | 148 ± 2 | 3454 | 11538 | 11878 | 12359 | 6.65% | 6.65% |
| s512 b2 (1024 tok) bf16 flash | 144 ± 2 | 7120 | 12310 | 12958 | 13439 | 13.71% | 13.71% |
| s512 b4 (2048 tok) bf16 flash | 178 ± 1 | 11536 | 16644 | 16682 | 17163 | 22.21% | 22.21% |
| s512 b4 (2048 tok) bf16 flash +ckpt | 226 ± 1 | 9079 | 11982 | 15560 | 16041 | 17.48% | 23.31% |
| s512 b6 (3072 tok) bf16 flash | 244 ± 0 | 12591 | 20972 | 21792 | 22273 | 24.24% | 24.24% |
| s512 b8 (4096 tok) bf16 flash | **OOM** | — | 22946 | 23166 | 23647 | — | — |
| s512 b8 (4096 tok) bf16 flash +ckpt | 366 ± 0 | 11197 | 15934 | 19732 | 20213 | 21.56% | 28.74% |
| s2048 b1 (2048 tok) bf16 flash +ckpt | 253 ± 3 | 8103 | 11982 | 15564 | 16045 | 21.11% | 28.14% |
| s512 b4 (2048 tok) bf16 flash +ckpt | 240 ± 0 | 8542 | 11982 | 15560 | 16041 | 16.45% | 21.93% |
| s2048 b1 (2048 tok) bf16 mem_efficient | 207 ± 0 | 9892 | 16645 | 16684 | 17165 | 25.77% | 25.77% |
| s512 b4 (2048 tok) bf16 mem_efficient | 182 ± 1 | 11231 | 16644 | 16682 | 17163 | 21.62% | 21.62% |
| s2048 b1 (2048 tok) fp32 mem_efficient | 420 ± 0 | 4875 | 18345 | 19000 | 19475 | 12.70% | 12.70% |
| s512 b4 (2048 tok) fp32 mem_efficient | 344 ± 2 | 5948 | 18344 | 18998 | 19473 | 11.45% | 11.45% |
| s2048 b1 (2048 tok) bf16 flash | 188 ± 0 | 10907 | 16645 | 16684 | 17165 | 28.41% | 28.41% |
| s512 b4 (2048 tok) bf16 flash | 178 ± 1 | 11486 | 16644 | 16682 | 17163 | 22.11% | 22.11% |
| s2048 b1 (2048 tok) bf16 flash +ckpt | 231 ± 4 | 8858 | 11982 | 15564 | 16045 | 23.07% | 30.76% |
| s512 b4 (2048 tok) bf16 flash +ckpt | 236 ± 6 | 8678 | 11982 | 15560 | 16041 | 16.71% | 22.28% |
| s2048 b1 (2048 tok) bf16 flash | 189 ± 0 | 10861 | 16645 | 16684 | 17165 | 28.29% | 28.29% |
| s2048 b1 (2048 tok) bf16 flash +ckpt | 243 ± 1 | 8426 | 11982 | 15564 | 16045 | 21.95% | 29.26% |
| s512 b4 (2048 tok) bf16 flash +ckpt | 227 ± 1 | 9024 | 11982 | 15560 | 16041 | 17.37% | 23.17% |
| s2048 b1 (2048 tok) bf16 flash +ckpt | 227 ± 1 | 9035 | 11982 | 15564 | 16045 | 23.53% | 31.38% |
| s512 b4 (2048 tok) bf16 flash +ckpt | 228 ± 3 | 8975 | 11982 | 15560 | 16041 | 17.28% | 23.04% |
| s2048 b1 (2048 tok) bf16 cudnn | 191 ± 1 | 10723 | 16645 | 16684 | 17169 | 27.93% | 27.93% |
| s512 b4 (2048 tok) bf16 cudnn | 179 ± 0 | 11457 | 16644 | 16682 | 17167 | 22.06% | 22.06% |
| s2048 b1 (2048 tok) bf16 flash | 188 ± 0 | 10899 | 16645 | 16684 | 17165 | 28.39% | 28.39% |
| s512 b4 (2048 tok) bf16 flash | 177 ± 2 | 11542 | 16644 | 16682 | 17163 | 22.22% | 22.22% |
| s2048 b1 (2048 tok) bf16 math | **OOM** | — | 22338 | 22666 | 23147 | — | — |
| s512 b4 (2048 tok) bf16 math | 228 ± 0 | 8966 | 19328 | 19448 | 19929 | 17.26% | 17.26% |
| s2048 b1 (2048 tok) bf16 mem_efficient | 206 ± 0 | 9927 | 16645 | 16684 | 17165 | 25.86% | 25.86% |
| s512 b4 (2048 tok) bf16 mem_efficient | 182 ± 0 | 11245 | 16644 | 16682 | 17163 | 21.65% | 21.65% |
| s1024 b2 (2048 tok) bf16 flash | 180 ± 0 | 11351 | 16644 | 16682 | 17163 | 24.42% | 24.42% |
| s2048 b1 (2048 tok) bf16 flash | 188 ± 0 | 10907 | 16645 | 16684 | 17165 | 28.41% | 28.41% |
| s256 b8 (2048 tok) bf16 flash | 176 ± 0 | 11623 | 16643 | 16682 | 17163 | 21.06% | 21.06% |
| s512 b4 (2048 tok) bf16 flash | 177 ± 0 | 11560 | 16644 | 16682 | 17163 | 22.26% | 22.26% |
| s1024 b4 (4096 tok) bf16 flash | **OOM** | — | 22946 | 23166 | 21273 | — | — |
| s2048 b2 (4096 tok) bf16 flash | **OOM** | — | 22947 | 23166 | 21273 | — | — |
| s512 b8 (4096 tok) bf16 flash | **OOM** | — | 22946 | 23166 | 21273 | — | — |

### GEMM 峰值扫描（`results/gemm_peak.json`）

| N (N×N×N) | bf16 TFLOPS | fp16 TFLOPS | fp32 TFLOPS |
|---|---|---|---|
| 1024 | 120.5 | 99.9 | 40.7 |
| 2048 | 155.6 | 148.8 | 48.1 |
| 3072 | 145.8 | 153.7 | 45.1 |
| 4096 | 138.4 | 142.9 | 46.6 |
| 6144 | 144.8 | 156.4 | 46.4 |
| 8192 | 146.9 | 128.0 | 43.4 |
| 12288 | 143.4 | 139.6 | OOM/跳过 |
| 16384 | 146.4 | 136.3 | OOM/跳过 |

| 模型真实 GEMM 形状 | bf16 TFLOPS |
|---|---|
| tok=2048 qkv_q [2048×1024]×[1024×2048] | 125.6 |
| tok=2048 kv [2048×1024]×[1024×1024] | 117.9 |
| tok=2048 o_proj [2048×2048]×[2048×1024] | 123.0 |
| tok=2048 gate_up [2048×1024]×[1024×3072] | 117.6 |
| tok=2048 down [2048×3072]×[3072×1024] | 155.4 |
| tok=2048 lm_head [2048×1024]×[1024×151936] | 105.6 |
| tok=8192 qkv_q [8192×1024]×[1024×2048] | 150.4 |
| tok=8192 kv [8192×1024]×[1024×1024] | 158.1 |
| tok=8192 o_proj [8192×2048]×[2048×1024] | 163.3 |
| tok=8192 gate_up [8192×1024]×[1024×3072] | 149.4 |
| tok=8192 down [8192×3072]×[3072×1024] | 138.4 |
| tok=8192 lm_head [8192×1024]×[1024×151936] | 149.7 |
| tok=16384 qkv_q [16384×1024]×[1024×2048] | 150.3 |
| tok=16384 kv [16384×1024]×[1024×1024] | 142.7 |
| tok=16384 o_proj [16384×2048]×[2048×1024] | 141.8 |
| tok=16384 gate_up [16384×1024]×[1024×3072] | 156.0 |
| tok=16384 down [16384×3072]×[3072×1024] | 134.5 |
| tok=16384 lm_head [16384×1024]×[1024×151936] | 149.4 |

| 带宽模式 | GB/s | 占 1008 GB/s |
|---|---|---|
| copy | 866 | 85.9% |
| add | 918 | 91.0% |
| reduce | 948 | 94.1% |

### 冷启动逐步耗时（`results/step_curve.json`，前 20 步）

| step | ms | 相对稳态 |
|---|---|---|
| 0 | 1595.2 | +754.5% |
| 1 | 187.4 | +0.4% |
| 2 | 186.8 | +0.1% |
| 3 | 186.6 | -0.1% |
| 4 | 186.4 | -0.2% |
| 5 | 186.2 | -0.3% |
| 6 | 186.8 | +0.0% |
| 7 | 186.3 | -0.2% |
| 8 | 186.6 | -0.0% |
| 9 | 187.3 | +0.3% |
| 10 | 186.3 | -0.2% |
| 11 | 187.2 | +0.3% |
| 12 | 186.7 | +0.0% |
| 13 | 186.7 | +0.0% |
| 14 | 186.8 | +0.1% |
| 15 | 186.7 | +0.0% |
| 16 | 186.6 | -0.0% |
| 17 | 186.4 | -0.1% |
| 18 | 186.7 | -0.0% |
| 19 | 186.4 | -0.2% |

### profiler · ref_s2048_b1（`results/profiles/ref_s2048_b1.json`）

SDPA 实测后端 kernel：`attention_flash`

| 算子类别 | 占 CUDA 时间 |
|---|---|
| optimizer | 28.81% |
| gemm | 28.66% |
| elementwise | 26.23% |
| attention_flash | 8.58% |
| softmax | 5.76% |
| reduction | 1.90% |
| index_embed | 0.05% |
| other | 0.00% |

| # | 算子 | 类别 | self CUDA ms | 占比 | 调用数 |
|---|---|---|---|---|---|
| 1 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn` | gemm | 53.61 | 9.82% | 591 |
| 2 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 31.84 | 5.83% | 30 |
| 3 | `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_` | attention_flash | 30.92 | 5.67% | 84 |
| 4 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 30.74 | 5.63% | 15 |
| 5 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | gemm | 28.93 | 5.30% | 504 |
| 6 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_` | gemm | 25.08 | 4.60% | 252 |
| 7 | `void at::native::vectorized_elementwise_kernel<4, at::nati` | elementwise | 24.89 | 4.56% | 1857 |
| 8 | `void at::native::elementwise_kernel<128, 2, at::native::gp` | elementwise | 23.58 | 4.32% | 2028 |
| 9 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 23.44 | 4.30% | 15 |
| 10 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 23.44 | 4.30% | 15 |

### profiler · ref_s512_b4（`results/profiles/ref_s512_b4.json`）

SDPA 实测后端 kernel：`attention_flash`

| 算子类别 | 占 CUDA 时间 |
|---|---|
| optimizer | 30.62% |
| gemm | 30.43% |
| elementwise | 27.81% |
| softmax | 6.10% |
| attention_flash | 2.99% |
| reduction | 2.00% |
| index_embed | 0.05% |
| other | 0.00% |

| # | 算子 | 类别 | self CUDA ms | 占比 | 调用数 |
|---|---|---|---|---|---|
| 1 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn` | gemm | 53.66 | 10.44% | 591 |
| 2 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 31.86 | 6.20% | 30 |
| 3 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 30.74 | 5.98% | 15 |
| 4 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | gemm | 28.72 | 5.59% | 504 |
| 5 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_` | gemm | 25.15 | 4.89% | 252 |
| 6 | `void at::native::vectorized_elementwise_kernel<4, at::nati` | elementwise | 24.85 | 4.84% | 1857 |
| 7 | `void at::native::elementwise_kernel<128, 2, at::native::gp` | elementwise | 23.47 | 4.57% | 2028 |
| 8 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 23.47 | 4.57% | 15 |
| 9 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 23.44 | 4.56% | 15 |
| 10 | `void at::native::unrolled_elementwise_kernel<at::native::d` | elementwise | 22.29 | 4.34% | 1524 |

### profiler · prof_ckpt_s2048_b1（`results/profiles/prof_ckpt_s2048_b1.json`）

SDPA 实测后端 kernel：`attention_flash`

| 算子类别 | 占 CUDA 时间 |
|---|---|
| gemm | 29.49% |
| elementwise | 28.45% |
| optimizer | 25.26% |
| attention_flash | 9.79% |
| softmax | 5.03% |
| reduction | 1.93% |
| index_embed | 0.05% |
| other | 0.00% |

| # | 算子 | 类别 | self CUDA ms | 占比 | 调用数 |
|---|---|---|---|---|---|
| 1 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn` | gemm | 81.91 | 13.15% | 1095 |
| 2 | `void at::native::vectorized_elementwise_kernel<4, at::nati` | elementwise | 36.98 | 5.94% | 3201 |
| 3 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 31.87 | 5.12% | 30 |
| 4 | `void at::native::elementwise_kernel<128, 2, at::native::gp` | elementwise | 31.35 | 5.03% | 2868 |
| 5 | `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_` | attention_flash | 30.87 | 4.96% | 84 |
| 6 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 30.75 | 4.94% | 15 |
| 7 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | gemm | 28.79 | 4.62% | 504 |
| 8 | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_trai` | attention_flash | 28.47 | 4.57% | 168 |
| 9 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_` | gemm | 24.73 | 3.97% | 252 |
| 10 | `void at::native::unrolled_elementwise_kernel<at::native::d` | elementwise | 24.19 | 3.88% | 1692 |

### profiler · prof_fp32_s2048_b1（`results/profiles/prof_fp32_s2048_b1.json`）

SDPA 实测后端 kernel：`attention_mem_efficient`

| 算子类别 | 占 CUDA 时间 |
|---|---|
| gemm | 45.65% |
| attention_mem_efficient | 28.13% |
| optimizer | 12.76% |
| elementwise | 9.85% |
| softmax | 2.55% |
| reduction | 1.05% |
| index_embed | 0.02% |
| other | 0.00% |

| # | 算子 | 类别 | self CUDA ms | 占比 | 调用数 |
|---|---|---|---|---|---|
| 1 | `fmha_cutlassB_f32_aligned_64x64_k128_sm80(PyTorchMemEffAtt` | attention_mem_efficient | 264.00 | 21.39% | 84 |
| 2 | `ampere_sgemm_128x64_tn` | gemm | 165.83 | 13.44% | 588 |
| 3 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_nt` | gemm | 143.20 | 11.60% | 423 |
| 4 | `ampere_sgemm_128x64_nn` | gemm | 98.79 | 8.01% | 420 |
| 5 | `fmha_cutlassF_f32_aligned_64x128_rf_sm80(PyTorchMemEffAtte` | attention_mem_efficient | 83.16 | 6.74% | 84 |
| 6 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn` | gemm | 79.83 | 6.47% | 171 |
| 7 | `ampere_sgemm_128x128_tn` | gemm | 55.07 | 4.46% | 3 |
| 8 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 31.87 | 2.58% | 30 |
| 9 | `void at::native::elementwise_kernel<128, 2, at::native::gp` | elementwise | 31.39 | 2.54% | 2364 |
| 10 | `void at::native::(anonymous namespace)::multi_tensor_apply` | optimizer | 30.75 | 2.49% | 15 |

### bound 判定（`results/bound_analysis.json`）

- E1: GEMM 家族占 CUDA 时间 37.2% -> 非 GEMM 占多数，memory 侧主导
- E2: GEMM 类算子内部达到实测峰值的 90.3% -> 这些 kernel 自身接近算力上限
- E3: s512/b=4 时固定开销占单步 64.4% -> 固定开销显著，小 batch 下是 launch/访存 主导
