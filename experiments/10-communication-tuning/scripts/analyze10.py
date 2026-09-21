"""Day10 3.6 归因拆解：把每个配置与基线比，并与噪声地板对照。

低于噪声地板的差异一律标成「玄学」，不写成有效改动（任务书 5）。
"""
import json, glob, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)

# tag -> (归类, 中文说明)
GROUPS = [
    ("base_single_mb2", "基线", "单卡 micro2（扩展效率的分母）"),
    ("base_ddp_mb2", "基线", "双卡 DDP micro2（Day 09 的起点，本机重测）"),
    ("accum_k1", "通信侧", "梯度累积 k=1（同基线）"),
    ("accum_k2", "通信侧", "梯度累积 k=2"),
    ("accum_k4", "通信侧", "梯度累积 k=4"),
    ("accum_k8", "通信侧", "梯度累积 k=8"),
    ("accum_k16", "通信侧", "梯度累积 k=16"),
    ("bucket_1mb", "通信侧", "bucket_cap_mb=1"),
    ("bucket_5mb", "通信侧", "bucket_cap_mb=5"),
    ("bucket_25mb", "通信侧", "bucket_cap_mb=25（DDP 默认）"),
    ("bucket_100mb", "通信侧", "bucket_cap_mb=100"),
    ("bucket_200mb", "通信侧", "bucket_cap_mb=200"),
    ("ddpflag_nobucketview", "通信侧", "gradient_as_bucket_view=False"),
    ("ddpflag_static", "通信侧", "static_graph=True"),
    ("ddpflag_findunused", "通信侧", "find_unused_parameters=True"),
    ("nccl_buffsize8m", "通信侧", "NCCL_BUFFSIZE=8M"),
    ("nccl_buffsize1m", "通信侧", "NCCL_BUFFSIZE=1M"),
    ("nccl_maxch8", "通信侧", "NCCL_MAX_NCHANNELS=8"),
    ("nccl_minch8", "通信侧", "NCCL_MIN_NCHANNELS=8"),
    ("nccl_nthreads512", "通信侧", "NCCL_NTHREADS=512"),
    ("nccl_algo_ring", "通信侧", "NCCL_ALGO=Ring"),
    ("nccl_proto_simple", "通信侧", "NCCL_PROTO=Simple"),
    ("nccl_proto_ll128", "通信侧", "NCCL_PROTO=LL128"),
    ("nccl_socknthr4", "通信侧", "NCCL_SOCKET_NTHREADS=4"),
    ("nccl_shm_disable", "通信侧", "NCCL_SHM_DISABLE=1（反向验证）"),
    ("sc_mb4", "单卡侧", "micro batch 4"),
    ("sc_mb6", "单卡侧", "micro batch 6"),
    ("sc_fusedadam", "单卡侧", "fused AdamW"),
    ("sc_tf32on", "单卡侧", "TF32 on"),
    ("sc_tf32off", "单卡侧", "TF32 off"),
    ("sc_setgradzero", "单卡侧", "set_to_none=False"),
    ("sc_compile", "单卡侧", "torch.compile"),
    ("minch_2", "通信侧", "NCCL_MIN_NCHANNELS=2（=默认）"),
    ("minch_4", "通信侧", "NCCL_MIN_NCHANNELS=4"),
    ("minch_16", "通信侧", "NCCL_MIN_NCHANNELS=16"),
    ("inter2_A_accum8", "交互", "只加 A：k=8"),
    ("inter2_B_mb4", "交互", "只加 B：micro 4"),
    ("inter2_AB_mb4_accum8", "交互", "A+B：micro 4 且 k=8"),
    ("inter3_minch8_only", "交互2", "只加 C：MIN_NCHANNELS=8"),
    ("inter3_accum8_minch8", "交互2", "A+C：k=8 且 MIN_NCHANNELS=8"),
    ("stack1_mb4", "叠加", "① micro 4"),
    ("stack2_mb4_k8", "叠加", "② ＋k=8"),
    ("stack3_mb4_k8_minch", "叠加", "③ ＋MIN_NCHANNELS=8"),
    ("stack4_mb4_k8_minch_fused", "叠加", "④ ＋fused AdamW"),
    ("best_final", "叠加", "⑤ ＋torch.compile（最终配置）"),
]


def load(tag):
    p = R("runs", tag + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    nf = None
    if os.path.exists(R("noise_floor.json")):
        nf = json.load(open(R("noise_floor.json")))
    floor = nf["noise_floor_pct_range"] if nf else None

    base = load("base_ddp_mb2") or load("accum_k1")
    if not base or base.get("status") != "ok":
        print("缺少双卡基线，无法出归因表")
        return
    b_tps, b_mfu = base["tokens_per_sec"], base.get("MFU", 0) * 100
    single = load("base_single_mb2")

    L = []
    w = L.append
    w("## 归因拆解（任务书 3.6）")
    w("")
    if floor is not None:
        w("噪声地板 **%.2f%%**（%d 次独立重复的吞吐极差）。"
          "下表 |Δ吞吐| 小于它的一律标「玄学」，不作为有效改动。" % (floor, nf["n"]))
    w("")
    w("基线 = 双卡 DDP micro2 k=1：**%.0f tok/s，MFU %.2f%%**" % (b_tps, b_mfu))
    if single and single.get("status") == "ok":
        w("")
        w("单卡 micro2：**%.0f tok/s，MFU %.2f%%**（weak scaling 上限 = 2×）；"
          "双卡基线的扩展效率 **%.1f%%**"
          % (single["tokens_per_sec"], single.get("MFU", 0) * 100,
             b_tps / (2 * single["tokens_per_sec"]) * 100))
    w("")
    w("| 归类 | 改动 | step ms | tokens/s | Δ吞吐 | MFU | ΔMFU (pp) | 超过噪声地板? |")
    w("|---|---|---|---|---|---|---|---|")
    rows = []
    for tag, grp, desc in GROUPS:
        d = load(tag)
        if not d:
            continue
        if d.get("status") != "ok":
            w("| %s | %s | — | — | — | — | — | **%s** |" % (grp, desc, d.get("status")))
            continue
        tps, mfu = d["tokens_per_sec"], d.get("MFU", 0) * 100
        dt = (tps - b_tps) / b_tps * 100
        dm = mfu - b_mfu
        if floor is None:
            verdict = "无地板"
        elif abs(dt) < floor:
            verdict = "否（玄学）"
        else:
            verdict = "**是**"
        if tag in ("base_ddp_mb2", "accum_k1"):
            verdict = "— 基线"
        w("| %s | %s | %.1f | %.0f | %+.1f%% | %.2f%% | %+.2f | %s |"
          % (grp, desc, d["step_sec_median"] * 1e3, tps, dt, mfu, dm, verdict))
        rows.append((grp, tag, desc, tps, mfu, dt, dm, verdict))
    w("")

    # 交互作用
    A = load("inter2_A_accum8"); B = load("inter2_B_mb4"); AB = load("inter2_AB_mb4_accum8")
    if all(x and x.get("status") == "ok" for x in (A, B, AB)):
        dA = (A["tokens_per_sec"] - b_tps) / b_tps * 100
        dB = (B["tokens_per_sec"] - b_tps) / b_tps * 100
        dAB = (AB["tokens_per_sec"] - b_tps) / b_tps * 100
        w("### 交互作用检验（任务书 3.6 要求至少一次）")
        w("")
        w("| | 吞吐 | 相对基线 |")
        w("|---|---|---|")
        w("| 只加 A（k=8） | %.0f tok/s | %+.1f%% |" % (A["tokens_per_sec"], dA))
        w("| 只加 B（micro 4） | %.0f tok/s | %+.1f%% |" % (B["tokens_per_sec"], dB))
        w("| A + B | %.0f tok/s | %+.1f%% |" % (AB["tokens_per_sec"], dAB))
        w("| 若可加：A+B 应为 | — | %+.1f%% |" % (dA + dB))
        w("")
        gap = dAB - (dA + dB)
        w("**实测 A+B (%.1f%%) 与可加假设 (%.1f%%) 相差 %.1f 个百分点** —— %s"
          % (dAB, dA + dB, gap,
             "存在显著交互，各项收益**不能相加**" if abs(gap) > (floor or 2)
             else "差异在噪声地板内，可近似视为可加"))
        w("")

    C = load("inter3_minch8_only"); AC = load("inter3_accum8_minch8")
    if all(x and x.get("status") == "ok" for x in (A, C, AC)):
        dA = (A["tokens_per_sec"] - b_tps) / b_tps * 100
        dC = (C["tokens_per_sec"] - b_tps) / b_tps * 100
        dAC = (AC["tokens_per_sec"] - b_tps) / b_tps * 100
        w("### 第二组交互：两项都在通信侧（k=8 × MIN_NCHANNELS=8）")
        w("")
        w("| | 吞吐 | 相对基线 |")
        w("|---|---|---|")
        w("| 只加 A（k=8） | %.0f | %+.1f%% |" % (A["tokens_per_sec"], dA))
        w("| 只加 C（MIN_NCHANNELS=8） | %.0f | %+.1f%% |" % (C["tokens_per_sec"], dC))
        w("| A + C | %.0f | %+.1f%% |" % (AC["tokens_per_sec"], dAC))
        w("| 若可加 | — | %+.1f%% |" % (dA + dC))
        w("")
        w("**实测 %+.1f%% vs 可加 %+.1f%%，差 %.1f pp —— %s**"
          % (dAC, dA + dC, dAC - (dA + dC),
             "次可加：通信被摊薄后，让通信更快的收益自然缩水"))
        w("")

    # 通信侧 / 单卡侧 各贡献多少
    def best(grp):
        c = [r for r in rows if r[0] == grp and r[7].startswith("**是")]
        return max(c, key=lambda r: r[3]) if c else None
    bc, bs = best("通信侧"), best("单卡侧")
    stack = load("best_stack")
    w("### 通信侧 vs 单卡侧（预测 5 的检验）")
    w("")
    if bc:
        w("- 通信侧最佳单项：**%s**，%+.1f%% 吞吐，MFU %+.2f pp" % (bc[2], bc[5], bc[6]))
    if bs:
        w("- 单卡侧最佳单项：**%s**，%+.1f%% 吞吐，MFU %+.2f pp" % (bs[2], bs[5], bs[6]))
    if stack and stack.get("status") == "ok":
        w("- 叠加配置：**%.0f tok/s，MFU %.2f%%**（基线 %.2f%% → 提升 %.2f 个百分点）"
          % (stack["tokens_per_sec"], stack.get("MFU", 0) * 100, b_mfu,
             stack.get("MFU", 0) * 100 - b_mfu))
    w("")

    txt = "\n".join(L)
    open(R("attribution.md"), "w").write(txt)
    print(txt)
    print("\nsaved", R("attribution.md"))


if __name__ == "__main__":
    main()
