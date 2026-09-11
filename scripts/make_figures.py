"""Render the project's result figures as PNGs.

Reads results/viz_data.json (itself rebuilt from results/*.jsonl by
scripts/build_viz_data.py) and writes docs/figures/*.png.

Colour follows the validated reference palette by ROLE, never by rank:
  slot 1 blue #2a78d6 = the subject / "after"
  slot 2 orange #eb6834 = the comparison series
  red #e34948 = negative polarity (diverging only)
  gray #898781 = de-emphasised context
  blue ramp = sequential magnitude
Diverging colour carries SIGN, categorical colour carries IDENTITY. Cell and
bar values are always printed directly, so colour never carries data alone.

Usage (from project root):
  .venv/Scripts/python scripts/make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

VIZ = "results/viz_data.json"
OUTDIR = os.path.join("docs", "figures")

# ── 参考调色板（按角色引用，非按排名） ────────────────────────────────────
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
S1, S2, GRAY, NEG = "#2a78d6", "#eb6834", "#898781", "#e34948"
BAND = "#f0efec"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", RAMP)

DPI = 200


def setup():
    for name in ("Microsoft YaHei", "SimHei", "DengXian"):
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })


# ── 文字对比度：填充色之上选墨或白 ────────────────────────────────────────
def _lum(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def ink_on(hex_color):
    return "#ffffff" if _lum(hex_color) < 0.42 else INK


# ── 数据末端 4px 圆角、贴基线端直角的条形 ────────────────────────────────
def _path(x0, x1, y0, y1, r, side):
    r = max(0.0, min(r, (x1 - x0), (y1 - y0) / 2))
    v, c = [], []
    if side == "right":
        v += [(x0, y0), (x1 - r, y0)]; c += [Path.MOVETO, Path.LINETO]
        v += [(x1, y0), (x1, y0 + r)]; c += [Path.CURVE3, Path.CURVE3]
        v += [(x1, y1 - r)]; c += [Path.LINETO]
        v += [(x1, y1), (x1 - r, y1)]; c += [Path.CURVE3, Path.CURVE3]
        v += [(x0, y1)]; c += [Path.LINETO]
    elif side == "left":
        v += [(x1, y0), (x0 + r, y0)]; c += [Path.MOVETO, Path.LINETO]
        v += [(x0, y0), (x0, y0 + r)]; c += [Path.CURVE3, Path.CURVE3]
        v += [(x0, y1 - r)]; c += [Path.LINETO]
        v += [(x0, y1), (x0 + r, y1)]; c += [Path.CURVE3, Path.CURVE3]
        v += [(x1, y1)]; c += [Path.LINETO]
    else:  # up
        v += [(x0, y0), (x0, y1 - r)]; c += [Path.MOVETO, Path.LINETO]
        v += [(x0, y1), (x0 + r, y1)]; c += [Path.CURVE3, Path.CURVE3]
        v += [(x1 - r, y1)]; c += [Path.LINETO]
        v += [(x1, y1), (x1, y1 - r)]; c += [Path.CURVE3, Path.CURVE3]
        v += [(x1, y0)]; c += [Path.LINETO]
    v += [v[0]]; c += [Path.CLOSEPOLY]
    return Path(v, c)


def hbar(ax, base, end, y, h, color, r=0.05, z=3):
    x0, x1 = (base, end) if end >= base else (end, base)
    ax.add_patch(PathPatch(_path(x0, x1, y - h / 2, y + h / 2, r,
                                 "right" if end >= base else "left"),
                           facecolor=color, edgecolor="none", zorder=z))


def vbar(ax, x, base, end, w, color, r, z=3):
    y0, y1 = (base, end) if end >= base else (end, base)
    ax.add_patch(PathPatch(_path(x - w / 2, x + w / 2, y0, y1, r, "up"),
                           facecolor=color, edgecolor="none", zorder=z))


def strip(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        vis = s in keep
        ax.spines[s].set_visible(vis)
        if vis:
            ax.spines[s].set_linewidth(1)
            ax.spines[s].set_color(AXIS)


def titles(ax, main, sub=None, main_pt=30, sub_pt=8):
    """标题/副标题按 axes 的图坐标定位，避免与彼此及绘图区重叠。

    main_pt / sub_pt 是相对 axes 顶边的**点数**偏移（换算成图坐标后与图高无关，
    所以同一组数值在各张图上间距一致）；有子图标题时调大，给它让位。
    """
    p = ax.get_position()
    fig = ax.figure
    h_in = fig.get_size_inches()[1]
    dp = lambda pt: pt / 72.0 / h_in          # 点 → 图坐标
    fig.text(p.x0, p.y1 + dp(main_pt), main, fontsize=13.5, color=INK,
             va="bottom", ha="left")
    if sub:
        fig.text(p.x0, p.y1 + dp(sub_pt), sub, fontsize=9.5, color=INK2,
                 va="bottom", ha="left")


def legend(ax, entries, y=-0.13, **kw):
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                          markerfacecolor=c, markeredgecolor="none", label=t)
               for c, t in entries]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, y),
              frameon=False, fontsize=9.5, handletextpad=0.6, ncol=len(entries),
              labelcolor=INK2, **kw)


def finish(fig, name):
    fig.savefig(os.path.join(OUTDIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


def fmt_p(p):
    return ("%.0e" % p) if p < 0.001 else ("%.3f" % p)


# ── 图 1：四个基线 ───────────────────────────────────────────────────────
def fig_baselines(D):
    rows = D["baselines"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, (key, lab, cap) in zip(axes, [("em", "EM（精确匹配）", 0.55),
                                          ("f1", "F1（token 级）", 0.70)]):
        for i, r in enumerate(rows):
            y = len(rows) - 1 - i
            hl = r["name"] == "ReAct"
            hbar(ax, 0, r[key], y, 0.52, S1 if hl else GRAY)
            ax.text(r[key] + 0.012, y, "%.1f%%" % (r[key] * 100), va="center",
                    ha="left", fontsize=10, color=INK,
                    fontweight="bold" if hl else "normal", zorder=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["name"] for r in rows][::-1], fontsize=11, color=INK)
        for t in ax.get_yticklabels():
            if t.get_text() == "ReAct":
                t.set_fontweight("bold")
        ax.set_xlim(0, cap)
        ax.set_ylim(-0.65, len(rows) - 0.35)
        xt = [i / 10 for i in range(0, int(cap * 10) + 1)]
        ax.set_xticks(xt)
        ax.set_xticklabels(["%d%%" % round(v * 100) for v in xt], fontsize=9)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        strip(ax, ("left",))
        ax.set_title(lab, fontsize=11, color=INK, pad=10)
    axes[0].legend(handles=[plt.Line2D([], [], marker="s", linestyle="none",
                                       markersize=9, markerfacecolor=S1,
                                       markeredgecolor="none", label="ReAct（重点）"),
                            plt.Line2D([], [], marker="s", linestyle="none",
                                       markersize=9, markerfacecolor=GRAY,
                                       markeredgecolor="none", label="对照基线")],
                   loc="lower left", bbox_to_anchor=(0, -0.34), frameon=False,
                   fontsize=9.5, ncol=2, handletextpad=0.6, labelcolor=INK2)
    titles(axes[0], "四个基线：ReAct 的动作空间值多少",
           "ReAct 与 Act-only 动作空间完全相同，唯一区别是删掉 Thought —— "
           "两者相差 17.2pp，即「交错推理」本身的净贡献。",
           main_pt=48, sub_pt=26)
    finish(fig, "fig1_baselines.png")


# ── 图 2：消融 ΔEM 发散条形 + 噪声带 ────────────────────────────────────
def fig_ablation(D):
    arms = sorted([a for a in D["ablation"] if a["name"] != "ReAct"],
                  key=lambda a: a["d_em"])
    nf = D["meta"]["noise_floor_pp"] / 100
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    lim = 0.085
    ax.axvspan(-nf, nf, color=BAND, zorder=0)
    ax.text(0, -0.62, "实测噪声带 ±1.6pp", ha="center", va="center",
            fontsize=9, color=MUTED, zorder=2)
    for i, a in enumerate(arms):
        y = i
        hbar(ax, 0, a["d_em"], y, 0.56, S1 if a["d_em"] >= 0 else NEG)
        off = 0.0022 if a["d_em"] >= 0 else -0.0022
        ax.text(a["d_em"] + off, y, "%+.1fpp" % (a["d_em"] * 100), va="center",
                ha="left" if a["d_em"] >= 0 else "right", fontsize=10,
                color=INK, fontweight="bold", zorder=4)
        ax.text(-lim + 0.002, y, "$%.2f" % a["cost"], va="center", ha="left",
                fontsize=8.5, color=MUTED, zorder=2)
        if a["p"] < 0.05:
            ax.text(0.104, y, fmt_p(a["p"]), va="center", ha="left",
                    fontsize=8.5, color=INK2, zorder=2)
    ax.axvline(0.100, color=GRID, lw=1, zorder=1)
    ax.text(0.104, len(arms) - 0.3, "McNemar p", va="bottom", ha="left",
            fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a["name"] for a in arms], fontsize=11, color=INK)
    for t, a in zip(ax.get_yticklabels(), arms):
        if a["p"] < 0.05:
            t.set_fontweight("bold")
    ax.set_xlim(-lim, 0.152)
    ax.set_ylim(-0.75, len(arms) - 0.15)
    ticks = [-0.08, -0.04, 0, 0.04, 0.08]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["-8pp", "-4pp", "0", "+4pp", "+8pp"], fontsize=9.5)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    ax.tick_params(axis="both", length=0)
    strip(ax, ())
    legend(ax, [(S1, "正收益"), (NEG, "负收益"),
                (BAND, "实测噪声带 ±1.6pp（带内不可解释）")], y=-0.075)
    titles(ax, "12 条消融臂：相对 ReAct baseline 的 ΔEM",
           "以 ReAct 47.2% 为零线。灰带内（±1.6pp）的差值一律不可解释，"
           "带外看 McNemar 精确检验；左侧数字为单次 500 题成本。")
    finish(fig, "fig2_ablation.png")


# ── 图 3：成本 × 效果 ───────────────────────────────────────────────────
def fig_cost(D):
    final = next(a for a in D["ablation"] if a["name"] == "cotsc+content")
    pts = [{"name": a["name"], "em": a["em"], "cost": a["cost"]}
           for a in D["ablation"] if a["name"] != "ReAct"]
    pts.append({"name": "ReAct", "em": D["baseline_react"]["em"],
                "cost": D["baseline_react"]["cost"]})
    labels = {
        # cotsc 与 cotsc+content 只差 0.28 美元，居中标签会压到上面那个点，
        # 所以 cotsc 改为右挂。
        "cotsc+content": (0, 0.0095, "center"),
        "cotsc": (0.22, -0.003, "left"),
        "content": (0, 0.0095, "center"),
        "ReAct-SC": (0, 0.0095, "center"),
        "ReAct": (-0.16, 0.0, "right"),
    }
    cluster = [p for p in pts if p["name"] not in labels]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.add_patch(Rectangle((2.05, 0.414), 0.79, 0.083, facecolor="none",
                           edgecolor=GRID, lw=1, linestyle=(0, (4, 3)), zorder=1))
    for p in cluster:
        ax.scatter(p["cost"], p["em"], s=46, color=GRAY, zorder=2,
                   edgecolor=SURFACE, linewidth=1.6)
    for p in pts:
        if p["name"] not in labels:
            continue
        hl = p["name"] != "ReAct"
        ax.scatter(p["cost"], p["em"], s=95 if hl else 70, color=S1, zorder=3,
                   edgecolor=SURFACE, linewidth=1.8)
        dx, dy, ha = labels[p["name"]]
        ax.text(p["cost"] + dx, p["em"] + dy, p["name"], ha=ha, va="bottom",
                fontsize=10.5, color=INK, fontweight="bold", zorder=4)
    ax.text(3.02, 0.455, "8 条其余消融臂", ha="left", va="center", fontsize=9.5,
            color=MUTED, zorder=4)
    ax.text(3.02, 0.439, "$2.0–2.8 · EM 42–49%", ha="left", va="center",
            fontsize=8.5, color=MUTED, zorder=4)
    ax.set_xlim(-0.35, 12.9)
    ax.set_ylim(0.402, 0.572)
    yt = [0.42, 0.45, 0.48, 0.51, 0.54]
    ax.set_yticks(yt)
    ax.set_yticklabels(["%d%%" % round(v * 100) for v in yt], fontsize=9.5)
    ax.set_xticks(range(0, 13, 2))
    ax.set_xticklabels(["$%d" % v for v in range(0, 13, 2)], fontsize=9.5)
    ax.set_xlabel("单次 500 题成本（USD）", fontsize=10, color=INK2)
    ax.set_ylabel("EM", fontsize=10, color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    strip(ax, ())
    titles(ax, "成本 × 效果：投票投什么，差 5 倍成本",
           "左上角更好。最终系统（\\$%.2f / %.1f%%）支配 ReAct-SC（\\$%.2f / %.1f%%）——"
           "同为约 5pp 的提升，选「轻量 CoT 补样」而非「完整轨迹重跑」，成本差 5.3 倍。"
           % (final["cost"], final["em"] * 100, 11.27, 52.4))
    finish(fig, "fig3_cost_effect.png")


# ── 图 4：held-out 哑铃图 ───────────────────────────────────────────────
def fig_heldout(D):
    ho = D["heldout"]
    final = next(a for a in D["ablation"] if a["name"] == "cotsc+content")
    rows = [("调参集 off0（已调参）", D["baseline_react"]["em"], final["em"],
             500, final["p"]),
            ("held-out off500（从未调参）", ho["baseline"]["em"], ho["combo"]["em"],
             ho["paired_n"], ho["p"])]
    fig, ax = plt.subplots(figsize=(8.8, 2.8))
    for i, (lab, b, c, n, p) in enumerate(rows):
        y = 1 - i
        ax.plot([b, c], [y, y], color=S1, lw=3, solid_capstyle="round", zorder=2)
        ax.scatter([b], [y], s=110, color=GRAY, zorder=3, edgecolor=SURFACE, lw=2)
        ax.scatter([c], [y], s=110, color=S1, zorder=3, edgecolor=SURFACE, lw=2)
        ax.text(b - 0.005, y, "%.1f%%" % (b * 100), ha="right", va="center",
                fontsize=11, color=INK2)
        ax.text(c + 0.005, y, "%.1f%%" % (c * 100), ha="left", va="center",
                fontsize=11, color=INK, fontweight="bold")
        ax.text((b + c) / 2, y + 0.19, "%+.1fpp" % ((c - b) * 100), ha="center",
                fontsize=10.5, color=INK, fontweight="bold")
        ax.text((b + c) / 2, y - 0.27, "n = %d · McNemar p = %s" % (n, fmt_p(p)),
                ha="center", fontsize=9, color=MUTED)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=INK)
    ax.set_xlim(0.428, 0.552)
    ax.set_ylim(-0.6, 1.6)
    xt = [0.44, 0.46, 0.48, 0.50, 0.52, 0.54]
    ax.set_xticks(xt)
    ax.set_xticklabels(["%d%%" % round(v * 100) for v in xt], fontsize=9.5)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    strip(ax, ())
    legend(ax, [(GRAY, "ReAct baseline"), (S1, "cotsc + content")], y=-0.20)
    titles(ax, "held-out 验证：提升不是对调参集的过拟合",
           "调参集 +7.2pp 在从未调参的 500 题上复现为 +5.8pp，轻微衰减但强显著。")
    finish(fig, "fig4_heldout.png")


# ── 图 5：题型切片 ──────────────────────────────────────────────────────
def fig_slices(D):
    rows = D["slices"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bw, gap = 0.30, 0.05
    for i, r in enumerate(rows):
        for j, (v, color) in enumerate([(r["baseline_em"], S1),
                                        (r["final_em"], S2)]):
            x = i + (j - 0.5) * (bw + gap)
            vbar(ax, x, 0, v, bw, color, 0.05)
            ax.text(x, v + 0.018, "%.1f%%" % (v * 100), ha="center", va="bottom",
                    fontsize=10, color=INK, fontweight="bold", zorder=4)
    ax.set_xlim(-0.62, len(rows) - 0.38)
    ax.set_ylim(0, 0.90)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(["%s\nn = %d · %+.1fpp" % (r["type"], r["n"], r["delta"] * 100)
                        for r in rows], fontsize=10.5, color=INK)
    yt = [0, 0.2, 0.4, 0.6, 0.8]
    ax.set_yticks(yt)
    ax.set_yticklabels(["%d%%" % round(v * 100) for v in yt], fontsize=9.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    strip(ax, ("bottom",))
    legend(ax, [(S1, "ReAct baseline"), (S2, "cotsc + content")], y=-0.16)
    titles(ax, "提升从哪来：题型切片",
           "提升几乎全来自 bridge 题 —— 正是需要二跳检索的场景。")
    finish(fig, "fig5_slices.png")


# ── 图 6：失败归因热力图 ────────────────────────────────────────────────
def fig_attribution(D):
    A = D["attribution"]
    cats = list(A["categories"])
    short = {"A": "粒度/格式错", "B": "检索未达", "C": "二跳断链",
             "D": "歧义/错页", "E": "推理误用", "F": "标签/数据问题",
             "G": "其他"}
    stages = ["baseline", "content", "cotsc2", "combo"]
    vals = [[A["counts"][s].get(c, 0) for c in cats] for s in stages]
    maxv = max(max(r) for r in vals)
    fig, ax = plt.subplots(figsize=(9.8, 3.4))
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            color = SEQ(v / maxv)
            ax.add_patch(Rectangle((j + 0.012, i + 0.025), 0.976, 0.95,
                                   facecolor=color, edgecolor="none"))
            ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center",
                    fontsize=12.5, fontweight="bold", color=ink_on(to_hex(color)))
    ax.set_xlim(0, len(cats))
    ax.set_ylim(len(stages), 0)
    ax.set_xticks([j + 0.5 for j in range(len(cats))])
    ax.set_xticklabels(["%s %s" % (c, short[c]) for c in cats], fontsize=10.5,
                       color=INK)
    ax.set_yticks([i + 0.5 for i in range(len(stages))])
    ax.set_yticklabels(stages, fontsize=11, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(vmin=0, vmax=maxv))
    cb = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("条数", fontsize=9, color=INK2, labelpad=6)
    cb.set_ticks([0, 5, 10, 15, 20])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=0)
    titles(ax, "失败归因：改进后还剩什么（每阶段精读 50 条）",
           "每阶段 n = 50。格内已直接标数，颜色只作辅助。A 粒度桶是最大残桶且两臂都不治；"
           "E 推理误用 22%→14% 是投票的贡献；C+D 断链/歧义 18%→8% 是全文兜底的贡献。")
    finish(fig, "fig6_attribution.png")


# ── 图 7：规则信号小多图 ────────────────────────────────────────────────
def fig_signals(D):
    S = D["attribution"]["rule_signals_full_failures"]
    stages = ["baseline", "content", "cotsc2", "combo"]
    defs = [("empty_pred", "空答案 empty pred", "步数耗尽的直接后果 · 125 → 2"),
            ("forced_finish", "强制结束 forced finish", "步数耗尽率 · 25% → 18%"),
            ("near_miss_f1>=0.8", "近似错 F1≥0.8 但 EM=0", "归一化层的结构性残留")]
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4))
    for ax, (k, title, _desc) in zip(axes, defs):
        vals = [S[k][s] for s in stages]
        top = max(vals) * 1.30
        for i, v in enumerate(vals):
            vbar(ax, i, 0, v, 0.52, S1, 0.05)
            ax.text(i, v + top * 0.032, str(v), ha="center", va="bottom",
                    fontsize=10, color=INK, fontweight="bold", zorder=4)
        ax.set_xlim(-0.62, len(stages) - 0.38)
        ax.set_ylim(0, top)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, fontsize=9, color=MUTED)
        ax.set_yticks([])
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        strip(ax, ("bottom",))
        ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")
    titles(axes[0], "规则信号（全量 500 题）",
           "每张小图独立纵轴。空答案 125→2 与强制结束 25%→18% 是两条改进臂各自最直接的证据。",
           main_pt=44, sub_pt=26)
    finish(fig, "fig7_signals.png")


def main():
    setup()
    os.makedirs(OUTDIR, exist_ok=True)
    with open(VIZ, encoding="utf-8") as fh:
        D = json.load(fh)
    print("writing %s/" % OUTDIR)
    fig_baselines(D)
    fig_ablation(D)
    fig_cost(D)
    fig_heldout(D)
    fig_slices(D)
    fig_attribution(D)
    fig_signals(D)


if __name__ == "__main__":
    main()
