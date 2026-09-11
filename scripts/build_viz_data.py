"""Rebuild the dashboard's data file from the raw result files.

Every measured number in results/viz_data.json (and therefore in
docs/dashboard.html) is recomputed here from the per-question traces in
results/*.jsonl -- EM / F1 are re-averaged from the per-question rows, the
McNemar b/c counts come from joining each arm against the ReAct baseline on
`idx`, and the p-values are exact two-sided binomial tests. Cost and call
totals are read from the sibling *_summary.json.

Only the prose fields (arm `layer` / `mech` / `verdict`) are hand-authored,
since they describe intent rather than measurement.

Usage (from project root):
  .venv/Scripts/python scripts/build_viz_data.py
  .venv/Scripts/python scripts/build_viz_data.py --inject   # also refresh docs/dashboard.html
"""

import argparse
import glob
import json
import os
from math import comb

RESULTS = "results"
OUT_JSON = os.path.join(RESULTS, "viz_data.json")
DASHBOARD = os.path.join("docs", "dashboard.html")

# ── 臂定义 ────────────────────────────────────────────────────────────────
# name → 结果文件 stem。cotsc 臂的 EM/F1 取 cotsc2 run（该 run 无成本计数器
# bug），成本取同配置的 cotsc run —— 与 AGENTS.md 记录一致。
BASELINES = [
    ("Direct", "direct_n500_seed233_off0_baseline_20260908_204855",
     "闭卷基线：不给检索工具，直接凭参数知识作答"),
    ("CoT", "cot_n500_seed233_off0_baseline_20260908_204919",
     "推理基线：一次思维链作答，无动作空间"),
    ("Act-only", "actonly_n500_seed233_off0_baseline_20260908_233917",
     "动作基线：动作空间与 ReAct 相同，仅删掉 Thought"),
    ("ReAct", "react_n500_seed233_off0_baseline_20260908_233604",
     "基线/参照系：交错 Thought→Action→Observation，最多 7 步"),
]

ARMS = [
    ("rag", "工具·观测", "BM25 选句替代导语前 5 句", "显著负",
     dict(stem="react_n500_seed233_off0_rag_20260909_091412")),
    ("memory-combo", "混合叠加", "case+refl+rag 三者全开", "显著负",
     dict(stem="react_n500_seed233_off0_combo_20260909_092709")),
    ("fallback v1", "工具·检索", "标题 miss 时砍尾词重试", "噪声",
     dict(stem="react_n500_seed233_off0_fallback_20260909_101559")),
    ("fallback v2", "工具·检索", "标题 miss 时 FTS 相似标题跟随", "噪声",
     dict(stem="react_n500_seed233_off0_fallback2_20260909_125117")),
    ("selfcheck", "策略·验证", "finish 前 LLM 验证门", "噪声",
     dict(stem="react_n500_seed233_off0_selfcheck_20260909_102315")),
    ("case", "策略·记忆", "注入相似题检索案例", "噪声",
     dict(stem="react_n500_seed233_off0_case_20260909_091904")),
    ("refl", "策略·记忆", "注入 LLM 生成的教训", "噪声",
     dict(stem="react_n500_seed233_off0_refl_20260909_092305")),
    ("concise", "策略·prompt", "答案粒度约束", "噪声",
     dict(stem="react_n500_seed233_off0_concise_20260909_102016")),
    ("content", "工具·检索", "标题 miss 时全文 BM25 兜底", "不显著",
     dict(stem="react_n500_seed233_off0_content_20260909_124827")),
    ("ReAct-SC", "策略·兜底", "5 条完整轨迹多数投票", "显著正",
     dict(stem="react_n500_seed233_off0_sc_20260909_125518")),
    ("cotsc", "策略·兜底", "步数耗尽时 CoT 自洽投票", "显著正",
     dict(stem="react_n500_seed233_off0_cotsc2_20260909_111248",
          cost_stem="react_n500_seed233_off0_cotsc_20260909_110401")),
    ("cotsc+content", "最终系统", "两臂正交叠加", "显著正",
     dict(stem="react_n500_seed233_off0_cotsccontent_20260909_152240")),
]

HELDOUT = ("react_n500_seed233_off500_heldout_20260909_155938",
           "react_n500_seed233_off500_heldout_20260909_160247")

# 同配置复跑，用于估计 temp=0 下的测量抖动
REPEATS = [("cotsc", "react_n500_seed233_off0_cotsc_20260909_110401")]

BASE_ARM = "ReAct"


# ── 读取与统计 ────────────────────────────────────────────────────────────
def load_traces(stem):
    """idx → 逐题记录（pandas 落盘把数值写成了字符串，这里统一转回 float）。"""
    rows = {}
    with open(os.path.join(RESULTS, stem + ".jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[int(r["idx"])] = {
                "em": float(r["em"]), "f1": float(r["f1"]),
                "pred": (r.get("pred") or "").strip(),
                "forced_finish": bool(r.get("forced_finish")),
                "type": r.get("type"),
            }
    return rows


def load_summary(stem):
    with open(os.path.join(RESULTS, stem + "_summary.json"), encoding="utf-8") as fh:
        return json.load(fh)


def mcnemar(base, arm):
    """b = 基线对&臂错；c = 基线错&臂对；精确双侧二项检验。"""
    shared = base.keys() & arm.keys()
    b = sum(1 for i in shared if base[i]["em"] >= 1.0 and arm[i]["em"] < 1.0)
    c = sum(1 for i in shared if base[i]["em"] < 1.0 and arm[i]["em"] >= 1.0)
    n = b + c
    if n == 0:
        return b, c, 1.0, len(shared)
    k = min(b, c)
    p = min(1.0, 2.0 * sum(comb(n, j) for j in range(k + 1)) / (2.0 ** n))
    return b, c, p, len(shared)


def aggregate(stem):
    """逐题重算 EM/F1，并从 summary 取成本与调用数。"""
    rows = load_traces(stem)
    n = len(rows)
    em = sum(r["em"] for r in rows.values()) / n
    f1 = sum(r["f1"] for r in rows.values()) / n
    summ = load_summary(stem)
    fails = [r for r in rows.values() if r["em"] < 1.0]
    return {
        "rows": rows,
        "em": round(em, 6), "f1": f1, "n": n,
        "cost": summ["llm"]["cost_usd_upper_bound"],
        "avg_steps": summ["avg_steps"], "avg_calls": summ["avg_llm_calls"],
        "empty_pred": sum(1 for r in rows.values() if not r["pred"]),
        "forced_finish": sum(1 for r in rows.values() if r["forced_finish"]),
        "near_miss": sum(1 for r in fails if r["f1"] >= 0.8),
        "n_fail": len(fails),
        "_summary_em": summ["em"],
    }


def slices(base_rows, final_rows):
    out = []
    for kind in ("bridge", "comparison"):
        idxs = [i for i, r in base_rows.items() if r["type"] == kind]
        if not idxs:
            continue
        b = sum(base_rows[i]["em"] for i in idxs) / len(idxs)
        f = sum(final_rows[i]["em"] for i in idxs) / len(idxs)
        out.append({"type": kind, "n": len(idxs),
                    "baseline_em": b, "final_em": f, "delta": f - b})
    return out


# ── 组装 ──────────────────────────────────────────────────────────────────
def build():
    base = aggregate(
        next(stem for name, stem, _ in BASELINES if name == BASE_ARM))
    data = {
        "meta": {"model": "deepseek-v4-flash", "temperature": 0, "seed": 233,
                 "n_eval": base["n"], "max_steps": 7, "noise_floor_pp": 1.6},
        "baseline_react": {
            "em": base["em"], "f1": base["f1"], "cost": base["cost"],
            "avg_steps": base["avg_steps"], "empty_pred": base["empty_pred"],
            "forced_finish": base["forced_finish"], "n_fail": base["n_fail"],
        },
        "baselines": [], "ablation": [],
    }

    for name, stem, _desc in BASELINES:
        a = aggregate(stem)
        if stem == next(s for n, s, _ in BASELINES if n == BASE_ARM):
            b, c, p = 0, 0, 1.0
        else:
            b, c, p, _ = mcnemar(base["rows"], a["rows"])
        data["baselines"].append({
            "name": name, "em": a["em"], "f1": a["f1"], "cost": a["cost"],
            "avg_steps": a["avg_steps"], "avg_calls": a["avg_calls"],
            "b": b, "c": c, "p": p,
        })

    entries = list(ARMS)
    # 基线行也进消融图（ΔEM=0），与仪表盘一致
    entries.append((BASE_ARM, "基线", "交错 Thought/Action/Observation", "噪声",
                    dict(stem=next(s for n, s, _ in BASELINES if n == BASE_ARM))))

    for name, layer, mech, verdict, spec in entries:
        a = aggregate(spec["stem"])
        cost_src = aggregate(spec["cost_stem"]) if spec.get("cost_stem") else a
        cost_warn = None
        if spec.get("cost_stem"):
            calls = load_summary(spec["stem"])["llm"]["calls"]
            cost_warn = ("成本计数器异常：calls=%d，按 avg_llm_calls×n 应为 %d"
                         "（改用同配置 run %s 的成本）"
                         % (calls, round(a["avg_calls"] * a["n"]),
                            "_".join(spec["cost_stem"].split("_")[-2:])))
        b, c, p, _ = (0, 0, 1.0, 0) if name == BASE_ARM else \
            mcnemar(base["rows"], a["rows"])
        data["ablation"].append({
            "name": name, "layer": layer, "mech": mech,
            "em": a["em"], "f1": a["f1"],
            "d_em": a["em"] - base["em"], "d_f1": a["f1"] - base["f1"],
            "cost": cost_src["cost"], "d_cost": cost_src["cost"] - base["cost"],
            "cost_warn": cost_warn,
            "avg_steps": a["avg_steps"], "avg_calls": a["avg_calls"],
            "b": b, "c": c, "p": p,
            "empty_pred": a["empty_pred"], "forced_finish": a["forced_finish"],
            "near_miss": a["near_miss"], "n_fail": a["n_fail"],
            "verdict": verdict,
        })

    # 消融表按 ΔEM 升序，与仪表盘发散图的读序一致
    data["ablation"].sort(key=lambda a: a["d_em"])

    hb, hc = aggregate(HELDOUT[0]), aggregate(HELDOUT[1])
    hb_b, hb_c, hb_p, hb_n = mcnemar(hb["rows"], hc["rows"])
    data["heldout"] = {
        "baseline": {"em": hb["em"], "f1": hb["f1"], "cost": hb["cost"], "n": hb["n"]},
        "combo": {"em": hc["em"], "f1": hc["f1"], "cost": hc["cost"], "n": hc["n"]},
        "delta": hc["em"] - hb["em"], "b": hb_b, "c": hb_c, "p": hb_p,
        "paired_n": hb_n,
    }

    combo = next(a for a in data["ablation"] if a["name"] == "cotsc+content")
    combo_rows = load_traces(next(s for n, _l, _m, _v, s in ARMS
                                  if n == "cotsc+content")["stem"])
    data["slices"] = slices(base["rows"], combo_rows)

    with open(os.path.join(RESULTS, "failure_attribution.json"), encoding="utf-8") as fh:
        attr = json.load(fh)
    data["attribution"] = attr

    data["repeats"] = []
    for name, stem in REPEATS:
        r = aggregate(stem)
        arm_em = next(a["em"] for a in data["ablation"] if a["name"] == name)
        data["repeats"].append({"name": name, "em": r["em"], "stem": stem,
                                "d_em": r["em"] - arm_em})
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inject", action="store_true",
                    help="同时把 JSON 注入 docs/dashboard.html 的 viz-data 标签")
    args = ap.parse_args()

    data = build()
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    print("wrote %s" % OUT_JSON)

    if args.inject:
        with open(DASHBOARD, encoding="utf-8") as fh:
            html = fh.read()
        head = '<script id="viz-data" type="application/json">'
        i = html.index(head) + len(head)
        j = html.index("</script>", i)
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        with open(DASHBOARD, "w", encoding="utf-8") as fh:
            fh.write(html[:i] + payload + html[j:])
        print("injected into %s" % DASHBOARD)


if __name__ == "__main__":
    main()
