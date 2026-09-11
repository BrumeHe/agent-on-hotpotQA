"""Answer scoring: delegates to the official HotpotQA evaluator
(src/hotpot_evaluate_v1.py). Never reimplement normalize/EM/F1 locally."""

from hotpot_evaluate_v1 import exact_match_score, f1_score, normalize_answer  # noqa: F401


def score_answer(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return {"em": float(em), "f1": f1, "prec": prec, "recall": recall}


def aggregate(rows):
    n = max(len(rows), 1)
    out = {"n": len(rows)}
    for k in ["em", "f1", "prec", "recall"]:
        out[k] = sum(r[k] for r in rows) / n
    out["avg_steps"] = sum(r.get("steps", 0) for r in rows) / n
    out["avg_llm_calls"] = sum(r.get("n_calls", 0) for r in rows) / n
    out["total_prompt_tokens"] = sum(r.get("prompt_tokens", 0) for r in rows)
    out["total_completion_tokens"] = sum(r.get("completion_tokens", 0) for r in rows)
    return out
