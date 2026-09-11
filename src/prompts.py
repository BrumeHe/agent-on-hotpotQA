"""Few-shot prompts, verbatim from ReAct's official prompts_naive.json.
REACT_PROMPT = the exact instruction from hotpotqa.ipynb + webthink_simple6."""

import json
import os

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts",
                            "prompts_naive.json")

with open(PROMPTS_PATH, encoding="utf-8") as f:
    _d = json.load(f)

REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""

REACT_PROMPT = REACT_INSTRUCTION + _d["webthink_simple6"]
COT_PROMPT = _d["cotqa_simple6"]

# Chat-model adaptation: bare completion-style few-shot makes chat models emit
# meta-answers ("Based on the provided examples...") instead of answering
# (observed 15/20 refusals, EM 5% -> artifact). One instruction line fixes it.
DIRECT_INSTRUCTION = ("Answer the following questions. "
                      "Respond with only the short answer, no explanation.\n")
DIRECT_PROMPT = DIRECT_INSTRUCTION + _d["webqa_simple6"]

# Concise arm: explicit answer-granularity instruction (targets EM=0/F1>=0.8
# near-misses like "1970 FIFA World Cup" vs "FIFA World Cup").
CONCISE_NOTE = ("When finishing, answer with the shortest form that still fully "
                "identifies the answer, and match the precision the question asks "
                "for (e.g. \"FIFA World Cup\", not \"1970 FIFA World Cup\").\n")
REACT_PROMPT_CONCISE = REACT_INSTRUCTION + CONCISE_NOTE + _d["webthink_simple6"]

ACT_PROMPT = _d["webact_simple6"]  # Act-only ablation arm (no Thoughts)

# Self-check arm: verification gate prompt used by SelfCheckReActAgent.
SELFCHECK_PROMPT = """Question: {question}
Evidence gathered:
{evidence}
Proposed answer: {answer}
Does the evidence explicitly support the proposed answer? Reply with only yes or no."""

# Strict self-check arm: the naive yes/no prompt false-accepted 20/20 wrong
# answers (always-yes bias). Force the verifier to cite an evidence sentence
# and check answer granularity before the verdict.
SELFCHECK_PROMPT_STRICT = """You are a strict answer verifier for a multi-hop QA task.

Question: {question}
Evidence gathered (Wikipedia observations):
{evidence}
Proposed answer: {answer}

Verification steps:
1. Quote the exact sentence(s) from the evidence that state the proposed answer. If none do, write "none".
2. Check the answer matches the granularity the question asks for (e.g. if the question asks for a nationality, "Israeli" is right and "Israel-American" is wrong; if it asks for a tournament, "FIFA World Cup" is right and "1970 FIFA World Cup" is wrong).

Reply in exactly this format:
Evidence quote: <quote or none>
Verdict: yes/no"""
