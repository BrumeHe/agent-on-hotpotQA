"""Agent classes for the HotpotQA fullwiki benchmark.

  BaseAgent
  ├── DirectAgent        closed-book, one LLM call
  ├── CoTAgent           chain-of-thought, one LLM call
  └── ReActAgent         retrieval + reasoning loop
        ├── ActOnlyAgent         webact arm: actions only, no thoughts
        ├── MemoryReActAgent     + episodic memory (case/reflection banks)
        ├── RAGReActAgent        + question-aware search observations
        ├── FallbackReActAgent   + deterministic query-simplification retry
        ├── FallbackV2ReActAgent + retry by following FTS similar-title hits
        ├── ConciseReActAgent    + explicit answer-granularity instruction
        ├── SelfCheckReActAgent  + LLM verification gate before finish[]
        ├── SelfCheckStrictReActAgent  + cite-evidence verification gate
        └── CotScReActAgent      + CoT self-consistency vote on forced finish

Capabilities are injected components (env via make_env, memory and the
verify_finish gate via constructor/overrides); default NullMemory + WikiEnv
+ accept-all verify_finish keep the baseline path byte-identical.

ReAct loops are faithful ports of webthink() from ReAct's official
hotpotqa.ipynb: up to 7 steps, one bad-call retry per step, forced finish[]
if the loop ends without finish, action first letter lowercased before
dispatch, and '\\n' stripped from observations.
"""

import re
from collections import Counter

from evaluate import normalize_answer
from memory import NullMemory
from prompts import (ACT_PROMPT, COT_PROMPT, DIRECT_PROMPT, REACT_PROMPT,
                     REACT_PROMPT_CONCISE, SELFCHECK_PROMPT,
                     SELFCHECK_PROMPT_STRICT)
from retrievers import (ContentFallbackWikiEnv, FallbackV2WikiEnv,
                        FallbackWikiEnv, QuestionAwareEnv)
from wiki_env import WikiEnv

# Act-only recovery: chat models often prepend reasoning or echo the
# "Action i:" prefix; extract the first well-formed action if present.
ACTION_RE = re.compile(r"(?i)\b(search|lookup|finish)\[([^\]\n]*)\]")


def _join_prompt(examples, question):
    prompt = examples
    if not prompt.endswith("\n"):
        prompt += "\n"
    return prompt + f"Question: {question}\n"


class BaseAgent:
    name = "base"

    def __init__(self, llm):
        self.llm = llm

    def run(self, question):
        raise NotImplementedError


class DirectAgent(BaseAgent):
    name = "direct"

    def __init__(self, llm, max_tokens=50):
        super().__init__(llm)
        self.max_tokens = max_tokens

    def run(self, question):
        prompt = _join_prompt(DIRECT_PROMPT, question) + "Answer:"
        res = self.llm.generate(prompt, stop=["\n"], max_tokens=self.max_tokens)
        answer = res.text.strip()
        return {"answer": answer, "trace": prompt + answer,
                "n_calls": 1, "n_badcalls": 0, "steps": 1, "searches": 0,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens}


class CoTAgent(BaseAgent):
    name = "cot"

    def __init__(self, llm, max_tokens=200):
        super().__init__(llm)
        self.max_tokens = max_tokens

    def run(self, question):
        prompt = _join_prompt(COT_PROMPT, question) + "Thought:"
        res = self.llm.generate(prompt, stop=["\nQuestion:"],
                                max_tokens=self.max_tokens)
        text = res.text.strip()
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip().split("\n")[0].strip()
        else:
            answer = text.split("\n")[-1].strip()
        return {"answer": answer, "trace": prompt + text,
                "n_calls": 1, "n_badcalls": 0, "steps": 1, "searches": 0,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens}


class ReActAgent(BaseAgent):
    name = "react"
    PROMPT = REACT_PROMPT

    def __init__(self, llm, db_path, memory=None, max_steps=7):
        super().__init__(llm)
        self.db_path = db_path
        self.memory = memory or NullMemory()
        self.max_steps = max_steps

    def make_env(self, question):
        return WikiEnv(self.db_path)

    def verify_finish(self, question, answer, observations):
        """Gate checked before a finish[] is dispatched; baseline always accepts."""
        return True

    def on_forced_finish(self, question, env):
        """Called when the step budget is exhausted. Returns (extra llm calls,
        trace note). The baseline just forces an empty finish[]."""
        env.step("finish[]")
        return 0, ""

    def run(self, question):
        env = self.make_env(question)
        memory = self.memory
        env.reset()
        prompt = _join_prompt(self.PROMPT, question) + memory.inject(question)
        n_calls, n_badcalls = 0, 0
        steps_detail = []
        obs_history = []
        done = False
        for i in range(1, self.max_steps + 1):
            step = {"i": i, "thought": "", "action": "", "obs": "", "done": False,
                    "badcall": False, "raw": None,
                    "prompt_tokens": 0, "completion_tokens": 0}
            n_calls += 1
            res = self.llm.generate(prompt + f"Thought {i}:",
                                    stop=[f"\nObservation {i}:"], max_tokens=100)
            step["prompt_tokens"] += res.prompt_tokens
            step["completion_tokens"] += res.completion_tokens
            thought_action = res.text
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except ValueError:
                n_badcalls += 1
                n_calls += 1
                step["badcall"] = True
                step["raw"] = thought_action
                thought = thought_action.strip().split("\n")[0]
                res = self.llm.generate(prompt + f"Thought {i}: {thought}\nAction {i}:",
                                        stop=["\n"], max_tokens=100)
                step["prompt_tokens"] += res.prompt_tokens
                step["completion_tokens"] += res.completion_tokens
                action = res.text.strip()
            if (action.lower().startswith("finish[")
                    and not self.verify_finish(question,
                                               action[action.find("[") + 1:-1],
                                               obs_history)):
                obs, done = ("Verification failed: evidence does not clearly "
                             "support the answer. Keep searching or revise."), False
                env.steps += 1  # keep the step metric aligned with loop steps
            elif action:
                obs, done = env.step(action[0].lower() + action[1:])
            else:
                obs, done = "Invalid action: ", False
            if obs.startswith("Could not find"):
                hint = memory.on_search_miss(action[action.find("[") + 1:-1])
                if hint:
                    obs += f" ({hint})"
            obs = obs.replace('\\n', '')
            obs_history.append(obs)
            step.update(thought=thought, action=action, obs=obs, done=done)
            steps_detail.append(step)
            prompt += (f"Thought {i}: {thought}\nAction {i}: {action}\n"
                       f"Observation {i}: {obs}\n")
            if done:
                break
        if not done:
            extra_calls, note = self.on_forced_finish(question, env)
            n_calls += extra_calls
            if note:
                prompt += note + "\n"
        return {"answer": env.answer or "", "trace": prompt,
                "n_calls": n_calls, "n_badcalls": n_badcalls,
                "steps": env.steps, "searches": env.num_searches,
                "prompt_tokens": sum(s["prompt_tokens"] for s in steps_detail),
                "completion_tokens": sum(s["completion_tokens"] for s in steps_detail),
                "steps_detail": steps_detail, "forced_finish": not done}


class ActOnlyAgent(ReActAgent):
    """webact_simple6 arm: same loop, but the model emits only
    Action/Observation steps, no Thoughts. (verify_finish hook not applied:
    self-check is a react-arm ablation.)"""
    name = "actonly"
    PROMPT = ACT_PROMPT

    def run(self, question):
        env = self.make_env(question)
        memory = self.memory
        env.reset()
        prompt = _join_prompt(self.PROMPT, question) + memory.inject(question)
        n_calls = 0
        steps_detail = []
        done = False
        for i in range(1, self.max_steps + 1):
            step = {"i": i, "thought": None, "action": "", "obs": "", "done": False,
                    "badcall": False, "raw": None,
                    "prompt_tokens": 0, "completion_tokens": 0}
            n_calls += 1
            res = self.llm.generate(prompt + f"Action {i}:",
                                    stop=[f"\nObservation {i}:"], max_tokens=100)
            step["prompt_tokens"] += res.prompt_tokens
            step["completion_tokens"] += res.completion_tokens
            text = res.text.strip()
            m = ACTION_RE.search(text)
            if m:
                action = f"{m.group(1).lower()}[{m.group(2)}]"
            else:
                action = text.split("\n")[0].strip()  # -> Invalid action, by protocol
                step["badcall"] = True  # not an official bad-call (no retry)
                step["raw"] = text
            if action:
                obs, done = env.step(action[0].lower() + action[1:])
            else:
                obs, done = "Invalid action: ", False
            if obs.startswith("Could not find"):
                hint = memory.on_search_miss(action[action.find("[") + 1:-1])
                if hint:
                    obs += f" ({hint})"
            obs = obs.replace('\\n', '')
            step.update(action=action, obs=obs, done=done)
            steps_detail.append(step)
            prompt += f"Action {i}: {action}\nObservation {i}: {obs}\n"
            if done:
                break
        if not done:
            env.step("finish[]")
        return {"answer": env.answer or "", "trace": prompt,
                "n_calls": n_calls,
                "n_badcalls": sum(s["badcall"] for s in steps_detail),
                "steps": env.steps, "searches": env.num_searches,
                "prompt_tokens": sum(s["prompt_tokens"] for s in steps_detail),
                "completion_tokens": sum(s["completion_tokens"] for s in steps_detail),
                "steps_detail": steps_detail, "forced_finish": not done}


class MemoryReActAgent(ReActAgent):
    """ReAct + episodic memory: identical loop, but the memory hooks
    (prompt lessons before step 1, search-miss hints) are live."""

    name = "react+memory"

    def __init__(self, llm, db_path, memory, **kw):
        if memory is None or isinstance(memory, NullMemory):
            raise ValueError("MemoryReActAgent requires a real memory bank")
        super().__init__(llm, db_path, memory=memory, **kw)


class RAGReActAgent(ReActAgent):
    """ReAct + question-aware retrieval: search observations are the
    sentences most relevant to the question instead of the first five.
    Pass memory= for the combined memory+RAG arm."""

    name = "react+rag"

    def make_env(self, question):
        return QuestionAwareEnv(self.db_path, question)


class FallbackReActAgent(ReActAgent):
    """ReAct + deterministic query-simplification retry on search miss."""

    name = "react+fallback"

    def make_env(self, question):
        return FallbackWikiEnv(self.db_path)


class ConciseReActAgent(ReActAgent):
    """ReAct with an explicit answer-granularity instruction."""

    name = "react+concise"
    PROMPT = REACT_PROMPT_CONCISE


class SelfCheckReActAgent(ReActAgent):
    """ReAct + an LLM verification gate before accepting a finish[]."""

    name = "react+selfcheck"

    def verify_finish(self, question, answer, observations):
        evidence = " ".join(observations)[-2000:]
        res = self.llm.generate(
            SELFCHECK_PROMPT.format(question=question, evidence=evidence,
                                    answer=answer),
            stop=["\n"], max_tokens=5)
        return not res.text.strip().lower().startswith("no")


class SelfCheckStrictReActAgent(SelfCheckReActAgent):
    """ReAct + a cite-evidence verification gate: the verifier must quote the
    supporting sentence and check answer granularity before the verdict.
    Unparseable verdicts are rejected (strict)."""

    name = "react+selfcheck-strict"

    def verify_finish(self, question, answer, observations):
        evidence = " ".join(observations)[-2000:]
        res = self.llm.generate(
            SELFCHECK_PROMPT_STRICT.format(question=question, evidence=evidence,
                                           answer=answer),
            max_tokens=150)
        m = re.search(r"verdict:\s*(yes|no)", res.text, re.I)
        return bool(m and m.group(1).lower() == "yes")


class FallbackV2ReActAgent(ReActAgent):
    """ReAct + search retries that follow the env's own FTS similar-title
    suggestions instead of blindly shortening the query."""

    name = "react+fallback2"

    def make_env(self, question):
        return FallbackV2WikiEnv(self.db_path)


class CotScReActAgent(ReActAgent):
    """ReAct -> CoT-SC (the paper's best HotpotQA combo): when the step budget
    is exhausted without a finish[], sample SAMPLES CoT completions at
    temperature 0.7 and finish with the majority answer, grouped by the
    official normalize_answer (earliest raw form breaks ties)."""

    name = "react+cotsc"
    SAMPLES = 5

    def on_forced_finish(self, question, env):
        prompt = _join_prompt(COT_PROMPT, question) + "Thought:"
        answers = []
        for _ in range(self.SAMPLES):
            res = self.llm.generate(prompt, stop=["\nQuestion:"],
                                    max_tokens=200, temperature=0.7)
            text = res.text.strip()
            if "Answer:" not in text:
                continue  # sample cut off mid-reasoning; its last line is
                          # a thought fragment, not an answer
            ans = text.split("Answer:")[-1].strip().split("\n")[0].strip()
            if ans:
                answers.append(ans)
        if not answers:
            env.step("finish[]")
            return self.SAMPLES, "[CoT-SC fallback: no usable answer]"
        best, _ = majority_vote(answers)
        env.step(f"finish[{best}]")
        return self.SAMPLES, f"[CoT-SC fallback: votes={answers} -> {best}]"


def majority_vote(answers):
    """Group answers by the official normalize_answer; highest count wins,
    earliest first occurrence breaks ties. Returns (winner_raw, groups)."""
    counts = Counter(normalize_answer(a) for a in answers)
    order = {}
    for a in answers:
        order.setdefault(normalize_answer(a), len(order))
    best_key = max(counts, key=lambda k: (counts[k], -order[k]))
    best = next(a for a in answers if normalize_answer(a) == best_key)
    return best, counts


class ContentFallbackReActAgent(ReActAgent):
    """ReAct + content-level retrieval: on a title miss, the env falls back
    to FTS5/BM25 over every page's intro paragraph (tool-layer arm)."""

    name = "react+content"

    def make_env(self, question):
        return ContentFallbackWikiEnv(self.db_path)


class ReActSCAgent(ReActAgent):
    """ReAct-SC: policy-layer self-consistency. Run K independent ReAct
    trajectories at temperature 0.7 and majority-vote the final answers
    (official normalize_answer grouping, earliest form breaks ties).
    Generalizes the cotsc arm's vote from forced-finish-only to always."""

    name = "react+sc"
    SAMPLES = 5

    def run(self, question):
        self.llm.default_temperature = 0.7
        try:
            runs = [super().run(question) for _ in range(self.SAMPLES)]
        finally:
            self.llm.default_temperature = 0.0
        answers = [r["answer"] for r in runs if r["answer"]]
        if answers:
            best, counts = majority_vote(answers)
        else:
            best, counts = "", Counter()
        note = (f"[ReAct-SC: {self.SAMPLES} trajectories, votes={answers}"
                f" -> {best}]\n")
        trace = note + "\n".join(
            f"=== trajectory {k} ===\n{r['trace']}" for k, r in enumerate(runs))
        return {"answer": best, "trace": trace,
                "n_calls": sum(r["n_calls"] for r in runs),
                "n_badcalls": sum(r["n_badcalls"] for r in runs),
                "steps": sum(r["steps"] for r in runs) / len(runs),
                "searches": sum(r["searches"] for r in runs),
                "prompt_tokens": sum(r["prompt_tokens"] for r in runs),
                "completion_tokens": sum(r["completion_tokens"] for r in runs),
                "steps_detail": [r["steps_detail"] for r in runs],
                "forced_finish": all(r["forced_finish"] for r in runs)}


class CotScContentReActAgent(CotScReActAgent):
    """Combo arm: content-level retrieval (tool layer) + CoT-SC vote on
    forced finish (policy layer). The two mechanisms target disjoint
    failure buckets (search miss vs exhausted step budget)."""

    name = "react+cotsc+content"

    def make_env(self, question):
        return ContentFallbackWikiEnv(self.db_path)
