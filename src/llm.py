"""DeepSeek API client (OpenAI-compatible).

Defaults for this project: deepseek-v4-flash, thinking mode DISABLED
(thinking mode ignores temperature, which would break reproducibility),
temperature=0. Pricing constants are peak rates => cost estimate is an
upper bound (off-peak is half, cache hits are ~30x cheaper).
"""

import os
import threading
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

PRICE_INPUT_PER_M = 0.44   # USD, peak, cache miss
PRICE_OUTPUT_PER_M = 1.32  # USD, peak


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class LLMClient:
    def __init__(self, model=DEFAULT_MODEL, thinking=False, max_retries=5):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not set. Put it in .env or the environment.")
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model
        self.thinking = thinking
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.default_temperature = 0.0

    def generate(self, prompt, stop=None, max_tokens=100, temperature=None):
        # temperature=None -> self.default_temperature (0.0 unless an SC arm
        # overrides it); keeps every existing call site byte-identical.
        if temperature is None:
            temperature = self.default_temperature
        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "enabled" if self.thinking else "disabled"}},
        )
        if not self.thinking:
            kwargs["temperature"] = temperature  # ignored by the API in thinking mode
        if stop:
            kwargs["stop"] = stop

        last_err = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(**kwargs)
                latency = time.time() - t0
                usage = resp.usage
                res = GenerateResult(
                    text=resp.choices[0].message.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_s=latency,
                )
                with self._lock:
                    self.total_prompt_tokens += res.prompt_tokens
                    self.total_completion_tokens += res.completion_tokens
                    self.total_calls += 1
                return res
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt * 2, 30))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def summary(self):
        with self._lock:
            pt, ct, n = (self.total_prompt_tokens,
                         self.total_completion_tokens, self.total_calls)
        cost = pt / 1e6 * PRICE_INPUT_PER_M + ct / 1e6 * PRICE_OUTPUT_PER_M
        return {"calls": n, "prompt_tokens": pt, "completion_tokens": ct,
                "cost_usd_upper_bound": round(cost, 4)}
