"""Talking to judges.

One backend, the OpenAI chat-completions shape, because everything speaks it:
Ollama, llama.cpp's server, LM Studio, vLLM, and the hosted APIs. A judge is
just a name plus a base_url and a model. An API key is read from the env var
you name, or skipped for a local server that doesn't want one.

The contract with the rest of the package: `ask()` always returns a string or
raises JudgeError. It never returns junk silently. Retries with backoff are
handled here so the jury layer stays about voting, not HTTP.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests


class JudgeError(RuntimeError):
    """A judge could not produce an answer after retries."""


@dataclass
class Judge:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None      # env var holding the key, if any
    temperature: float = 0.0
    timeout: int = 120
    max_retries: int = 4

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def ask(self, system: str, user: str) -> str:
        """Send one prompt, return the reply text. Raise JudgeError on failure."""
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        last = ""
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
            except requests.RequestException as exc:
                last = f"network error: {exc}"
                _sleep_backoff(attempt)
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, ValueError) as exc:
                    raise JudgeError(f"{self.name}: unexpected response shape: {exc}")

            # 429 and 5xx are worth waiting out. Everything else is fatal for
            # this call, so don't burn retries on a 400.
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}"
                _sleep_backoff(attempt, retry_after=resp.headers.get("Retry-After"))
                continue

            raise JudgeError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:200]}")

        raise JudgeError(f"{self.name}: giving up after {self.max_retries} tries ({last})")


def _sleep_backoff(attempt: int, retry_after: str | None = None) -> None:
    if retry_after:
        try:
            time.sleep(min(60.0, float(retry_after)))
            return
        except (TypeError, ValueError):
            pass
    # 1, 2, 4, 8 ... capped.
    time.sleep(min(30.0, 2.0 ** attempt))
