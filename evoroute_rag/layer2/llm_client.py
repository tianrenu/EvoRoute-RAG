"""LLM client with Circuit Breaker and exponential backoff retry."""

import time
from enum import Enum
from typing import Optional

import openai


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit Breaker: trips open after consecutive failures, auto-recovers after timeout."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise RuntimeError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class LLMClient:
    """MiniMax API client (OpenAI-compatible format) with CircuitBreaker + retry."""

    def __init__(self, base_url: str, api_key: str, model: str = "abab6.5-chat"):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.circuit_breaker = CircuitBreaker()
        self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

    def call(self, messages: list, max_retries: int = 3, timeout: int = 30) -> str:
        """Call LLM with exponential backoff retry (1s → 2s → 4s)."""

        def _call():
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=timeout,
            )
            return response.choices[0].message.content

        for attempt in range(max_retries):
            try:
                return self.circuit_breaker.call(_call)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = 2**attempt  # 1s, 2s, 4s
                time.sleep(wait_time)

        raise RuntimeError("LLM call failed after all retries")
