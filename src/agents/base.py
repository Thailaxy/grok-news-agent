import asyncio
import logging
from typing import Optional

import google.generativeai as genai
from groq import Groq
from groq import APIConnectionError as GroqAPIConnectionError
from groq import APIStatusError as GroqAPIStatusError
from groq import RateLimitError as GroqRateLimitError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import Config

logger = logging.getLogger(__name__)

# Transient error types we retry on. Groq typing can vary between SDK versions,
# so we include a couple of parents for safety.
_GROQ_RETRY_ERRORS = (GroqRateLimitError, GroqAPIConnectionError, GroqAPIStatusError)
# Gemini raises its own exceptions; we catch broadly from google.api_core if present.
try:
    from google.api_core import exceptions as gcore_exc

    _GEMINI_RETRY_ERRORS: tuple = (
        gcore_exc.ResourceExhausted,
        gcore_exc.ServiceUnavailable,
        gcore_exc.DeadlineExceeded,
        gcore_exc.InternalServerError,
    )
except Exception:  # pragma: no cover - defensive fallback
    _GEMINI_RETRY_ERRORS = ()


def _retrier(retry_on: tuple) -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type(retry_on),
        reraise=True,
    )


class BaseAgent:
    def __init__(
        self,
        name: str,
        instructions: str,
        llm_provider: Optional[str] = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.llm_provider = (llm_provider or Config.DEFAULT_LLM).lower()

        if self.llm_provider == "gemini":
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=Config.GEMINI_MODEL,
                system_instruction=instructions,
            )
        else:
            self.client = Groq(api_key=Config.GROQ_API_KEY)

    async def chat(self, prompt: str) -> str:
        if self.llm_provider == "gemini":
            return await self._chat_gemini(prompt)
        return await self._chat_groq(prompt)

    async def _chat_gemini(self, prompt: str) -> str:
        try:
            async for attempt in _retrier(_GEMINI_RETRY_ERRORS):
                with attempt:
                    response = await self.model.generate_content_async(prompt)
                    return response.text
        except RetryError as e:  # pragma: no cover - tenacity reraise=True bypasses this
            raise e
        raise RuntimeError("Gemini call returned no response")

    async def _chat_groq(self, prompt: str) -> str:
        def _call() -> str:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt},
                ],
                model=Config.GROQ_MODEL,
            )
            return response.choices[0].message.content

        async for attempt in _retrier(_GROQ_RETRY_ERRORS):
            with attempt:
                return await asyncio.to_thread(_call)
        raise RuntimeError("Groq call returned no response")
