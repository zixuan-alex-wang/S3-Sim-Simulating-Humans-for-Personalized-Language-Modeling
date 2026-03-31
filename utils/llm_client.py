import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional
from openai import AsyncOpenAI
from utils.config import (
    OPENAI_BASE_URL,
    OPENAI_API_KEY,
    MODEL_NAME,
    CONCURRENT_API_LIMIT,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)
class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        max_concurrent: int = CONCURRENT_API_LIMIT,
        log_dir: Optional[Path] = None,
    ):
        self.model = model or MODEL_NAME
        self.client = AsyncOpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._call_count = 0
        self._total_tokens = 0
        self._log_dir = log_dir or (OUTPUT_DIR / "llm_logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / f"calls_{int(time.time())}.jsonl"

    def _log_call(self, record: dict):
        """Append a structured log record to the JSONL log file."""
        record["timestamp"] = time.time()
        record["call_index"] = self._call_count
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to write LLM log: %s", e)

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
        call_type: str = "chat",
        metadata: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request with retry logic.

        Args:
            call_type: Label for logging (e.g., "user_simulator", "assistant", "judge").
            metadata: Extra context to include in the log record.

        Returns the assistant message content string.
        """
        async with self.semaphore:
            for attempt in range(1, API_RETRY_ATTEMPTS + 1):
                try:
                    kwargs = dict(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if response_format:
                        kwargs["response_format"] = response_format

                    t0 = time.time()
                    resp = await self.client.chat.completions.create(**kwargs)
                    elapsed = time.time() - t0

                    self._call_count += 1
                    usage = {}
                    if resp.usage:
                        self._total_tokens += resp.usage.total_tokens
                        usage = {
                            "prompt_tokens": resp.usage.prompt_tokens,
                            "completion_tokens": resp.usage.completion_tokens,
                            "total_tokens": resp.usage.total_tokens,
                        }
                    content = resp.choices[0].message.content or ""

                    # Log every call
                    self._log_call({
                        "call_type": call_type,
                        "model": self.model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "input_messages": [
                            {"role": m["role"], "content": m["content"][:300]}
                            for m in messages
                        ],
                        "output": content[:500],
                        "usage": usage,
                        "elapsed_s": round(elapsed, 2),
                        "attempt": attempt,
                        **(metadata or {}),
                    })

                    return content
                except Exception as e:
                    logger.warning(
                        "LLM call attempt %d/%d failed: %s",
                        attempt,
                        API_RETRY_ATTEMPTS,
                        e,
                    )
                    if attempt < API_RETRY_ATTEMPTS:
                        await asyncio.sleep(API_RETRY_DELAY * attempt)
                    else:
                        raise

    async def chat_json(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        call_type: str = "chat_json",
        metadata: Optional[dict] = None,
    ) -> dict:
        """Chat and parse the response as JSON. Falls back to extracting JSON from text."""
        text = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            call_type=call_type,
            metadata=metadata,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
            raise

    @property
    def stats(self) -> dict:
        return {
            "calls": self._call_count,
            "total_tokens": self._total_tokens,
            "log_file": str(self._log_file),
        }
