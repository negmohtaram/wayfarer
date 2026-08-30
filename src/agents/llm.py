import asyncio
import logging
from typing import Any, Callable, Coroutine

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.7) -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=temperature,
    )


async def call_with_retry(
    fn: Callable[..., Coroutine[Any, Any, Any]],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Call an async LLM function with exponential backoff on transient failures."""
    for attempt in range(max_attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_attempts,
                e,
                delay,
            )
            await asyncio.sleep(delay)
