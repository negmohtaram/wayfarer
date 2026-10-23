from typing import Any, Callable

from fastapi import Request


def get_llm_factory(request: Request) -> Callable[..., Any]:
    """The model factory the app was built with: Groq in production, a fake in tests."""
    return request.app.state.llm_factory
