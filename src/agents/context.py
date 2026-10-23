from dataclasses import dataclass
from typing import Any, Callable, Optional

from langgraph.runtime import Runtime


@dataclass
class LLMContext:
    """Run-time context for the graphs: which factory builds the models (None = Groq)."""
    llm_factory: Optional[Callable[..., Any]] = None


def llm_factory_from(runtime: Optional[Runtime]) -> Optional[Callable[..., Any]]:
    """The factory passed with this run, or None when the caller passed none (eval runner, scripts).

    Callers write `(factory or get_llm)`, so the default is looked up where the model is built
    and a patched get_llm still applies there.
    """
    if runtime is not None and runtime.context is not None:
        return runtime.context.llm_factory
    return None
