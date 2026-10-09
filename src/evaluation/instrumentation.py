"""
Instrumentation wrapper for the planning graph.

Does NOT modify any existing planning code. Collects:
  - per-node wall-clock timing (from stream chunk arrival times)
  - estimated LLM call count (1 per critic_panel pass = 1 + revision_count)
  - total runtime
  - errors surfaced through the graph log

LLM retry count is not tracked here — call_with_retry in src/agents/llm.py
does not expose a counter. This is a known gap; see docs/evaluation.md.
"""

import contextlib
import time
from typing import AsyncGenerator

import src.agents.nodes.assembler as _assembler_module
from src.agents.graph import planning_graph
from src.agents.replan_graph import replan_graph
from src.agents.state import PlanningState, ReplanningState
from src.evaluation.models import OperationalMetrics
from src.models.itinerary import Itinerary


@contextlib.asynccontextmanager
async def _inject_weather(weather_list: list[dict | None] | None):
    """Monkeypatch assembler.fetch_weather for scenarios with injected weather."""
    if weather_list is None:
        yield
        return

    original = _assembler_module.fetch_weather

    async def _mock(*args, **kwargs):
        return list(weather_list)

    _assembler_module.fetch_weather = _mock
    try:
        yield
    finally:
        _assembler_module.fetch_weather = original


async def run_planning(
    state: PlanningState,
    injected_weather: list[dict | None] | None = None,
) -> tuple[Itinerary | None, OperationalMetrics]:
    """
    Run the planning graph and return the final itinerary + operational metrics.
    Never raises — errors are captured in OperationalMetrics.errors.
    """
    node_runtimes: dict[str, float] = {}
    errors: list[str] = []
    itinerary: Itinerary | None = None

    t_total_start = time.perf_counter()
    t_prev = t_total_start

    try:
        async with _inject_weather(injected_weather):
            async for chunk in planning_graph.astream(state):
                t_now = time.perf_counter()
                node_name = list(chunk.keys())[0] if chunk else None
                if node_name:
                    node_runtimes[node_name] = round(t_now - t_prev, 3)
                    t_prev = t_now
                    node_state = chunk.get(node_name, {})
                    if node_name == "highlight_ranker":
                        itinerary = node_state.get("itinerary")
                    if node_state.get("error"):
                        errors.append(f"[{node_name}] {node_state['error']}")
    except Exception as exc:
        errors.append(str(exc))

    total_runtime = round(time.perf_counter() - t_total_start, 3)

    # Estimated LLM calls: 1 per critic_panel pass.
    # critic_panel runs (1 + revision_count) times.
    revision_count = itinerary.revision_count if itinerary else 0
    llm_call_count = 1 + revision_count

    return itinerary, OperationalMetrics(
        total_runtime_seconds=total_runtime,
        node_runtimes=node_runtimes,
        llm_call_count=llm_call_count,
        revision_count=revision_count,
        errors=errors,
    )


async def run_replanning(
    initial_itinerary: Itinerary,
    disruption_message: str,
    completed_day_indices: list[int],
    session_id: str,
) -> tuple[Itinerary | None, OperationalMetrics]:
    """
    Run the replanning graph and return the updated itinerary + operational metrics.
    Never raises.
    """
    from src.models.persona import TravelerPersona

    node_runtimes: dict[str, float] = {}
    errors: list[str] = []
    itinerary: Itinerary | None = None

    state: ReplanningState = {
        "session_id": session_id,
        "persona": initial_itinerary.persona,
        "itinerary": initial_itinerary,
        "disruption": disruption_message,
        "completed_day_indices": completed_day_indices,
        "log": [],
    }

    t_total_start = time.perf_counter()
    t_prev = t_total_start

    try:
        async for chunk in replan_graph.astream(state):
            t_now = time.perf_counter()
            node_name = list(chunk.keys())[0] if chunk else None
            if node_name:
                node_runtimes[node_name] = round(t_now - t_prev, 3)
                t_prev = t_now
                node_state = chunk.get(node_name, {})
                if node_name == "replan_assembler":
                    itinerary = node_state.get("itinerary")
                if node_state.get("error"):
                    errors.append(f"[{node_name}] {node_state['error']}")
    except Exception as exc:
        errors.append(str(exc))

    total_runtime = round(time.perf_counter() - t_total_start, 3)

    return itinerary, OperationalMetrics(
        total_runtime_seconds=total_runtime,
        node_runtimes=node_runtimes,
        llm_call_count=1,  # disruption_analyzer makes one LLM call
        revision_count=0,
        errors=errors,
    )
