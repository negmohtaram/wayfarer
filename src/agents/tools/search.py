from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current travel information about a place or experience.

    Use this to find recent opening hours, prices, closures, or notable updates
    that may not be in static knowledge bases.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n".join(
            f"- {r.get('title', '')}: {r.get('body', '')[:200]}"
            for r in results
        )
    except Exception as e:
        return f"Search unavailable: {e}"
