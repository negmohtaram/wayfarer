from langgraph.graph import StateGraph, START, END
from src.agents.state import PlanningState
from src.agents.nodes.experience_generator import experience_generator_node
from src.agents.nodes.optimizer import optimizer_node
from src.agents.nodes.assembler import assembler_node
from src.agents.nodes.critic_panel import critic_panel_node, route_after_critique
from src.agents.nodes.revision import revision_agent_node
from src.agents.nodes.highlight_ranker import highlight_ranker_node


def build_planning_graph():
    graph = StateGraph(PlanningState)

    graph.add_node("experience_generator", experience_generator_node)
    graph.add_node("optimizer", optimizer_node)
    graph.add_node("assembler", assembler_node)
    graph.add_node("critic_panel", critic_panel_node)
    graph.add_node("revision_agent", revision_agent_node)
    graph.add_node("highlight_ranker", highlight_ranker_node)

    graph.add_edge(START, "experience_generator")
    graph.add_edge("experience_generator", "optimizer")
    graph.add_edge("optimizer", "assembler")
    graph.add_edge("assembler", "critic_panel")
    graph.add_conditional_edges(
        "critic_panel",
        route_after_critique,
        {"revision_agent": "revision_agent", "regret_minimizer": "highlight_ranker"},
    )
    graph.add_edge("revision_agent", "critic_panel")
    graph.add_edge("highlight_ranker", END)

    return graph.compile()


planning_graph = build_planning_graph()
