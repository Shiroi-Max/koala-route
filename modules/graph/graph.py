from typing import Literal

from langgraph.graph import END, StateGraph

from modules.agents.controller_agent import ControllerAgent
from modules.agents.llm_agent import LLMAgent
from modules.agents.retriever_agent import RetrieverAgent
from modules.graph.agent_state import AgentState
from modules.vector import vector_store


def build_langgraph_controller_flow() -> StateGraph:
    """
    Construye un grafo LangGraph con esquema de estado definido, siguiendo el flujo:

        controlador -> router -> (consulta | llm) -> END

    Returns:
        StateGraph: Grafo LangGraph listo para ejecutar.
    """
    # Instanciar agentes
    retriever_agent = RetrieverAgent(vector_store)
    llm_agent = LLMAgent()
    controller = ControllerAgent()

    # Instanciar grafo
    workflow = StateGraph(AgentState)

    # Agregar nodos
    workflow.add_node("controlador", controller.run)
    workflow.add_node("consulta", retriever_agent.get_context)
    workflow.add_node("llm", llm_agent.generate_response)

    # Definir punto de entrada
    workflow.set_entry_point("controlador")

    workflow.add_conditional_edges("controlador", next_node)
    workflow.add_edge("consulta", "controlador")
    workflow.add_edge("llm", END)

    return workflow.compile()


def next_node(state: AgentState) -> Literal["consulta", "llm"]:
    """
    Determina el siguiente nodo a ejecutar basado en el estado actual.

    Args:
        state: Estado del grafo que contiene la entrada del usuario.

    Returns:
        str: Nombre del siguiente nodo a ejecutar.
    """
    if state.get("last_node") is None:
        return "consulta"

    return "llm"
