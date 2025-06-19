"""
Construcción del flujo principal del sistema usando LangGraph.

Este módulo define el grafo de agentes que implementa el flujo RAG (Retrieval-Augmented Generation)
para planificación de viajes. Coordina tres agentes principales:

1. `ControllerAgent`: Decide el siguiente paso en el flujo.
2. `RetrieverAgent`: Recupera contexto relevante desde el vector store.
3. `LLMAgent`: Genera la respuesta final usando un modelo LLM desplegado en Azure OpenAI.

El grafo sigue el siguiente esquema:

    controlador ──▶ (consulta ──▶ controlador)* ──▶ llm ──▶ END

Donde `consulta` puede repetirse en ciclos hasta que se complete el contexto y se pase a `llm`.

Requiere:
- LangGraph (`StateGraph`) para la definición del flujo.
- Implementaciones de los agentes (`ControllerAgent`, `RetrieverAgent`, `LLMAgent`).
- Estado de grafo definido en `AgentState`.
- `vector_store` configurado para búsquedas semánticas.
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from modules.agents.controller_agent import ControllerAgent
from modules.agents.llm_agent import LLMAgent
from modules.agents.retriever_agent import RetrieverAgent
from modules.graph.agent_state import AgentState
from modules.vector import vector_store


def build_langgraph_controller_flow() -> StateGraph:
    """
    Construye un grafo LangGraph con el flujo RAG completo: recuperación + generación.

    Este flujo conecta tres agentes en la siguiente lógica:
        controlador -> (consulta -> controlador)* -> llm -> END

    El controlador decide si se debe realizar una consulta al vector store o si se puede
    proceder directamente a la generación de respuesta con el modelo LLM.

    Returns:
        StateGraph: Grafo LangGraph ya compilado y listo para ejecutarse.
    """
    # Instanciar agentes
    retriever_agent = RetrieverAgent(vector_store)
    llm_agent = LLMAgent()
    controller = ControllerAgent()

    # Crear grafo con esquema de estado definido
    workflow = StateGraph(AgentState)

    # Registrar nodos en el grafo
    workflow.add_node("controlador", controller.run)
    workflow.add_node("consulta", retriever_agent.get_context)
    workflow.add_node("llm", llm_agent.generate_response)

    # Definir punto de entrada
    workflow.set_entry_point("controlador")

    # Definir transiciones condicionales
    workflow.add_conditional_edges("controlador", next_node)
    workflow.add_edge("consulta", "controlador")
    workflow.add_edge("llm", END)

    # Compilar el grafo para su ejecución
    return workflow.compile()


def next_node(state: AgentState) -> Literal["consulta", "llm"]:
    """
    Determina el siguiente nodo a ejecutar en el grafo, en función del estado actual.

    Si no se ha ejecutado aún ninguna recuperación (`last_node is None`), se envía a "consulta".
    En caso contrario, se asume que el contexto ya está preparado y se procede a "llm".

    Args:
        state (AgentState): Estado actual del flujo.

    Returns:
        Literal["consulta", "llm"]: Nombre del siguiente nodo.
    """
    if state.get("last_node") is None:
        return "consulta"

    return "llm"
