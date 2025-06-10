from langgraph.graph import StateGraph, END
from modules.agents.retriever_agent import RetrieverAgent
from modules.agents.llm_agent import LLMAgent
from modules.agents.controller_agent import ControllerAgent
from modules.vector import vector_store
from modules.llm import pipeline
from modules.schema.state_schema import StateSchema


def build_langgraph_controller_flow() -> StateGraph:
    """
    Construye un grafo LangGraph con esquema de estado definido, siguiendo el flujo:

        controlador -> router -> (consulta | llm) -> END

    Returns:
        StateGraph: Grafo LangGraph listo para ejecutar.
    """
    # Instanciar agentes
    retriever_agent = RetrieverAgent(vector_store)
    llm_agent = LLMAgent(pipeline)
    controller = ControllerAgent()

    graph = StateGraph(state_schema=StateSchema)

    graph.add_node("controlador", controller.run)
    graph.add_node("consulta", retriever_agent.get_context)
    graph.add_node("llm", llm_agent.generate_response)

    graph.set_entry_point("controlador")
    graph.add_edge("controlador", "consulta")
    graph.add_edge("controlador", "llm")
    graph.add_edge("consulta", "controlador")
    graph.add_edge("llm", END)

    return graph.compile()
