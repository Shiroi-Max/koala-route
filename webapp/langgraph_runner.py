# langgraph_runner.py
from modules.graph.graph import build_langgraph_controller_flow
from modules.schema.state_schema import StateSchema

dialogue_manager = build_langgraph_controller_flow()


def run_prompt(user_query: str) -> str:
    """
    Ejecuta una única interacción con el grafo.
    """
    state = StateSchema(input=user_query, response="")
    result = dialogue_manager.invoke(state)
    return result.response
