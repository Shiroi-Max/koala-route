# langgraph_runner.py
import traceback

from modules.graph.agent_state import AgentState
from modules.graph.graph import build_langgraph_controller_flow

dialogue_manager = build_langgraph_controller_flow()


def run_prompt(user_query: str) -> str:
    """
    Ejecuta una única interacción con el grafo.
    """

    try:
        state = AgentState(input=user_query, response="")
        result = dialogue_manager.invoke(state)
        return result.get("response")
    except Exception as e:
        print("💥 ERROR ejecutando el grafo:")
        traceback.print_exc()
        raise RuntimeError("Fallo en el grafo") from e
