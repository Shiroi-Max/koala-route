from dataclasses import dataclass

from modules.graph.agent_state import AgentState
from modules.prompt_utils import build_chatml_messages, load_prompt


@dataclass
class ControllerAgent:
    """
    Agente controlador que coordina el flujo RAG: recuperación + generación.

    Atributos:
        state: Estado del grafo que contiene la entrada del usuario y la respuesta generada.
    """

    def run(self, state: AgentState) -> AgentState:
        """
        Ejecuta el flujo completo: recupera contexto y genera la respuesta.

        Args:
            state: Estado del grafo que contiene la entrada del usuario.

        Returns:
            Siguiente nodo a ejecutar
        """
        response = state.get("response", "")

        if state.get("last_node") == "consulta":
            fallback_prompt = ""
            if not response:
                fallback_prompt = load_prompt("fallback_prompt")
            response = build_chatml_messages(state["input"], response, fallback_prompt)

        return {"response": response}
