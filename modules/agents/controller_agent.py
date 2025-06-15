from dataclasses import dataclass

from modules.graph.agent_state import AgentState
from modules.prompt_utils import build_chatml_messages


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
            system_prompt = ""
            if not response:
                system_prompt = (
                    "Tu sistema de recuperación no ha encontrado documentos útiles. "
                    "Responde con tu conocimiento general de forma clara, directa y en español."
                )
            response = build_chatml_messages(state["input"], response, system_prompt)

        return {"response": response}
