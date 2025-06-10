from dataclasses import dataclass
from modules.prompt_utils import build_chatml_messages
from modules.schema.state_schema import StateSchema


@dataclass
class ControllerAgent:
    """
    Agente controlador que coordina el flujo RAG: recuperación + generación.

    Atributos:
        state: Estado del grafo que contiene la entrada del usuario y la respuesta generada.
    """

    def run(self, state: StateSchema) -> dict:
        """
        Ejecuta el flujo completo: recupera contexto y genera la respuesta.

        Args:
            state: Estado del grafo que contiene la entrada del usuario.

        Returns:
            Siguiente nodo a ejecutar
        """
        response = state.response
        next_node = None

        if state.last_node is None:
            next_node = "consulta"
        elif state.last_node == "consulta":
            next_node = "llm"

            system_prompt = ""
            if not state.response:
                system_prompt = (
                    "Tu sistema de recuperación no ha encontrado documentos útiles. "
                    "Responde con tu conocimiento general de forma clara, directa y en español."
                )
            response = build_chatml_messages(state.input, state.response, system_prompt)

        return {
            "response": response,
            "next_node": next_node,
            "last_node": "controlador",
        }
