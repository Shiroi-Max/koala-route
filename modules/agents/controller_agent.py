"""
Agente controlador responsable de coordinar el flujo conversacional en una arquitectura RAG.

Este agente se encarga de preparar los mensajes en formato ChatML que serán enviados
al modelo LLM, integrando la entrada del usuario con el contexto recuperado previamente.
En caso de que la recuperación no proporcione resultados útiles, aplica un prompt de reserva.

Forma parte de la orquestación general y se activa tras la fase de recuperación de información.

Requiere:
- `build_chatml_messages` desde `prompt_utils.py` para ensamblar el contexto.
- `load_prompt` para cargar un prompt alternativo si no hay contexto disponible.
- Un estado (`AgentState`) con al menos `"input"` y opcionalmente `"response"` y `"last_node"`.
"""

from dataclasses import dataclass

from modules.graph.agent_state import AgentState
from modules.prompt_utils import build_chatml_messages, load_prompt


@dataclass
class ControllerAgent:
    """
    Agente controlador que coordina el flujo RAG: recuperación + generación.

    Este agente actúa como orquestador del proceso conversacional, preparando los mensajes
    en formato ChatML que luego serán procesados por el modelo LLM. Su lógica se activa
    principalmente después de la recuperación de contexto (por ejemplo, desde una búsqueda RAG).

    Requiere:
    - `build_chatml_messages` desde `prompt_utils.py` para construir los mensajes ChatML.
    - `load_prompt` para cargar un prompt de reserva en caso de que la recuperación falle.
    - Un estado que contenga al menos el campo `"input"` (entrada del usuario).
    """

    def run(self, state: AgentState) -> AgentState:
        """
        Ejecuta la lógica de control del flujo: prepara los mensajes en formato ChatML,
        especialmente después del nodo de recuperación ("consulta").

        Si el nodo anterior fue "consulta" y no se obtuvo contexto, se carga un prompt de fallback.

        Args:
            state (AgentState): Estado actual del grafo, que debe contener al menos
                                la clave `"input"` con la entrada del usuario.

        Returns:
            AgentState: Nuevo estado actualizado con los mensajes listos en `"response"`.
        """
        response = state.get("response", "")

        if state.get("last_node") == "consulta":
            fallback_prompt = ""
            if not response:
                fallback_prompt = load_prompt("fallback_prompt")
            response = build_chatml_messages(state["input"], response, fallback_prompt)

        return {"response": response}
