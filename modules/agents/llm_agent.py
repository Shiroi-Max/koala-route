"""
Agente LLM responsable de generar respuestas utilizando un modelo de lenguaje desplegado en Azure OpenAI.

Este agente recibe una lista de mensajes en formato ChatML (almacenados en el estado) y
utiliza la función `call_openai_chat` para obtener una respuesta conversacional del modelo,
como GPT-3.5-Turbo o GPT-4, configurado mediante Azure.

Forma parte de una arquitectura RAG, ejecutándose típicamente después de la preparación del prompt
por parte del controlador.

Requiere:
- `call_openai_chat` desde `llm.py` para realizar la llamada a la API de Azure OpenAI.
- Un estado (`AgentState`) que contenga la clave `"response"` con los mensajes ChatML.
"""

from dataclasses import dataclass

from modules.graph.agent_state import AgentState
from modules.llm import call_openai_chat


@dataclass
class LLMAgent:
    """
    Agente responsable de comunicarse con Azure OpenAI para generar una respuesta
    basada en los mensajes formateados en el estado.

    Este agente espera que el campo `state.get("response")` contenga una lista de mensajes
    en formato ChatML, lista para enviar al modelo.
    """

    def generate_response(self, state: AgentState) -> AgentState:
        """
        Genera una respuesta basada en los mensajes proporcionados.

        Args:
            state (AgentState): Estado actual del grafo, donde `state.get("response")`
                                contiene los mensajes en formato ChatML.

        Returns:
            AgentState: Nuevo estado actualizado con la respuesta generada y control de flujo.
        """
        result = call_openai_chat(state.get("response"))
        return {
            "response": result,
            "last_node": "llm",
        }
