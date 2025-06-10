"""
Agente responsable de generar respuestas usando un modelo LLM desplegado en Azure OpenAI.

Este agente utiliza el cliente `AzureOpenAI` para enviar prompts en formato ChatML y obtener
respuestas conversacionales desde el modelo configurado (por ejemplo, GPT-3.5-Turbo).

Requiere:
- `call_openai_chat` desde `llm.py`, que se encarga de llamar a la API de Azure OpenAI.
- Un estado que contenga los mensajes en formato ChatML (preparados previamente por el controlador).
"""

from dataclasses import dataclass
from modules.llm import call_openai_chat
from modules.schema.state_schema import StateSchema


@dataclass
class LLMAgent:
    """
    Agente responsable de comunicarse con Azure OpenAI para generar una respuesta
    basada en los mensajes formateados en el estado.

    Este agente espera que el campo `state.response` contenga una lista de mensajes
    en formato ChatML, lista para enviar al modelo.
    """

    def generate_response(self, state: StateSchema) -> dict:
        """
        Genera una respuesta basada en los mensajes proporcionados.

        Args:
            state (StateSchema): Estado actual del grafo, donde `state.response`
                                 contiene los mensajes en formato ChatML.

        Returns:
            dict: Nuevo estado actualizado con la respuesta generada y control de flujo.
        """
        result = call_openai_chat(state.response)
        return {
            "response": result,
            "next_node": "END",
            "last_node": "llm",
        }
