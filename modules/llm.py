"""
Inicializa el cliente de Azure OpenAI para la generación de respuestas conversacionales.

Este módulo realiza lo siguiente:
- Configura el cliente `AzureOpenAI` con las credenciales y endpoint definidos.
- Expone una función para generar respuestas usando el modelo desplegado (como GPT-3.5-Turbo).
- Utiliza el formato ChatML con roles (`system`, `user`, etc.) compatible con Azure OpenAI.

Expone:
- `call_openai_chat`: función que recibe una lista de mensajes y devuelve una respuesta generada
  por el modelo configurado en Azure OpenAI.
"""

from openai import AzureOpenAI
from config.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    MAX_COMPLETION_TOKENS,
)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

deployment_name = AZURE_OPENAI_DEPLOYMENT


def call_openai_chat(prompt_messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=deployment_name, messages=prompt_messages, temperature=0.7, max_tokens=MAX_COMPLETION_TOKENS
    )
    return response.choices[0].message.content
