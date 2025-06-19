"""
Inicializa el cliente de Azure OpenAI para la generación de respuestas conversacionales.

Este módulo realiza lo siguiente:
- Configura el cliente `AzureOpenAI` con las credenciales y endpoint definidos.
- Expone una función para generar respuestas usando el modelo desplegado (como GPT-3.5-Turbo o GPT-4).
- Utiliza el formato ChatML con roles (`system`, `user`, `assistant`) compatible con Azure OpenAI.

Expone:
- `call_openai_chat`: función que recibe una lista de mensajes ChatML y devuelve la respuesta generada
  por el modelo configurado.
"""

from openai import AzureOpenAI

from config.config import (
    API_VERSION,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    MAX_COMPLETION_TOKENS,
)

# Instancia del cliente de Azure OpenAI, configurado con credenciales del entorno
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# Nombre del modelo/despliegue definido en Azure (ej. "gpt-35-turbo")
deployment_name = AZURE_OPENAI_DEPLOYMENT


def call_openai_chat(prompt_messages: list[dict]) -> str:
    """
    Genera una respuesta a partir de una lista de mensajes en formato ChatML.

    Esta función se comunica con Azure OpenAI para obtener una respuesta del modelo
    configurado, utilizando los parámetros definidos en el entorno (como temperatura
    y número máximo de tokens).

    Args:
        prompt_messages (list[dict]): Lista de mensajes con estructura ChatML,
                                      incluyendo roles como 'system', 'user', 'assistant'.

    Returns:
        str: Contenido de la respuesta generada por el modelo.
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=prompt_messages,
        temperature=0.7,
        max_tokens=MAX_COMPLETION_TOKENS,
    )
    return response.choices[0].message.content
