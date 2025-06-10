"""
Este módulo contiene utilidades para construir prompts estructurados
"""

from config.config import ENCODING_NAME
from tiktoken import get_encoding

encoding = get_encoding(ENCODING_NAME)


def build_chatml_messages(
    user_query: str, context: str = "", system_prompt: str = ""
) -> list[dict]:
    """
    Construye una lista de mensajes en formato ChatML para Mistral 7B Instruct v0.3.

    Args:
        user_query (str): Pregunta del usuario.
        context (str): Texto del RAG si existe.
        system_prompt (str): Mensaje de rol del sistema.

    Returns:
        list[dict]: Mensajes en formato ChatML con roles: system, user.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if context:
        user_content = f"Dado el siguiente contexto:\n{context}\n\nResponde a la pregunta:\n{user_query}"
    else:
        user_content = user_query

    messages.append({"role": "user", "content": user_content})
    return messages


def count_tokens(messages: list[dict]) -> int:
    """
    Cuenta la cantidad total de tokens en un mensaje tipo ChatML.

    Args:
        messages (list): Lista de mensajes en formato [{"role": ..., "content": ...}]

    Returns:
        int: Número total de tokens que se enviarían al modelo.
    """
    num_tokens = 0
    for msg in messages:
        # Cada mensaje lleva tokens fijos + contenido
        num_tokens += 4  # tokens por role + content + delimitadores
        for _, value in msg.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # tokens extra para el assistant que sigue
    return num_tokens
