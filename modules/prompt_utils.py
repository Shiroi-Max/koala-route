"""
Este módulo contiene utilidades para construir prompts estructurados
"""

import re
import yaml
from tiktoken import get_encoding

from config.config import ENCODING_NAME, PROMPT_PATH

encoding = get_encoding(ENCODING_NAME)


def build_chatml_messages(
    user_query: str, context: str = "", system_prompt: str = ""
) -> list[dict]:
    """
    Construye una lista de mensajes en formato OpenAI Chat (ChatML)
    compatibles con modelos como gpt-3.5-turbo o gpt-4.

    Args:
        user_query (str): Pregunta del usuario.
        context (str): Texto del sistema de recuperación (RAG), si existe.
        system_prompt (str): Instrucciones del sistema (opcional). Si no se proporciona, se usa uno por defecto.

    Returns:
        list[dict]: Lista de mensajes con formato role-content esperados por la API de chat OpenAI.
    """

    full_system_prompt = load_prompt("default_system_prompt")
    if system_prompt:
        full_system_prompt += f"\n\n📝 Instrucciones adicionales:\n{system_prompt}"

    messages = [{"role": "system", "content": full_system_prompt}]

    if context:
        user_content = f"Dado el siguiente contexto:\n{context}\n\nResponde a la pregunta:\n{user_query}"
    else:
        user_content = user_query

    messages.append({"role": "user", "content": user_content})
    return messages


def load_prompt(key: str) -> str:
    """
    Carga un prompt del sistema por clave desde el archivo YAML.

    Args:
        key (str): Clave del prompt a recuperar.

    Returns:
        str: Texto del prompt solicitado.

    Raises:
        KeyError: Si la clave no existe en el archivo.
    """
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if key not in data:
        raise KeyError(f"❌ Clave de prompt '{key}' no encontrada en {PROMPT_PATH}")
    return data[key]


def load_formatted_prompt(key: str, **kwargs) -> str:
    """
    Carga un prompt del YAML y aplica formateo dinámico con kwargs.

    Args:
        key (str): Clave del prompt.
        **kwargs: Variables para sustituir en el template.

    Returns:
        str: Prompt formateado.
    """
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    template = data.get(key)
    if not template:
        raise KeyError(f"❌ Clave '{key}' no encontrada en el YAML.")

    return template.format(**kwargs)


def extract_user_interests_from_prompt(
    prompt_template: str, filled_prompt: str
) -> list[str]:
    """
    Extrae los intereses desde un prompt personalizado y su plantilla.

    Args:
        prompt_template (str): Plantilla con marcador `{interests}`.
        filled_prompt (str): Prompt ya formateado con los valores reales.

    Returns:
        list[str]: Lista de intereses extraídos.
    """
    # Escapamos caracteres especiales en la plantilla para usar como regex
    pattern = re.escape(prompt_template)

    # Reemplazamos el marcador {interests} por un grupo de captura (.*?)
    pattern = pattern.replace(re.escape("{interests}"), r"(?P<interests>.+?)")

    # Reemplazamos otros marcadores por comodines para que no molesten
    pattern = re.sub(r"\{[^{}]+\}", r".+?", pattern)

    # Buscamos usando re.DOTALL por si hay saltos de línea
    match = re.search(pattern, filled_prompt, re.DOTALL)
    if match:
        interests_str = match.group("interests")
        return [i.strip().lower() for i in interests_str.split(",") if i.strip()]
    return []
