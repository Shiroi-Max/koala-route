"""
Este módulo contiene utilidades para construir prompts estructurados compatibles con modelos de chat de OpenAI.

Funciones principales:
- Construcción de mensajes en formato ChatML (`build_chatml_messages`).
- Carga de prompts desde un archivo YAML (`load_prompt`, `load_formatted_prompt`).
- Extracción de intereses del usuario desde un prompt personalizado (`extract_user_interests_from_prompt`).

Todos los mensajes siguen el formato esperado por modelos como `gpt-3.5-turbo` o `gpt-4` cuando se usa la API de Azure OpenAI.
"""

import re
import yaml
from tiktoken import get_encoding

from config.config import ENCODING_NAME, PROMPT_PATH

# Codificador de tokens para medir longitud de prompts
encoding = get_encoding(ENCODING_NAME)


def build_chatml_messages(
    user_query: str, context: str = "", system_prompt: str = ""
) -> list[dict]:
    """
    Construye una lista de mensajes en formato ChatML (role-content) para usar en un modelo OpenAI.

    Si se proporciona contexto (por ejemplo, desde recuperación RAG), se incorpora en la instrucción.
    También puede añadirse un prompt del sistema adicional (por ejemplo, restricciones de tono o formato).

    Args:
        user_query (str): Pregunta del usuario.
        context (str): Texto contextual obtenido mediante recuperación (opcional).
        system_prompt (str): Instrucciones adicionales del sistema (opcional).

    Returns:
        list[dict]: Lista de mensajes con roles "system" y "user", listos para enviar al modelo.
    """
    # Cargar prompt base del sistema
    full_system_prompt = load_prompt("default_system_prompt")
    if system_prompt:
        full_system_prompt += f"\n\n📝 Instrucciones adicionales:\n{system_prompt}"

    messages = [{"role": "system", "content": full_system_prompt}]

    # Incluir contexto si está disponible
    if context:
        user_content = f"Dado el siguiente contexto:\n{context}\n\nResponde a la pregunta:\n{user_query}"
    else:
        user_content = user_query

    messages.append({"role": "user", "content": user_content})
    return messages


def load_prompt(key: str) -> str:
    """
    Carga un prompt desde el archivo YAML a partir de una clave.

    Args:
        key (str): Clave del prompt a recuperar.

    Returns:
        str: Texto del prompt correspondiente.

    Raises:
        KeyError: Si la clave no existe en el archivo YAML.
    """
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if key not in data:
        raise KeyError(f"❌ Clave de prompt '{key}' no encontrada en {PROMPT_PATH}")
    return data[key]


def load_formatted_prompt(key: str, **kwargs) -> str:
    """
    Carga un prompt desde el YAML y lo formatea con variables dinámicas.

    Args:
        key (str): Clave del prompt a recuperar.
        **kwargs: Variables para formatear el template (como {days}, {interests}, etc.).

    Returns:
        str: Prompt con las variables ya reemplazadas.
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
    Extrae los intereses del usuario desde un prompt ya formateado.

    Esta función compara un template con marcadores (ej. `{interests}`) y su versión
    ya rellenada, extrayendo el contenido que fue insertado dinámicamente en el campo `interests`.

    Args:
        prompt_template (str): Plantilla del prompt, con el marcador `{interests}`.
        filled_prompt (str): Prompt real con los datos ya sustituidos.

    Returns:
        list[str]: Lista de intereses (minúsculas y sin espacios).
    """
    # Escapar caracteres especiales del template para usarlo como regex
    pattern = re.escape(prompt_template)

    # Reemplazar {interests} por grupo de captura
    pattern = pattern.replace(re.escape("{interests}"), r"(?P<interests>.+?)")

    # Reemplazar otros marcadores como {days}, {budget}, etc. por comodines
    pattern = re.sub(r"\{[^{}]+\}", r".+?", pattern)

    # Buscar el contenido de interests dentro del prompt ya instanciado
    match = re.search(pattern, filled_prompt, re.DOTALL)
    if match:
        interests_str = match.group("interests")
        return [i.strip().lower() for i in interests_str.split(",") if i.strip()]
    return []
