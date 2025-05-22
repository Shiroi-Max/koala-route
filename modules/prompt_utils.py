"""
Este módulo contiene utilidades para construir prompts estructurados
específicamente diseñados para modelos de lenguaje tipo Mistral Instruct,
que requieren el formato [INST]...[/INST].
"""

from config.config import NOT_FOUND_MESSAGE

def build_mistral_prompt(user_query: str, context: str) -> str:
    """
    Construye un prompt estructurado para Mistral Instruct 7B.

    Si el contexto proviene del RAG y no contiene información relevante,
    se incluye un system prompt con instrucciones generales.

    Args:
        user_query (str): La pregunta original del usuario.
        context (str): El contexto generado por el RAG.

    Returns:
        str: Prompt en formato ChatML con [INST] ... [/INST]
    """
    if NOT_FOUND_MESSAGE in context:
        # Sin contexto relevante → prompt con instrucciones generales
        prompt = (
            "<s>[INST] "
            f"{NOT_FOUND_MESSAGE}\n\n"
            "Responde a la siguiente pregunta basada en tu conocimiento general:\n"
            f"Pregunta: {user_query} "
            "[/INST]"
        )
    else:
        # Con contexto → prompt instructivo
        prompt = (
            "<s>[INST] "
            f"Dado el siguiente contexto:\n{context}\n\n"
            f"Responde a la siguiente pregunta:\n{user_query} "
            "[/INST]"
        )
    return prompt

def extract_answer(text: str) -> str:
    return text.split("[/INST]")[-1].strip()