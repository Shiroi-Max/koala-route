"""
Agentes de recuperación y generación para el flujo RAG.

Este módulo define:
- `rag_agent`: recuperación de contexto relevante usando un retriever semántico.
- `llm_agent`: generación de respuesta basada en un prompt contextual.
- `controller_agent`: orquestador que combina ambos pasos para responder una consulta del usuario.
"""

from modules.llm import llm
from modules.embeddings import retriever


def rag_agent(query: str) -> str:
    """
    Agente RAG que recupera documentos relevantes desde el índice vectorial.

    Args:
        query (str): La pregunta o consulta del usuario.

    Returns:
        str: Texto combinado de los documentos recuperados.
    """
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


def llm_agent(prompt: str) -> str:
    """
    Agente LLM que genera una respuesta basada en el prompt y contexto dado.

    Args:
        prompt (str): Prompt que contiene la pregunta y el contexto.

    Returns:
        str: Respuesta generada por el modelo de lenguaje.
    """
    print(prompt)
    return llm.invoke(prompt)


def controller_agent(user_query: str) -> str:
    """
    Agente controlador que orquesta recuperación (RAG) y generación de respuesta final.

    Args:
        user_query (str): Consulta original del usuario.

    Returns:
        str: Respuesta final generada por el modelo, basada en el contexto recuperado.
    """
    context = rag_agent(user_query)

    if len(context.split()) > 400:
        context = " ".join(context.split()[:400])

    print("Agente Controlador pregunta al Agente LLM:")
    prompt = f"Dado el siguiente contexto:\n{context}\n\nResponde a la pregunta:\n{user_query}"

    response = llm_agent(prompt)
    return response
