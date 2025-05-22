"""
Agentes para orquestar el flujo RAG:
1. `rag_agent`: Recupera documentos relevantes y filtra por similitud.
2. `llm_agent`: Genera respuesta basada en el contexto recuperado.
3. `controller_agent`: Controlador que coordina todo.
"""

from modules.llm import pipe
from modules.embeddings import vector_store
from modules.prompt_utils import build_mistral_prompt, extract_answer
from config.config import NOT_FOUND_MESSAGE
from sklearn.metrics.pairwise import cosine_similarity


def rag_agent(query: str, similarity_threshold: float = 0.3) -> str:
    """
    Recupera documentos relevantes para una consulta y los filtra por similitud.

    Args:
        query (str): Consulta del usuario.
        retriever: Objeto retriever con método `invoke(query)`.
        embedding_function: Función que genera el embedding de un texto.
        similarity_threshold (float): Umbral mínimo de similitud.

    Returns:
        str: Contexto concatenado o mensaje indicando falta de relevancia.
    """
    docs = vector_store.similarity_search(query, k=3)
    query_embedding = vector_store.embedding_function(query)

    relevantes  = []
    for doc in docs:
        doc_embedding = vector_store.embedding_function(doc.page_content)
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        if similarity >= similarity_threshold:
            relevantes.append(doc)

    if not relevantes:
        return NOT_FOUND_MESSAGE

    return "\n\n".join(doc.page_content for doc in relevantes)


def llm_agent(user_query: str, context: str) -> str:
    """
    Genera una respuesta con el LLM a partir de la consulta y el contexto.

    Si no hay contexto, se instruye al modelo para que responda usando solo su conocimiento general.

    Args:
        user_query (str): Consulta original del usuario.
        context (str): Contexto proporcionado por el RAG.
        llm: Objeto LLM con método `invoke`.

    Returns:
        str: Respuesta generada.
    """
    prompt = build_mistral_prompt(user_query, context)
    result = pipe(prompt)[0]["generated_text"]
    return extract_answer(result)


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

    return llm_agent(user_query, context)
