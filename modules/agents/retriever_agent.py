"""
Agente responsable de recuperar contexto relevante desde un almacén vectorial usando búsqueda por similitud.

Este agente forma parte de la arquitectura RAG (Retrieval-Augmented Generation) y se encarga de:
- Realizar una búsqueda semántica en Azure Cognitive Search (vía `langchain_community.vectorstores.AzureSearch`).
- Filtrar los documentos según los intereses del usuario y la sección del contenido.
- Calcular similitudes con embeddings para aplicar un umbral de relevancia.
- Devolver las secciones útiles para ser usadas como contexto en la generación de respuestas.

Requiere:
- `RETRIEVER_K` y `SIMILARITY_THRESHOLD` definidos en `config`.
- Un vector store que implemente `.similarity_search` y `.embedding_function`.
- Funciones de `prompt_utils`: `extract_user_interests_from_prompt`, `load_prompt`.
- Un estado (`AgentState`) que contenga la entrada del usuario en `"input"`.
"""

from dataclasses import dataclass

from langchain_community.vectorstores import AzureSearch
from sklearn.metrics.pairwise import cosine_similarity

from config.config import RETRIEVER_K, SIMILARITY_THRESHOLD
from modules.graph.agent_state import AgentState
from modules.prompt_utils import extract_user_interests_from_prompt


@dataclass
class RetrieverAgent:
    """
    Agente responsable de recuperar documentos relevantes desde un almacén vectorial
    según una consulta del usuario y un umbral de similitud.

    Este agente filtra las secciones por interés temático y relevancia semántica antes de
    entregarlas como contexto al siguiente paso del flujo (por ejemplo, generación con LLM).

    Atributos:
        vector_store (AzureSearch): Almacén vectorial con métodos de búsqueda y embeddings.
    """

    vector_store: AzureSearch

    def get_context(self, state: AgentState) -> AgentState:
        """
        Recupera documentos relevantes para una consulta del usuario y filtra los resultados
        por similitud semántica y coincidencia temática con los intereses extraídos del prompt.

        - Realiza una búsqueda semántica con `similarity_search`.
        - Filtra secciones cuyo campo `metadata["section"]` coincida con los intereses.
        - Calcula la similitud con el embedding de la consulta y aplica un umbral.
        - Devuelve las secciones relevantes en el campo `"response"` del estado.

        Args:
            state (AgentState): Estado actual del grafo que debe incluir `"input"` con la consulta del usuario.

        Returns:
            AgentState: Estado actualizado con el contexto relevante en `"response"` y `"last_node"` marcado como `"consulta"`.
        """
        result = ""  # Inicializamos el resultado (bloque de contexto)
        retrieved_docs = []  # Lista para almacenar documentos recuperados

        # Consulta del usuario desde el estado
        user_query = state["input"]

        # Extraemos los intereses desde el prompt rellenado
        user_interests = extract_user_interests_from_prompt(user_query)

        # Normalizamos los intereses a minúsculas para hacer matching más robusto
        user_interests_normalized = [i.lower() for i in user_interests]

        # Búsqueda semántica en el vector store (Azure Search), top-k resultados
        docs = self.vector_store.similarity_search(user_query, k=RETRIEVER_K)

        # Calculamos el embedding del usuario para medir similitud más adelante
        query_embedding = self.vector_store.embedding_function(user_query)

        # Filtrado por coincidencia temática en metadatos de sección
        category_filtered_docs = []
        for doc in docs:
            categories = [c.lower() for c in doc.metadata.get("category", [])]
            
            if not user_interests_normalized or any(
                interest in categories for interest in user_interests_normalized
            ):
                category_filtered_docs.append(doc)

        # Si hay documentos que pasaron el primer filtro, evaluamos la similitud exacta
        if category_filtered_docs:
            relevant, similarities = [], []
            for doc in category_filtered_docs:
                # Combinamos título y contenido para obtener el embedding final
                title = doc.metadata.get("title", "")
                combined_text = f"{title}. {doc.page_content}"
                doc_embedding = self.vector_store.embedding_function(combined_text)

                # Similitud coseno entre consulta y documento
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                
                # Si supera el umbral, lo consideramos relevante
                if similarity >= SIMILARITY_THRESHOLD:
                    relevant.append(doc)
                    similarities.append(similarity)

            # Si hay documentos relevantes, los estructuramos en Markdown
            if relevant:
                result_sections = []
                retrieved_docs = []

                for doc in relevant:
                    title = doc.metadata.get("title", "Sin título")
                    section = doc.metadata.get("section", "Sin sección")
                    categories = doc.metadata.get("category", [])

                    content = doc.page_content.strip()
                    result_sections.append(f"## {title} > {section}\n\n{content}")

                    retrieved_docs.append({
                        "id": f"{title}#{section}",
                        "category": categories,
                    })

                # Unimos las secciones en un solo bloque de texto
                result = "\n\n".join(result_sections)

        # Devolvemos el estado actualizado con el contexto, el nodo actual y los documentos recuperados
        return {
            "response": result,
            "last_node": "consulta",
            "retrieved_docs": retrieved_docs,
        }
