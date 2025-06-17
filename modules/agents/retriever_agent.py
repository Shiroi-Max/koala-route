from dataclasses import dataclass

from langchain_community.vectorstores import AzureSearch
from sklearn.metrics.pairwise import cosine_similarity

from config.config import RETRIEVER_K, SIMILARITY_THRESHOLD
from modules.graph.agent_state import AgentState
from modules.prompt_utils import (
    extract_user_interests_from_prompt,
    load_prompt,
)


@dataclass
class RetrieverAgent:
    """
    Agente responsable de recuperar documentos relevantes desde un almacén vectorial
    según una consulta del usuario y un umbral de similitud.

    Atributos:
        vector_store: Objeto vectorial con `similarity_search` y `embedding_function`.
    """

    vector_store: AzureSearch

    def get_context(self, state: AgentState) -> AgentState:
        """
        Recupera documentos relevantes para una consulta y los filtra por similitud y secciones útiles según los intereses del usuario.

        Args:
            state (AgentState): Estado actual del grafo con la entrada del usuario.

        Returns:
            AgentState: Estado actualizado con la respuesta contextual y el nodo actual.
        """
        result = ""

        prompt_template = load_prompt("prompt_base")
        user_query = state["input"]
        user_interests = extract_user_interests_from_prompt(prompt_template, user_query)
        user_interests_normalized = [i.lower() for i in user_interests]

        docs = self.vector_store.similarity_search(user_query, k=RETRIEVER_K)
        query_embedding = self.vector_store.embedding_function(user_query)

        section_filtered_docs = []
        for doc in docs:
            section = doc.metadata.get("section", "").lower()
            if not user_interests_normalized or any(
                interest in section for interest in user_interests_normalized
            ):
                section_filtered_docs.append(doc)

        if section_filtered_docs:
            relevant, similarities = [], []
            for doc in section_filtered_docs:
                title = doc.metadata.get("title", "")
                combined_text = f"{title}. {doc.page_content}"
                doc_embedding = self.vector_store.embedding_function(combined_text)
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                if similarity >= SIMILARITY_THRESHOLD:
                    relevant.append(doc)
                    similarities.append(similarity)

            if relevant:
                result_sections, resumen = [], []
                for doc, sim in zip(relevant, similarities):
                    title = doc.metadata.get("title", "Sin título")
                    section = doc.metadata.get("section", "Sin sección")
                    content = doc.page_content.strip()

                    resumen.append(f"{title} > {section} (sim={sim:.2f})")
                    result_sections.append(f"## {title} > {section}\n\n{content}")

                print(f"🔍 Recuperados {len(relevant)} documentos relevantes")
                print("🔍 Documentos recuperados:", ", ".join(resumen))

                result = "\n\n".join(result_sections)
        return {
            "response": result,
            "last_node": "consulta",
        }
