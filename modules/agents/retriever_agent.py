from dataclasses import dataclass
from langchain_community.vectorstores import AzureSearch
from sklearn.metrics.pairwise import cosine_similarity
from modules.schema.state_schema import StateSchema


@dataclass
class RetrieverAgent:
    """
    Agente responsable de recuperar documentos relevantes desde un almacén vectorial
    según una consulta del usuario y un umbral de similitud.

    Atributos:
        vector_store: Objeto vectorial con `similarity_search` y `embedding_function`.
    """

    vector_store: AzureSearch

    def get_context(
        self, state: StateSchema, similarity_threshold: float = 0.3, k: int = 3
    ) -> dict:
        """
        Recupera documentos relevantes para una consulta y los filtra por similitud.

        Args:
            query: Consulta del usuario.
            similarity_threshold: Umbral mínimo de similitud para considerar un documento relevante.
            k: Número máximo de documentos a recuperar.

        Returns:
            Contexto concatenado de los documentos relevantes o mensaje alternativo.
        """
        docs = self.vector_store.similarity_search(state.input, k=k)
        query_embedding = self.vector_store.embedding_function(state.input)

        relevant = []
        for doc in docs:
            doc_embedding = self.vector_store.embedding_function(doc.page_content)
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            if similarity >= similarity_threshold:
                relevant.append(doc)

        if not relevant:
            result = ""
        else:
            result = "\n\n".join(doc.page_content for doc in relevant)

        return {
            "response": result,
            "next_node": "controlador",
            "last_node": "consulta",
        }
