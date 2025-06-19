"""
Módulo para inicializar los embeddings y el vector store de Azure Cognitive Search.

Este módulo configura dos componentes esenciales para la fase de recuperación semántica (RAG):

1. `embeddings`: Modelo de embeddings basado en HuggingFace, usado para convertir textos en vectores.
2. `vector_store`: Objeto `AzureSearch` de LangChain configurado para buscar documentos relevantes
   a través de similitud semántica sobre un índice existente en Azure.

Exporta:
- `vector_store`: Instancia lista para ser usada por el agente de recuperación (`RetrieverAgent`).
"""

from langchain_community.vectorstores import AzureSearch
from langchain_huggingface import HuggingFaceEmbeddings

from config.config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    EMBEDDING_MODEL_ID,
    INDEX_NAME,
)

# ---------- EMBEDDINGS ----------
# Inicializa el modelo de embeddings desde HuggingFace usando el identificador configurado.
# Este modelo se encargará de convertir consultas y documentos en vectores numéricos.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)

# ---------- VECTOR STORE ----------
# Configura el almacén vectorial utilizando Azure Cognitive Search.
# Este objeto permite realizar búsquedas por similitud utilizando los vectores generados por el modelo de embeddings.
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=INDEX_NAME,
    embedding_function=embeddings.embed_query,
)
