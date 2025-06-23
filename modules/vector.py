"""
Módulo para inicializar los embeddings y el vector store de Azure Cognitive Search.

Este módulo configura dos componentes esenciales para la fase de recuperación semántica (RAG):

1. `embeddings`: Modelo de embeddings de Azure OpenAI, encargado de convertir textos en vectores
   utilizando un despliegue configurado del modelo `text-embedding-3-large` (u otro compatible).
2. `vector_store`: Objeto `AzureSearch` de LangChain configurado para realizar búsquedas híbridas
   (semánticas + léxicas) sobre un índice existente en Azure Cognitive Search.

Exporta:
- `vector_store`: Instancia lista para ser utilizada por el agente de recuperación (`RetrieverAgent`).
"""

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch

from config.config import (
    AZURE_OPENAI_EMBEDDINGS_API_KEY,
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDINGS_ENDPOINT,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    INDEX_NAME,
    API_VERSION_EMBEDDINGS,
)

# ---------- EMBEDDINGS ----------
# Inicializa el modelo de embeddings usando Azure OpenAI.
# Este modelo convierte las consultas y los documentos en vectores numéricos (embeddings),
# que posteriormente se utilizan para realizar búsquedas semánticas en Azure Cognitive Search.
# El despliegue del modelo debe haberse realizado previamente en Azure OpenAI y configurado en el sistema.
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_EMBEDDINGS_ENDPOINT,
    openai_api_key=AZURE_OPENAI_EMBEDDINGS_API_KEY,
    openai_api_type="azure",
    openai_api_version=API_VERSION_EMBEDDINGS,
)


# ---------- VECTOR STORE ----------
# Configura el almacén vectorial utilizando Azure Cognitive Search.
# Este objeto permite realizar búsquedas por similitud utilizando los vectores generados por el modelo de embeddings.
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=INDEX_NAME,
    embedding_function=embeddings.embed_query,
)
