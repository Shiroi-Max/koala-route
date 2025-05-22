"""
Módulo para inicializar los embeddings y el vector store de Azure Search.

Este módulo expone:
- `retriever`: un objeto LangChain que permite recuperar documentos relevantes
  mediante búsqueda semántica sobre el índice configurado.
"""

from langchain_community.vectorstores import AzureSearch
from langchain_huggingface import HuggingFaceEmbeddings
from config.config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    INDEX_NAME,
    EMBEDDING_MODEL_ID,
)

# ---------- EMBEDDINGS ----------
# Se inicializa el modelo de embeddings de HuggingFace
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)

# ---------- VECTOR STORE ----------
# Se configura el almacén vectorial de Azure Search con el índice deseado
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=INDEX_NAME,
    embedding_function=embeddings.embed_query,
)