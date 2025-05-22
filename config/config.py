"""
Configuración central del proyecto RAG + LLM.

Este módulo define:
- Las claves necesarias para conectarse a Azure Cognitive Search.
- Los identificadores de los modelos de embeddings y LLM.
- Los nombres de índices y rutas de documentos utilizadas en el flujo principal.
"""

import os

# ---------- CREDENCIALES DE AZURE ----------
# Endpoints y claves API (deben definirse como variables de entorno)
os.environ["AZURE_SEARCH_ENDPOINT"] = (
    ""
)
os.environ["AZURE_SEARCH_KEY"] = ""

# ---------- PARÁMETROS GENERALES ----------
DOCS_PATH = "data"  # Carpeta local con los archivos .txt
INDEX_NAME = "docs"  # Nombre del índice en Azure Cognitive Search

# ---------- MODELOS ----------
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Modelo de embeddings
LLM_MODEL_ID = "google/gemma-2b-it"  # Modelo LLM final

# ---------- LECTURA DE VARIABLES DE ENTORNO ----------
AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
