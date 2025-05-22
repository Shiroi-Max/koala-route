"""
Módulo para cargar documentos .txt en Azure Cognitive Search con embeddings.

Este módulo:
- Lee el contenido de un archivo .txt especificado por el usuario.
- Genera su vector de embedding utilizando el vector_store existente.
- Envía el documento al índice vectorial en Azure Search.

Uso:
    python uploader.py --file info.txt --title "LangChain Overview"
"""

import os
import uuid
import argparse
from typing import Optional
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from config.config import AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, INDEX_NAME, DOCS_PATH
from modules.embeddings import vector_store


def upload_txt_document(file_name: str, title: Optional[str] = None):
    """
    Carga un archivo .txt al índice de Azure Search con su embedding.

    Args:
        file_name (str): Nombre del archivo dentro de `DOCS_PATH` (ej. "info.txt").
        title (Optional[str]): Título opcional del documento. Si no se especifica, se usará el nombre del archivo.

    Returns:
        Any: Resultado de la operación `upload_documents()`.
    """
    file_path = os.path.join(DOCS_PATH, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"❌ Archivo no encontrado: {file_path}")

    # Leer contenido
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Generar embedding
    embedding = vector_store.embedding_function(content)

    # Preparar documento
    document = {
        "id": str(uuid.uuid4()),
        "content": content,
        "content_vector": embedding,
        "source": file_path,
        "title": title or os.path.splitext(file_name)[0],
    }

    # Subir a Azure Search
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    result = search_client.upload_documents(documents=[document])
    print(f"✅ Documento '{file_name}' subido con éxito.")
    return result


def main():
    """
    Permite al usuario subir un documento especificando los parámetros por línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Sube un archivo .txt a Azure Cognitive Search."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Nombre del archivo .txt dentro de la carpeta docs/",
    )
    parser.add_argument("--title", help="Título opcional del documento")

    args = parser.parse_args()

    try:
        upload_txt_document(file_name=args.file, title=args.title)
    except Exception as e:
        print(f"❌ Error al subir el documento: {e}")


if __name__ == "__main__":
    main()
