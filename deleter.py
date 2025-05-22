"""
Módulo para eliminar documentos del índice de Azure Cognitive Search por ID.

Uso:
    python deleter.py --id 2beebada-685e-4fdd-97b1-38a83f093250
    python deleter.py --id id1 --id id2
"""

import argparse
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from config.config import AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, INDEX_NAME


def delete_documents_by_id(document_ids: list[str]):
    """
    Elimina uno o varios documentos del índice de Azure Search por sus ID.

    Args:
        document_ids (list[str]): Lista de IDs de documentos a eliminar.

    Returns:
        Any: Resultado de la operación `upload_documents()` con acción 'delete'.
    """
    batch = [{"@search.action": "delete", "id": doc_id} for doc_id in document_ids]

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    result = search_client.upload_documents(documents=batch)
    print(f"✅ Eliminados {len(document_ids)} documento(s).")
    return result


def main():
    """
    Ejecuta la eliminación desde CLI permitiendo varios IDs como argumentos.
    """
    parser = argparse.ArgumentParser(
        description="Elimina documentos por ID del índice de Azure Search."
    )
    parser.add_argument(
        "--id", required=True, nargs="+", help="Uno o más IDs de documentos a eliminar."
    )

    args = parser.parse_args()

    try:
        delete_documents_by_id(args.id)
    except HttpResponseError as e:
        print(f"❌ Error de respuesta HTTP al eliminar documentos: {e}")
    except Exception as e:
        print(f"❌ Error inesperado al eliminar documentos: {e}")


if __name__ == "__main__":
    main()
