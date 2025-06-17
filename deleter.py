"""
Módulo para eliminar documentos del índice de Azure Cognitive Search por ID.

Uso:
    python deleter.py --id 2beebada-685e-4fdd-97b1-38a83f093250
    python deleter.py --id id1 id2
    python deleter.py --all
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
    """
    if not document_ids:
        print("ℹ️ No hay IDs para eliminar.")
        return

    batch = [{"@search.action": "delete", "id": doc_id} for doc_id in document_ids]

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    search_client.upload_documents(documents=batch)
    print(f"✅ Eliminados {len(document_ids)} documento(s).")


def delete_all_documents():
    """
    Recupera todos los IDs del índice y los elimina.
    """
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    print("🔍 Recuperando todos los IDs del índice...")
    all_ids = []
    results = search_client.search(search_text="*", select=["id"], top=1000)
    for result in results:
        all_ids.append(result["id"])

    if not all_ids:
        print("ℹ️ No se encontraron documentos en el índice.")
        return

    # Borrar en bloques de 1000 usando la misma función
    chunk_size = 1000
    for i in range(0, len(all_ids), chunk_size):
        chunk = all_ids[i : i + chunk_size]
        delete_documents_by_id(chunk)

    print(f"🎉 Eliminación completada. Total eliminados: {len(all_ids)}")


def main():
    """
    Ejecuta la eliminación desde CLI permitiendo múltiples IDs o eliminar todo con --all.
    """
    parser = argparse.ArgumentParser(
        description="Elimina documentos del índice de Azure Search."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--id", nargs="+", help="Uno o más IDs de documentos a eliminar."
    )
    group.add_argument(
        "--all", action="store_true", help="Elimina todos los documentos del índice."
    )

    args = parser.parse_args()

    try:
        if args.all:
            delete_all_documents()
        else:
            delete_documents_by_id(args.id)
    except HttpResponseError as e:
        print(f"❌ Error de respuesta HTTP al eliminar documentos: {e}")
    except Exception as e:
        print(f"❌ Error inesperado al eliminar documentos: {e}")


if __name__ == "__main__":
    main()
