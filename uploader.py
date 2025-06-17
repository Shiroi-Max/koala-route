"""
Módulo para cargar documentos .md en Azure Cognitive Search con embeddings.

Este módulo:
- Lee el contenido de un archivo .txt especificado por el usuario.
- Genera su vector de embedding utilizando el vector_store existente.
- Envía el documento al índice vectorial en Azure Search.

Uso:
    python uploader.py --file info.md --title "City information"
"""

import os
import uuid
import re
import argparse
from typing import Optional
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from config.config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    INDEX_NAME,
    DOCS_PATH,
    SECTION_TO_CATEGORIES,
)
from modules.vector import vector_store


def split_markdown_sections(text: str, min_length: int = 200) -> list[tuple[str, str]]:
    """
    Divide un documento Markdown en secciones por encabezado.

    Args:
        text (str): Contenido completo del archivo Markdown.
        min_length (int): Longitud mínima de cada sección.

    Returns:
        list[tuple[str, str]]: Lista de tuplas (section_title, section_body).
    """
    pattern = r"(?=\n*#+ )"
    raw_chunks = re.split(pattern, text)
    sections = []

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if len(chunk) < min_length:
            continue

        lines = chunk.splitlines()
        header = lines[0].lstrip("#").strip() if lines else "Sin título"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        if len(body) >= min_length:
            sections.append((header or "Sin título", body))

    return sections


def upload_md_document(file_name: str, title: Optional[str] = None):
    """
    Carga un archivo .md al índice de Azure Search con sus secciones vectorizadas.

    Args:
        file_name (str): Nombre del archivo Markdown en DOCS_PATH.
        title (Optional[str]): Título principal del documento.
    """
    file_path = os.path.join(DOCS_PATH, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"❌ Archivo no encontrado: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    section_chunks = split_markdown_sections(content)

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    documents = []
    for i, (section_title, section_text) in enumerate(section_chunks, start=1):
        embedding = vector_store.embedding_function(section_text)
        categories = SECTION_TO_CATEGORIES.get(section_title.strip(), [])

        documents.append(
            {
                "id": str(uuid.uuid4()),
                "title": (title or os.path.splitext(file_name)[0]).capitalize(),
                "section": section_title,
                "category": categories,  # ← Añadido
                "content": section_text,
                "content_vector": embedding,
                "source": f"{file_path}#section-{i}",
            }
        )

    result = search_client.upload_documents(documents=documents)
    print(f"✅ Subidas {len(documents)} secciones de '{file_name}'")
    return result


def upload_all_md_documents():
    """
    Sube todos los archivos .md encontrados en DOCS_PATH al índice de Azure Search.

    Itera sobre todos los ficheros .md en la carpeta especificada,
    ejecutando `upload_md_document()` sobre cada uno de ellos.
    """
    md_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".md")]

    if not md_files:
        print("⚠️ No se encontraron archivos .md en la carpeta DOCS_PATH.")
        return

    for file_name in md_files:
        try:
            print(f"⬆️ Subiendo '{file_name}'...")
            upload_md_document(file_name=file_name)
        except Exception as e:
            print(f"❌ Error al subir '{file_name}': {e}")


def main():
    """
    Permite al usuario subir uno o todos los documentos Markdown mediante argumentos CLI.
    """
    parser = argparse.ArgumentParser(
        description="Sube uno o varios archivos .md a Azure Cognitive Search con división por secciones."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        help="Nombre del archivo .md dentro de la carpeta data/ (ej: info.md)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Sube todos los archivos .md dentro de la carpeta data/",
    )
    parser.add_argument(
        "--title", help="Título opcional del documento base (solo con --file)"
    )

    args = parser.parse_args()

    try:
        if args.all:
            upload_all_md_documents()
        else:
            upload_md_document(file_name=args.file, title=args.title)
    except Exception as e:
        print(f"❌ Error al subir el documento: {e}")


if __name__ == "__main__":
    main()
