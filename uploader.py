"""
Módulo para cargar documentos .md en Azure Cognitive Search con embeddings.

Este módulo:
- Lee el contenido de un archivo .txt especificado por el usuario.
- Genera su vector de embedding utilizando el vector_store existente.
- Envía el documento al índice vectorial en Azure Search.

Uso:
    python uploader.py --file info.md
    python uploader.py --all
"""

import argparse
import os
import re
import uuid
from typing import List, Optional, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from config.config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    DOCS_PATH,
    INDEX_NAME,
    SECTION_TO_CATEGORIES,
)
from modules.vector import vector_store


def split_markdown_sections(text: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """
    Extrae el título principal y divide un documento Markdown en secciones por encabezados de nivel 2 (##).

    Args:
        text (str): Contenido completo del archivo Markdown.

    Returns:
        Tuple[str | None, list[tuple[str, str]]]:
            - Título principal (extraído del primer encabezado de nivel 1).
            - Lista de tuplas con secciones (titulo, contenido).
    """
    # Extraer título principal (primer encabezado de nivel 1)
    title_match = re.search(r"^\s*#\s+(.*)", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else None

    # Dividir por encabezados de nivel 2
    pattern = r"(?:^|\n)(## .+?)(?=\n## |\Z)"  # captura cada sección ## ... hasta el siguiente ## o fin de texto
    matches = re.findall(pattern, text, flags=re.DOTALL)

    sections = []
    for match in matches:
        lines = match.strip().splitlines()
        section_title = lines[0].lstrip("#").strip()
        section_body = "\n".join(lines[1:]).strip()
        sections.append((section_title, section_body))

    return title, sections


def upload_md_document(file_name: str):
    """
    Carga un archivo .md al índice de Azure Search con sus secciones vectorizadas.

    El título del documento se extrae automáticamente del primer encabezado de nivel 1 (# Ciudad).

    Args:
        file_name (str): Nombre del archivo Markdown en DOCS_PATH.
    """
    file_path = os.path.join(DOCS_PATH, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"❌ Archivo no encontrado: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Obtener título y secciones desde el Markdown
    title, section_chunks = split_markdown_sections(content)

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
                "title": title,
                "section": section_title,
                "category": categories,
                "content": section_text,
                "content_vector": embedding,
                "source": f"{file_path}#section-{i}",
            }
        )

    result = search_client.upload_documents(documents=documents)
    print(f"✅ Subidas {len(documents)} secciones de '{title}'")
    return result


def upload_all_md_documents():
    """
    Sube todos los archivos .md encontrados en DOCS_PATH al índice de Azure Search.

    Itera sobre todos los ficheros .md en la carpeta especificada,
    ejecutando `upload_md_document()` sobre cada uno de ellos.
    """
    md_files = [
        f
        for f in os.listdir(DOCS_PATH)
        if f.endswith(".md") and f.lower() != "template.md"
    ]

    if not md_files:
        print("⚠️ No se encontraron archivos .md en la carpeta DOCS_PATH.")
        return

    for file_name in md_files:
        try:
            print(f"⬆️ Subiendo '{file_name}'...")
            upload_md_document(file_name)
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

    args = parser.parse_args()

    try:
        if args.all:
            upload_all_md_documents()
        else:
            upload_md_document(args.file)
    except Exception as e:
        print(f"❌ Error al subir el documento: {e}")


if __name__ == "__main__":
    main()
