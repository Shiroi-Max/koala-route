"""
Definición del esquema de estado para el grafo LangGraph.

Este módulo define `AgentState`, un diccionario tipado que representa el estado compartido
entre los agentes durante la ejecución del flujo RAG. Es utilizado por LangGraph como
estructura base para pasar información entre nodos.

Los campos son opcionales (`total=False`) para permitir flexibilidad durante el flujo.

Campos:
- input (str): Entrada original del usuario.
- response (str): Mensaje de respuesta generado o contexto recuperado (en formato ChatML o texto plano).
- last_node (str): Último nodo ejecutado en el flujo (por ejemplo, "consulta" o "llm").
- retrieved_docs (List[str]): Lista de identificadores de documentos relevantes recuperados,
  con el formato "título#sección".
"""


from typing import List, TypedDict


class AgentState(TypedDict, total=False):
    """
    Estado compartido entre los agentes del grafo LangGraph.

    Atributos:
        input (str): Entrada del usuario.
        response (str | None): Respuesta generada o contexto recuperado.
        last_node (str | None): Nombre del último nodo ejecutado.
        retrieved_docs (List[str] | None): Documentos relevantes recuperados en formato "título#sección".
    """

    input: str
    response: str = None
    last_node: str = None
    retrieved_docs: List[str] = None

