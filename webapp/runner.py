"""
Ejecutor principal del grafo LangGraph.

Este módulo sirve como punto de entrada para interactuar con el flujo de agentes definido
en el grafo de LangGraph. Permite ejecutar una única interacción completa con el sistema RAG,
recibiendo una entrada del usuario y devolviendo tanto la respuesta generada como los
documentos relevantes recuperados.

Responsabilidades:
- Instancia el grafo mediante `build_langgraph_controller_flow`.
- Expone la función `run_prompt`, que toma una consulta del usuario y devuelve un diccionario
  con la respuesta final generada por el modelo y la lista de documentos utilizados.

La gestión de errores se realiza mediante trazas impresas, facilitando la depuración
en entornos de desarrollo local.
"""


import traceback

from modules.graph.agent_state import AgentState
from modules.graph.graph import build_langgraph_controller_flow

# Construcción del grafo de agentes (controlador + retriever + LLM)
dialogue_manager = build_langgraph_controller_flow()


def run_prompt(user_query: str) -> str:
    """
    Ejecuta una única interacción con el grafo de agentes a partir de una consulta del usuario.

    Este flujo sigue la lógica:
        input del usuario → recuperación opcional → generación con LLM → respuesta final

    Args:
        user_query (str): Texto introducido por el usuario (consulta o petición de itinerario).

    Returns:
        dict: Diccionario con los siguientes campos:
            - "generated_response": Respuesta generada por el modelo.
            - "retrieved_docs": Lista de identificadores de documentos recuperados (formato "título#sección").

    Raises:
        RuntimeError: Si ocurre algún error durante la ejecución del grafo.
    """
    try:
        # Crear el estado inicial con la entrada del usuario
        state = AgentState(input=user_query, response="")

        # Ejecutar el grafo
        result = dialogue_manager.invoke(state)

        # Retornar la respuesta generada
        return {
            "generated_response": result.get("response", ""),
            "retrieved_docs": result.get("retrieved_docs", []),
        }

    except Exception as e:
        print("💥 ERROR ejecutando el grafo:")
        traceback.print_exc()
        raise RuntimeError("Fallo en el grafo") from e
