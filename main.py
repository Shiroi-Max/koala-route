"""
Script principal para interactuar con el sistema RAG + LLM de forma continua usando POO y LangGraph.

- Recupera contexto desde Azure Cognitive Search usando embeddings (RetrieverAgent).
- Genera respuestas con el modelo LLM configurado (LLMAgent, ej. Mistral, Gemma...).
- Orquesta el flujo completo mediante ControllerAgent y LangGraph.
- Acepta múltiples preguntas consecutivas desde la consola.
- Finaliza escribiendo 'salir' o presionando Ctrl+C.

Requiere:
- modules.graph.graph.build_langgraph_controller_flow

Uso:
    python main.py
"""

from modules.graph.agent_state import AgentState
from modules.graph.graph import build_langgraph_controller_flow


def main():
    """
    Ejecuta una sesión interactiva usando el grafo LangGraph.
    """
    print("🧠 Sistema RAG + LLM (Escribe 'salir' para terminar)\n")

    # Construir grafo
    dialogue_manager = build_langgraph_controller_flow()

    try:
        while True:
            user_query = input("❓ Tu pregunta: ").strip()

            if user_query.lower() in {"salir", "exit", "quit"}:
                print("👋 Cerrando sesión.")
                break

            if not user_query:
                continue  # Ignorar entrada vacía

            # Ejecutar flujo LangGraph con estado inicial
            state = AgentState(input=user_query, response="")
            result = dialogue_manager.invoke(state)

            print("\n💬 Respuesta:\n", result.get("response"), "\n")

    except KeyboardInterrupt:
        print("\n👋 Sesión interrumpida manualmente.")
    except Exception as e:
        print(f"\n⚠️ Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
    main()
