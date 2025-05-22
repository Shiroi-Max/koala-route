"""
Script principal para interactuar con el sistema RAG + LLM de forma continua.

- Recupera contexto desde Azure Cognitive Search usando embeddings.
- Genera respuestas con el modelo LLM configurado (ej. Gemma, Mistral, etc.).
- Acepta múltiples preguntas consecutivas desde la consola.
- Finaliza escribiendo 'salir' o presionando Ctrl+C.

Requiere:
- `controller_agent` desde agents.py

Uso:
    python main.py
"""

from modules.agents import controller_agent


def main():
    """
    Ejecuta una sesión interactiva para preguntas al sistema RAG.
    """
    print("🧠 Sistema RAG + LLM (Escribe 'salir' para terminar)\n")

    try:
        while True:
            user_query = input("❓ Tu pregunta: ").strip()

            if user_query.lower() in {"salir", "exit", "quit"}:
                print("👋 Cerrando sesión.")
                break

            if not user_query:
                continue  # Ignorar entrada vacía

            respuesta = controller_agent(user_query=user_query)

            print("\n💬 Respuesta:\n", respuesta, "\n")

    except KeyboardInterrupt:
        print("\n👋 Sesión interrumpida manualmente.")


if __name__ == "__main__":
    main()
