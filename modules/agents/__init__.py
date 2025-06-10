"""
agents

Este submódulo agrupa los agentes responsables de cada etapa del sistema RAG + LLM.

Incluye:

- RetrieverAgent: Encapsula la lógica de recuperación de contexto semántico desde un vector store (Azure Search).
- LLMAgent: Gestiona la interacción con el modelo de lenguaje para generar respuestas.
- ControllerAgent: Orquesta el flujo completo entre recuperación y generación de texto.

Uso típico:
    from modules.agents import ControllerAgent, RetrieverAgent, LLMAgent
"""

from .retriever_agent import RetrieverAgent
from .llm_agent import LLMAgent
from .controller_agent import ControllerAgent

__all__ = ["RetrieverAgent", "LLMAgent", "ControllerAgent"]
