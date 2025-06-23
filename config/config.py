"""
Configuración central del proyecto KoalaRoute (RAG + LLM + Azure).

Este módulo define:
- Las credenciales necesarias para conectarse a Azure Cognitive Search y Azure OpenAI.
- Las rutas, índices y nombres de modelo utilizados por los agentes del sistema.
- Límites de tokens para controlar el coste y optimizar el uso del modelo.
- Parámetros de recuperación semántica y filtrado de chunks.
"""

import os

from dotenv import load_dotenv

# ===============================
# 🔁 LECTURA DE VARIABLES DE ENTORNO
# ===============================
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]


AZURE_OPENAI_EMBEDDINGS_ENDPOINT = os.environ["AZURE_OPENAI_EMBEDDINGS_ENDPOINT"]
AZURE_OPENAI_EMBEDDINGS_API_KEY = os.environ["AZURE_OPENAI_EMBEDDINGS_API_KEY"]
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]

# ===============================
# 📁 PARÁMETROS GENERALES
# ===============================
DOCS_PATH = "data"  # Ruta local con documentos fuente (.md)
INDEX_NAME = "docs"  # Nombre del índice vectorial en Azure Search
UI_OPTIONS_PATH = "config/ui_options.yaml"  # Ruta al archivo de opciones de la interfaz
PROMPT_PATH = "config/prompts.yaml"  # Ruta al archivo con prompts del sistema
SCENARIOS_PATH = "config/test_cases.yaml"  # Ruta al archivo con escenarios de prueba

# ===============================
# 🧠 MODELOS UTILIZADOS
# ===============================
API_VERSION_LLM = "2024-02-15-preview"  # Versión de la API de Azure OpenAI
API_VERSION_EMBEDDINGS = "2024-02-01"  # Versión de la API de embeddings
ENCODING_NAME = "cl100k_base"  # Codificación de tokens para compatibilidad con GPT

# ===============================
# 🎯 LÍMITES DE TOKENS (para economizar y cumplir cuotas)
# ===============================
MAX_PROMPT_TOKENS = 512  # Máx tokens permitidos en el prompt (entrada)
MAX_COMPLETION_TOKENS = 850  # Máx tokens generados por la respuesta

# ===============================
# 🌡️ PARÁMETROS DE GENERACIÓN DE RESPUESTAS
# ===============================
TEMPERATURE = 0.4  # Control de aleatoriedad en la generación de respuestas

# ===============================
# 🔍 PARÁMETROS DEL RETRIEVER AGENT
# ===============================
RETRIEVER_K = 15  # Máx documentos devueltos por búsqueda semántica
SIMILARITY_THRESHOLD = (
    0.4  # Umbral mínimo de similitud para considerar un chunk relevante
)

# ===============================
# 📊 PARÁMETROS DE EVALUACIÓN
# ===============================
K_EVAL_THRESHOLD = 0.6  # Porcentaje mínimo de documentos relevantes para evaluación

# ===============================
# 🧩 MAPEO DE SECCIONES A CATEGORÍAS (para filtrado por interés)
# Este diccionario vincula los encabezados de los .md con intereses del usuario.
# ===============================
SECTION_TO_CATEGORIES = {
    # 🏖️ Información general sobre la ciudad y sus características
    "Descripción General": [
        "Naturaleza",
        "Playas",
        "Cultura",
        "Aventura",
        "Gastronomía",
        "Festivales y eventos",
        "Turismo urbano",
        "Clima agradable",
        "Viaje económico",
        "Transporte público",
        "Historia y curiosidades",
    ],
    # 🌤️ Información sobre el clima general de la ciudad
    "Clima": ["Clima agradable"],
    # 🏞️ Lugares turísticos representativos: naturaleza, cultura, aventura
    "Principales Atracciones": ["Naturaleza", "Aventura", "Cultura", "Turismo urbano"],
    # 🚍 Opciones de movilidad dentro de la ciudad
    "Transporte": ["Transporte público", "Viaje económico"],
    # 🛏️ Información sobre alojamiento accesible o adecuado según perfil
    "Alojamiento": ["Viaje económico", "Familia", "Mochilero"],
    # 🍲 Recomendaciones gastronómicas y cultura culinaria
    "Comida y Bebida": ["Gastronomía"],
    # 🎉 Actividades culturales y festividades locales
    "Eventos y Festivales": ["Festivales y eventos", "Cultura"],
    # 🧳 Recomendaciones prácticas y tips para viajeros
    "Consejos Útiles": ["Aventura", "Viaje económico", "Transporte público"],
    # 🕵️ Datos históricos o llamativos sobre la ciudad
    "Datos Curiosos": ["Historia y curiosidades"],
}
