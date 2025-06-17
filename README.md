# 🧠 RAG + LLM con LangChain, Hugging Face y Azure AI Search

Este proyecto implementa un sistema de Recuperación y Generación Aumentada (RAG) que:

- Usa Azure Cognitive Search como base vectorial.  
- Aplica embeddings locales con `sentence-transformers` y `torch` (CPU/GPU).  
- Utiliza Azure OpenAI (GPT-3.5-Turbo) para generación de respuestas mediante LangGraph.  
- Integra un orquestador de agentes con LangGraph para gestionar flujo de recuperación y generación.  
- Dispone de interfaz web con Streamlit para interacción amigable.  
- Soporta gestión de tokens y límites para optimizar costos.  
- Utiliza `.env` para gestión segura de credenciales.  
- Soporta aceleración con `accelerate` para aprovechar GPU en embeddings.

---

## 🚀 Requisitos

- Python **3.12**  
- GPU compatible con **CUDA 12.1** (recomendado para embeddings y aceleración)  
- Dependencias gestionadas con `pyproject.toml`  
- Archivo `.env` configurado con credenciales de Azure

---

## ⚡ Instalación

### 1. Crea y activa un entorno virtual

En Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 2. Instala PyTorch con soporte GPU (CUDA 12.1)

> 🛠️ Este paso es **obligatorio** antes de instalar el resto.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Instala el resto de dependencias del proyecto

```bash
pip install .
```

Si quieres también herramientas de desarrollo y webapp:

```bash
pip install ".[dev]"
```
### 4. Crea y configura tu archivo .env

```ini
AZURE_SEARCH_ENDPOINT=https://<tu-endpoint>.search.windows.net
AZURE_SEARCH_KEY=<tu-clave-secreta>
AZURE_OPENAI_API_KEY=<tu-api-key>
AZURE_OPENAI_ENDPOINT=https://<openai-endpoint>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=koalaroute-gpt35
```

---

## 🧪 Uso

### Ejecutar interfaz por consola

```bash
python main.py
```

Puedes hacer preguntas consecutivas al modelo.  
Escribe `salir` para cerrar la sesión.

---

### Ejecutar interfaz web con Streamlit

```bash
streamlit run webapp/app.py
```

Permite planificar viajes con filtros, duración, presupuesto e intereses.

---

### Subir documentos al índice Azure Cognitive Search
Los documentos deben estar almacenados en ``DOCS_PATH``, constante definida en ``config.py``

```bash
python uploader.py --file info.txt --title "LangChain Overview"
```

```bash
python uploader.py --all
```

---

### Eliminar documentos
```bash
python deleter.py --id <id_documento> <id_documento_2> ...
```

```bash
python deleter.py --all
```
---

## 📂 Estructura del proyecto

```arduino
koalaRoute/
├── config/
│   ├── config.py
|   ├── prompts.yaml
|   └── ui_options.yaml
├── data/
│   └── template.md
├── modules/
│   ├── agents/
|   |   ├── controller_agent.py
|   |   ├── llm_agent.py
|   |   └── retriever_agent.py
│   ├── graph/
|   |   ├── agent_state.py
|   |   └── graph.py
│   ├── llm.py
│   ├── prompt_utils.py
│   └── vector.py
├── webapp/
│   ├── app.py
|   └── langgraph_runner.py 
├── main.py
├── uploader.py
├── deleter.py
├── pyproject.toml
├── .env
└── README.md
```

---

## ✅ Estado

- Orquestación con LangGraph funcionando con Azure OpenAI y Azure Cognitive Search.
- Embeddings con sentence-transformers acelerados con torch y accelerate.
- Interfaz web Streamlit con filtros avanzados y control de tokens.
- Manejo de tokens con tiktoken para evitar sobrecostos.
- Configuración segura con .env.
- Documentación y scripts para subir/eliminar documentos en índice.

---

## 📌 Notas

- El sistema está preparado para cambiar fácilmente entre LLM local y Azure OpenAI.
- Soporta extensiones para guardar itinerarios, historial, y futuras integraciones con FastAPI o Gradio.
- Se recomienda usar entorno virtual y evitar instalar dependencias globalmente.
- Ajusta límites de tokens y prompts para optimizar costos en Azure OpenAI.