# 🧠 RAG + LLM con LangChain, Hugging Face y Azure AI Search

Este proyecto implementa un sistema de Recuperación y Generación aumentada (RAG) que:

- Usa Azure Cognitive Search como base vectorial.
- Aplica embeddings locales con `sentence-transformers`.
- Utiliza un modelo LLM local (ej. Gemma 2B, Mistral 7B) para generar respuestas.
- Funciona con PyTorch y aceleración GPU (CUDA 12.1).

---

## 🚀 Requisitos

- Python **3.12**
- GPU compatible con **CUDA 12.1**
- Dependencias gestionadas con `pyproject.toml`

---

## ⚡ Instalación

### 1. Crea y activa un entorno virtual

En Linux/macOS:

    python3 -m venv .venv
    source .venv/bin/activate

En Windows:

    python -m venv .venv
    .venv\Scripts\activate

---

### 2. Instala PyTorch con soporte GPU (CUDA 12.1)

> 🛠️ Este paso es **obligatorio** antes de instalar el resto.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Instala el resto del proyecto

```bash
pip install .
```

Esto instalará LangChain, Azure, Hugging Face, sentence-transformers, widgets y más.

---

## 🧪 Uso

### Iniciar una sesión interactiva

```bash
python main.py
```

Puedes hacer preguntas consecutivas al modelo.  
Escribe `salir` para cerrar la sesión.

---

### Subir documentos al índice

```bash
python uploader.py --file info.txt --title "LangChain Overview"
```

El archivo debe estar en la carpeta `data/`.

---

### Eliminar documentos por ID
```bash
    python deleter.py --id <id_documento>
```
---

## 📂 Estructura del proyecto

    rag_llm_project/
    ├── config/
    │   └── config.py
    ├── data/
    │   └── info.txt
    ├── modules/
    │   ├── agents.py
    │   ├── embeddings.py
    │   ├── llm.py
    ├── main.py
    ├── uploader.py
    ├── deleter.py
    ├── pyproject.toml
    └── README.md

---

## ✅ Estado

- ✅ Entrenamiento y generación local funcionando
- ✅ Compatible con LangChain 0.2.x y `langchain-huggingface`
- ✅ Indexación y búsqueda en Azure Cognitive Search
- ✅ Soporte para modelos ligeros (DistilBERT, Flan-T5) o potentes (Gemma, Mistral)

---

## 📌 Notas

- Puedes adaptar fácilmente para servirlo con FastAPI, Streamlit o Gradio.
- Admite cualquier modelo Hugging Face compatible con `transformers.pipeline`.