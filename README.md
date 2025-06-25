<p align="center">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="license" />
  <img src="https://img.shields.io/badge/Built%20with-Python%203.12-blue.svg" alt="python" />
  <img src="https://img.shields.io/badge/Powered%20by-Azure%20OpenAI%20%7C%20LangGraph%20%7C%20Streamlit-orange.svg" alt="powered by" />
</p>

<p align="center">
  <a href="docs/TFM_Utica_Maxim.pdf" download>
    <img src="https://img.shields.io/badge/📘%20Download%20TFM-TFM_Utica_Maxim.pdf-blue" alt="Download TFM"/>
  </a>
</p>

# 🐨 Koala Route 🐨

This project implements a **Retrieval-Augmented Generation (RAG)** system that:

- Uses Azure Cognitive Search as a vector store for efficient semantic retrieval.  
- Employs Azure OpenAI (GPT-3.5-Turbo) for response generation via LangGraph.  
- Integrates an agent orchestrator using LangGraph to modularly manage the retrieval and generation flow.  
- Uses Azure OpenAI's `text-embedding-3-large` model to convert documents and queries into vectors, ensuring full compatibility with the configured Azure Search index.  
- Provides a web interface built with Streamlit for user-friendly interaction and evaluation of test scenarios.  
- Supports token usage management and quota limits to optimize API call costs.  
- Uses `.env` for secure and centralized management of Azure credentials and endpoints.

---

## 🧠 Project Overview

**Title**: Retrieval-Augmented Generation with LLMs and Multi-Agent Orchestration for Travel Planning  
**Author**: Maxim Utica Babyak  
**Degree**: Master’s in Artificial Intelligence  
**University**: Universidad Alfonso X El Sabio (UAX)  
**Date**: June 2025  
**Language**: Spanish  

This project implements a complete system for semantic retrieval and text generation, evaluated under controlled scenarios, with a focus on efficiency, result quality, and modular agent-based orchestration.

You can read the full thesis here:  
📘 [TFM_Utica_Maxim.pdf](docs/TFM_Utica_Maxim.pdf)

---

## 🚀 Requirements

- Python **3.12**  
- Dependencies managed via `pyproject.toml`  
- A `.env` file configured with Azure credentials

---

## 📸 Preview

![alt text](images/preview.png)

---

## ⚡ Installation

### 1. Create and activate a virtual environment

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 2. Install project dependencies

```bash
pip install .
```

If you also want development tools:

```bash
pip install ".[dev]"
```

---

### 3. Create and configure your `.env` file

```ini
# 🔎 Azure Cognitive Search
AZURE_SEARCH_ENDPOINT=https://<your-endpoint>.search.windows.net
AZURE_SEARCH_KEY=<your-secret-key>

# 🤖 Azure OpenAI for generation
AZURE_OPENAI_ENDPOINT=https://<your-openai-endpoint>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT=<your-generation-deployment-name>

# 📐 Azure OpenAI for embeddings
AZURE_OPENAI_EMBEDDINGS_ENDPOINT=https://<your-openai-endpoint>.openai.azure.com/
AZURE_OPENAI_EMBEDDINGS_API_KEY=<your-embedding-api-key>
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=<your-embedding-deployment-name>
```

---

## 🧪 Usage

### Launch main web interface (travel planner)

```bash
streamlit run webapp/app.py
```

Allows planning trips with filters such as duration, budget, and interests, generating personalized itineraries.

---

### Launch test scenario evaluation interface

```bash
streamlit run webapp/app_test.py
```

Allows selecting predefined YAML scenarios, running the RAG system, and visualizing:

- The model-generated response (itinerary).  
- Documents retrieved from Azure Cognitive Search.  
- Evaluation metrics such as Adaptive Recall and Semantic Coherence to assess system performance.

---

### Upload documents to Azure Cognitive Search index

Documents must be stored in `DOCS_PATH`, a constant defined in `config.py`:

```bash
python uploader.py --file info.md
```

```bash
python uploader.py --all
```

---

### Delete documents from Azure Cognitive Search index

```bash
python deleter.py --id <document_id> <document_id_2> ...
```

```bash
python deleter.py --all
```

---

## 📂 Project Structure

```
koalaRoute/
├── config/
│   ├── config.py
│   ├── prompts.yaml
│   ├── test_cases.yaml
│   └── ui_options.yaml
├── data/
│   └── template.md
├── modules/
│   ├── agents/
│   │   ├── controller_agent.py
│   │   ├── llm_agent.py
│   │   └── retriever_agent.py
│   ├── graph/
│   │   ├── agent_state.py
│   │   └── graph.py
│   ├── llm.py
│   ├── prompt_utils.py
│   └── vector.py
├── webapp/
│   ├── evaluations/
│   │   ├── evaluator.py
│   │   └── scenario_utils.py
│   ├── app_test.py
│   ├── app.py
│   └── runner.py
├── main.py
├── uploader.py
├── deleter.py
├── .env
├── pyproject.toml
└── README.md
```

---

## ✅ Status

- Functional orchestration with LangGraph, integrating Azure OpenAI (GPT-3.5-Turbo) and Azure Cognitive Search.  
- Embeddings managed via Azure OpenAI, compatible with the configured vector index.  
- Streamlit web interface for scenario selection, result visualization, and generation control.  
- Automatic response evaluation using metrics such as Recall@k and semantic coherence.  
- Optimized token management with `tiktoken` to prevent overuse in model calls.  
- Sensitive variables and external configuration managed through `.env`.  
- Includes scripts and documentation for uploading, deleting, and managing documents in the vector index.

---

## 📌 Notes

- The system is prepared to easily switch between local LLM and Azure OpenAI.  
- Supports future extensions for itinerary saving, history tracking, and integration with FastAPI or Gradio.  
- It is recommended to use a virtual environment and avoid installing dependencies globally.  
- Adjust token limits and prompt sizes to optimize costs in Azure OpenAI.