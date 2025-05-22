"""
Módulo para inicializar el modelo LLM final usado para generación de texto.

Este módulo expone:
- `llm`: un objeto `HuggingFacePipeline` de LangChain que encapsula el modelo LLM
         cargado desde Hugging Face Transformers, listo para usar en agentes.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config.config import LLM_MODEL_ID

# ---------- TOKENIZER ----------
# Se carga el tokenizer asociado al modelo LLM (por ejemplo, Gemma 2B)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, use_fast=True)

# ---------- MODELO LLM ----------
# Se carga el modelo de lenguaje para generación causal
# - `device_map="auto"` asigna el modelo automáticamente a GPU/CPU según disponibilidad
# - `torch_dtype="auto"` selecciona automáticamente el tipo de tensor óptimo (ej. float16)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID, device_map="auto"
)

# ---------- PIPELINE ----------
# Se crea un pipeline de generación de texto
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.7,
    max_new_tokens=512,
    do_sample=True
)

# ---------- LLM WRAPPER ----------
# Se envuelve el pipeline en un objeto de LangChain
# llm = HuggingFacePipeline(pipeline=pipe)
