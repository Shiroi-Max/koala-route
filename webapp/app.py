"""
Interfaz principal de la aplicación KoalaRoute utilizando Streamlit.

Esta aplicación permite a los usuarios generar itinerarios de viaje por Australia de forma personalizada,
mediante un sistema basado en LLM y recuperación de contexto (RAG). El usuario introduce sus preferencias
y el sistema construye un prompt estructurado para generar una respuesta coherente y contextualizada.

Características:
- Entrada de texto libre para deseos del viaje.
- Selección de duración, presupuesto, tipo de viaje e intereses.
- Visualización del uso de tokens antes de enviar.
- Generación del itinerario mediante `run_prompt` (flujo LLM + RAG).

Uso:
Ejecutar `streamlit run webapp/app.py` desde la raíz del proyecto.
"""

import streamlit as st
import yaml
from webapp.runner import run_prompt

from config.config import MAX_PROMPT_TOKENS, UI_OPTIONS_PATH
from modules.prompt_utils import encoding, load_formatted_prompt

# ---------- CARGA DE OPCIONES DE UI DESDE YAML ----------
with open(UI_OPTIONS_PATH, "r", encoding="utf-8") as f:
    ui_options = yaml.safe_load(f)

# ---------- CONFIGURACIÓN GENERAL DE LA PÁGINA ----------
st.set_page_config(page_title="KoalaRoute", page_icon="🐨", layout="centered")

st.title("🐨 KoalaRoute 🐨")
st.markdown("Planifica tu aventura perfecta por Australia con inteligencia y estilo.")

# ---------- ENTRADA DE DETALLES DEL VIAJE ----------
st.markdown("## ✈️ Detalles del viaje")
user_query = st.text_input("¿Qué te gustaría hacer o visitar?")

col1, col2 = st.columns(2)
with col1:
    days = st.number_input("🗓️ Duración (en días)", min_value=1, max_value=7, value=3)
with col2:
    budget = st.selectbox("💰 Presupuesto", ui_options["presupuestos"])

# ---------- PREFERENCIAS ADICIONALES ----------
st.markdown("## 🌟 Preferencias del viaje")
col3, col4 = st.columns(2)
with col3:
    travel_type = st.selectbox("👥 Tipo de viaje", ui_options["tipos_viaje"])
with col4:
    interests = st.multiselect(
        "🧽 Intereses",
        ui_options["intereses"],
        default=["Naturaleza"],
    )

# ---------- CÁLCULO Y VISUALIZACIÓN DE TOKENS ----------
user_token_count = len(encoding.encode(user_query))
tokens_remaining = max(MAX_PROMPT_TOKENS, 0)
progress_ratio = (
    min(user_token_count / tokens_remaining, 1.0) if tokens_remaining > 0 else 1.0
)

st.markdown("## 📏 Tokens disponibles para tu mensaje")
st.progress(
    progress_ratio,
    text=f"{user_token_count} / {tokens_remaining} tokens usados en tu mensaje",
)

# ---------- CONSTRUCCIÓN DEL PROMPT ----------
interest_str = ", ".join(interests) if interests else "cualquier tipo de actividad"

# Prompt completo con sistema + entrada del usuario
full_prompt = (
    load_formatted_prompt(
        "prompt_base",
        days=days,
        budget=budget.lower(),
        travel_type=travel_type.lower(),
        interests=interest_str,
    )
    + user_query
)

# ---------- BOTÓN Y LÓGICA DE GENERACIÓN ----------
if st.button("🦘 Generar itinerario"):
    if not user_query.strip():
        st.warning("Por favor, describe tu viaje.")
    elif not isinstance(days, int) or not 1 <= days <= 30:
        st.error("❌ La duración del viaje debe estar entre 1 y 30 días.")
    elif user_token_count > tokens_remaining:
        st.error(
            f"❌ Tu mensaje usa {user_token_count} tokens, pero solo puedes usar un máximo de {tokens_remaining} "
            "tokens debido al espacio reservado para instrucciones del sistema. Reduce la longitud o complejidad del mensaje."
        )
    else:
        with st.spinner("⛺ Trazando tu ruta ideal..."):
            try:
                response = run_prompt(full_prompt)["generated_response"]
                st.success("🗺️ Tu itinerario personalizado:")
                st.markdown(response, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
