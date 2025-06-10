# app.py
import streamlit as st
from langgraph_runner import run_prompt
from modules.prompt_utils import build_chatml_messages, count_tokens
from config.config import MAX_PROMPT_TOKENS

st.set_page_config(page_title="KoalaRoute", page_icon="🧭", layout="centered")

st.title("🐨 KoalaRoute")
st.markdown("Planifica tu aventura perfecta por Australia con inteligencia y estilo.")

# Entradas de usuario
st.markdown("## ✈️ Detalles del viaje")
user_query = st.text_input("¿Qué te gustaría hacer o visitar?")

col1, col2 = st.columns(2)
with col1:
    days = st.number_input("🗓️ Duración (en días)", min_value=1, max_value=30, value=7)
with col2:
    budget = st.selectbox("💰 Presupuesto", ["Económico", "Medio", "Alto"])

st.markdown("## 🎯 Preferencias del viaje")
col3, col4 = st.columns(2)
with col3:
    travel_type = st.selectbox(
        "👥 Tipo de viaje", ["Individual", "Pareja", "Familia", "Mochilero"]
    )
with col4:
    interests = st.multiselect(
        "🧭 Intereses",
        ["Naturaleza", "Playas", "Cultura", "Aventura", "Gastronomía"],
        default=["Naturaleza"],
    )

# Construcción del prompt
interest_str = ", ".join(interests) if interests else "cualquier tipo de actividad"
full_prompt = (
    f"Eres un planificador de viajes experto en Australia. "
    f"Quiero un itinerario **día a día** para un viaje de {days} días, "
    f"con presupuesto **{budget.lower()}**, viajando en formato **{travel_type.lower()}**, "
    f"centrado en los siguientes intereses: **{interest_str}**. "
    f"Detalles adicionales: {user_query}."
)

# Mensajes en formato ChatML
messages = build_chatml_messages(full_prompt)
token_count = count_tokens(messages)
progress_ratio = min(token_count / MAX_PROMPT_TOKENS, 1.0)

# Visualización del uso de tokens
st.markdown("## 📏 Uso de tokens")
st.progress(
    progress_ratio, text=f"{token_count} / {MAX_PROMPT_TOKENS} tokens utilizados"
)

# Botón de envío
if st.button("🦘 Generar itinerario"):
    if not user_query.strip():
        st.warning("Por favor, describe tu viaje.")
    elif token_count > MAX_PROMPT_TOKENS:
        st.error(
            f"❌ Tu entrada tiene {token_count} tokens, lo cual excede el máximo permitido ({MAX_PROMPT_TOKENS}). "
            "Reduce la longitud o complejidad del mensaje."
        )
    else:
        with st.spinner("⛺ Trazando tu ruta ideal..."):
            try:
                response = run_prompt(full_prompt)
                st.success("🗺️ Tu itinerario personalizado:")
                st.markdown(response, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
