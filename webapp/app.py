# app.py
import streamlit as st
import yaml
from langgraph_runner import run_prompt

from config.config import MAX_PROMPT_TOKENS
from modules.prompt_utils import build_chatml_messages, count_tokens, encoding

# Cargar opciones de interfaz desde archivo YAML
with open("config/ui_options.yaml", "r", encoding="utf-8") as f:
    ui_options = yaml.safe_load(f)

st.set_page_config(page_title="KoalaRoute", page_icon="🐨", layout="centered")

st.title("🐨 KoalaRoute 🐨")
st.markdown("Planifica tu aventura perfecta por Australia con inteligencia y estilo.")

# Entradas de usuario
st.markdown("## ✈️ Detalles del viaje")
user_query = st.text_input("¿Qué te gustaría hacer o visitar?")

col1, col2 = st.columns(2)
with col1:
    days = st.number_input("🗓️ Duración (en días)", min_value=1, max_value=7, value=3)
with col2:
    budget = st.selectbox("💰 Presupuesto", ui_options["presupuestos"])

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

# Construcción del prompt base
interest_str = ", ".join(interests) if interests else "cualquier tipo de actividad"
prompt_base = (
    f"Eres un planificador de viajes experto en Australia. "
    f"Quiero un itinerario día a día para un viaje de {days} días, "
    f"con presupuesto {budget.lower()}, viajando en formato {travel_type.lower()}, "
    f"centrado en los siguientes intereses: {interest_str}. "
    f"Detalles adicionales: "
)

# Tokens del sistema (prompt fijo)
base_messages = build_chatml_messages(prompt_base)
base_token_count = count_tokens(base_messages)

# Tokens del input del usuario
user_token_count = len(encoding.encode(user_query))
tokens_remaining = max(MAX_PROMPT_TOKENS - base_token_count, 0)
progress_ratio = (
    min(user_token_count / tokens_remaining, 1.0) if tokens_remaining > 0 else 1.0
)

# Visualización dinámica del uso de tokens del usuario
st.markdown("## 📏 Tokens disponibles para tu mensaje")
st.progress(
    progress_ratio,
    text=f"{user_token_count} / {tokens_remaining} tokens usados en tu mensaje",
)

# Prompt completo
full_prompt = prompt_base + user_query

# Botón de envío
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
                response = run_prompt(full_prompt)
                st.success("🗺️ Tu itinerario personalizado:")
                st.markdown(response, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
