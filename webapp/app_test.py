"""
Interfaz web para evaluación de escenarios RAG mediante Streamlit.

Este módulo define una aplicación Streamlit que permite seleccionar y ejecutar escenarios de prueba
predefinidos. Cada escenario contiene una consulta del usuario, información contextual y una respuesta esperada.
El flujo de evaluación consiste en:

1. Selección de un escenario desde un desplegable.
2. Construcción del prompt de entrada usando una plantilla y los parámetros del escenario.
3. Ejecución del sistema completo RAG (recuperación + generación) mediante `run_prompt`.
4. Evaluación de la respuesta generada frente a la esperada con métricas de recuperación y coherencia semántica.
5. Visualización del resultado, incluyendo:
   - Métricas de evaluación.
   - Respuesta generada (itinerario propuesto).
   - Documentos recuperados durante el proceso.

Componentes principales:
- `load_formatted_prompt`: Construye el prompt estructurado a partir de los datos del escenario.
- `run_prompt`: Ejecuta el grafo de LangGraph y devuelve la respuesta generada y los documentos utilizados.
- `Evaluator`: Calcula métricas como `Recall adaptativo` y `Coherencia Semántica`.

Requiere archivos de escenario YAML ubicados en el directorio correspondiente, accesibles mediante `get_available_scenarios`.

Uso:
Ejecutar `streamlit run webapp/app_test.py` desde la raíz del proyecto.
"""

import streamlit as st

from modules.prompt_utils import load_formatted_prompt
from webapp.evaluation.evaluator import Evaluator
from webapp.evaluation.scenario_utils import (
    get_available_scenarios,
    load_scenario_by_name,
)
from webapp.runner import run_prompt

st.set_page_config(page_title="KoalaTest", page_icon="🐨🛠️", layout="centered")
st.title("Evaluador de escenarios de prueba")

available_scenarios = get_available_scenarios()

if not available_scenarios:
    st.warning("No se encontraron archivos de escenario.")
else:
    selected_scenario = st.selectbox("Selecciona un escenario:", available_scenarios)

    if st.button("Ejecutar evaluación"):
        with st.spinner("Procesando escenario..."):
            # Cargar datos del escenario
            scenario_data = load_scenario_by_name(selected_scenario)

            # Construir prompt completo
            interest_str = (
                ", ".join(scenario_data["interests"])
                if scenario_data.get("interests")
                else "cualquier tipo de actividad"
            )
            full_prompt = (
                load_formatted_prompt(
                    "prompt_base",
                    days=scenario_data["duration_days"],
                    budget=scenario_data["budget"].lower(),
                    travel_type=scenario_data["travel_type"].lower(),
                    interests=interest_str,
                )
                + scenario_data["user_input"]
            )

            # Ejecutar el sistema
            generated_output = run_prompt(full_prompt)

            # Añadir resultados al escenario para evaluación
            generated_response = generated_output.get("generated_response", "")
            retrieved_docs = generated_output.get("retrieved_docs", [])

            scenario_data["retrieved_docs"] = retrieved_docs

            # Evaluar
            evaluator = Evaluator()
            result = evaluator.evaluate_scenario(scenario_data)

        st.success("✅ Evaluación completada.")
        st.subheader("📊 Resultado de la evaluación")
        st.json(result)

        st.subheader("📝 Respuesta generada (itinerario)")
        st.markdown(generated_response)

        if retrieved_docs:
            st.subheader("📚 Documentos recuperados")
            for doc in retrieved_docs:
                st.markdown(f"- `{doc.get('id')}`")
