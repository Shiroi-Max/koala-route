"""
Módulo de utilidades para la gestión de escenarios de prueba.

Este módulo proporciona funciones para interactuar con un único archivo YAML que contiene
la definición de múltiples escenarios de evaluación. Cada escenario representa un caso de uso
potencial para el sistema RAG (Recuperación Aumentada con Generación), incluyendo campos como
nombre, duración, intereses, entrada del usuario, documentos esperados y respuesta de referencia.

Funciones disponibles:
- `get_available_scenarios()`: Devuelve una lista con los nombres de todos los escenarios definidos.
- `load_scenario_by_name(name)`: Carga y devuelve un escenario específico a partir de su nombre.

El archivo YAML utilizado está ubicado en la ruta definida por `SCENARIOS_PATH`, y es compartido
por la interfaz de Streamlit para evaluar casos de prueba.

Dependencias:
- PyYAML para parseo del archivo YAML.
- Streamlit para mostrar errores y advertencias al usuario en la interfaz gráfica.
"""


import os
from typing import List

import streamlit as st
import yaml

from config.config import SCENARIOS_PATH


def get_available_scenarios() -> List[str]:
    """
    Devuelve una lista con los nombres de los escenarios definidos
    dentro del archivo único de escenarios YAML.

    Returns:
        List[str]: Lista de nombres de escenarios.
    """
    if not os.path.exists(SCENARIOS_PATH):
        st.warning("No se encontró el archivo de escenarios.")
        return []

    try:
        with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, list):
                return [s["name"] for s in data if "name" in s]
            else:
                st.error("El archivo de escenarios no tiene el formato esperado.")
                return []
    except Exception as e:
        st.error(f"Error cargando los escenarios: {e}")
        return []


def load_scenario_by_name(scenario_name: str) -> dict:
    """
    Busca y devuelve un escenario específico por su nombre dentro del archivo único.

    Args:
        scenario_name (str): Nombre del escenario.

    Returns:
        dict: Escenario encontrado, o {} si no existe.
    """

    try:
        with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
            scenarios = yaml.safe_load(f)
            for s in scenarios:
                if s.get("name") == scenario_name:
                    return s
        st.warning(f"No se encontró el escenario: {scenario_name}")
        return {}
    except Exception as e:
        st.error(f"Error cargando el escenario: {e}")
        return {}
