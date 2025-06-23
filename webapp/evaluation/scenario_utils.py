# scenario_utils.py

import os

import streamlit as st
import yaml

from config.config import SCENARIOS_PATH


def get_available_scenarios():
    """
    Devuelve una lista con los nombres de los escenarios definidos
    dentro del archivo único de escenarios YAML.
    """

    if not os.path.exists(SCENARIOS_PATH):
        st.warning("No se encontró el archivo de escenarios.")
        return []

    try:
        with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
            scenarios = yaml.safe_load(f)
            return [s["name"] for s in scenarios if "name" in s]
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
