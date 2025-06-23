"""
Módulo de evaluación de escenarios para sistemas RAG.

Este módulo define la clase `Evaluator`, encargada de medir la calidad de las respuestas generadas
por el sistema, tanto a nivel de recuperación como de generación. Es utilizado para comparar
automáticamente las salidas del sistema con respuestas de referencia en escenarios de prueba definidos.

Funcionalidades:
- `recall_at_k`: Evalúa la calidad de la recuperación mediante la métrica de Recall adaptativo,
  ajustando dinámicamente el valor de `k` según el número de documentos recuperados.
- `semantic_similarity`: Calcula la similitud semántica entre la respuesta generada y una respuesta
  de referencia utilizando embeddings y similitud coseno.
- `evaluate_scenario`: Evalúa un escenario completo, devolviendo un diccionario con las métricas
  obtenidas.

Requisitos:
- El escenario debe incluir los campos `generated_response` y `reference_response` para evaluar
  la coherencia semántica.
- Si `evaluate` es `True`, también se requieren `expected_relevant_docs` y `retrieved_docs`.

Dependencias:
- `embeddings`: Objeto de embeddings compartido, inicializado desde `modules.vector`.
- `K_EVAL_THRESHOLD`: Proporción configurable de documentos usados como top-K en el cálculo de recall.
"""


from math import ceil
from typing import Dict, List, Set

from sklearn.metrics.pairwise import cosine_similarity

from config.config import K_EVAL_THRESHOLD
from modules.vector import embeddings


class Evaluator:
    def __init__(self):
        """
        Inicializa el evaluador utilizando el objeto de embeddings definido en `modules.vector`.
        Este objeto permite convertir texto en vectores para calcular similitud semántica.
        """
        self.embeddings = embeddings

    def recall_at_k(self, relevant_docs: Set[str], retrieved_docs: List[Dict]) -> float:
        """
        Calcula el recall adaptativo entre los documentos relevantes esperados y los recuperados.

        El valor de k se ajusta dinámicamente según la proporción `K_EVAL_THRESHOLD` y la longitud
        de la lista de documentos recuperados.

        Args:
            relevant_docs (Set[str]): Conjunto de identificadores de documentos relevantes esperados.
            retrieved_docs (List[Dict]): Lista de documentos recuperados por el sistema.

        Returns:
            float: Valor de recall@k, entre 0.0 y 1.0. Si no hay documentos relevantes, retorna 0.0.
        """
        k = max(1, ceil(K_EVAL_THRESHOLD * len(retrieved_docs)))
        top_k_ids = [doc["id"] for doc in retrieved_docs[:k]]
        relevant_retrieved = sum(1 for doc_id in top_k_ids if doc_id in relevant_docs)
        total_relevant = len(relevant_docs)
        return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

    def thematic_coverage(self, interests: List[str], retrieved_docs: List[Dict]) -> float:
        """
        Calcula la cobertura temática de los documentos recuperados respecto a los intereses del usuario.

        Esta métrica evalúa qué proporción de los intereses proporcionados por el usuario
        están presentes en las categorías de los documentos recuperados. Se normaliza todo el texto
        a minúsculas para una comparación robusta.

        Si no se especifican intereses, se asume cobertura total (1.0).

        Args:
            interests (List[str]): Lista de intereses definidos por el usuario.
            retrieved_docs (List[Dict]): Lista de documentos recuperados, cada uno con una clave "category"
                                        que contiene una lista de etiquetas temáticas.

        Returns:
            float: Porcentaje de intereses cubiertos por las categorías de los documentos (entre 0.0 y 1.0).
        """
        if not interests:
            return 1.0

        # Normalizar intereses y categorías
        interests_normalized = {i.strip().lower() for i in interests}
        retrieved_categories = set()

        for doc in retrieved_docs:
            for cat in doc.get("category", []):
                retrieved_categories.add(cat.strip().lower())

        matches = sum(1 for i in interests_normalized if i in retrieved_categories)
        return matches / len(interests_normalized) if interests_normalized else 0.0

    def semantic_similarity(self, generated: str, reference: str) -> float:
        """
        Calcula la similitud semántica entre una respuesta generada y una respuesta de referencia,
        usando embeddings y similitud coseno.

        Args:
            generated (str): Respuesta generada por el sistema.
            reference (str): Respuesta de referencia esperada.

        Returns:
            float: Valor de similitud coseno entre los embeddings, entre -1.0 y 1.0.
        """
        emb_generated = self.embeddings.embed_query(generated)
        emb_reference = self.embeddings.embed_query(reference)
        return cosine_similarity([emb_generated], [emb_reference])[0][0]

    def evaluate_scenario(self, scenario: Dict) -> Dict:
        """
        Evalúa un escenario de prueba comparando los resultados generados con los esperados.

        Esta función calcula varias métricas para analizar la calidad de la recuperación y la generación:
        - Recall adaptativo: porcentaje de documentos esperados que se encuentran entre los recuperados (ajustado por k dinámico).
        - Cobertura Temática: proporción de intereses del usuario que están representados en las categorías de los documentos recuperados.
        - Coherencia Semántica: similitud entre la respuesta generada y una respuesta de referencia.

        La evaluación de recuperación solo se ejecuta si el campo `evaluate` está presente y es True.

        Args:
            scenario (Dict): Diccionario con la información del escenario

        Returns:
            Dict: Diccionario con las métricas obtenidas para el escenario.
        """

        results = {"Escenario": scenario["name"]}

        if scenario.get("evaluate", True):
            # Métrica de recall adaptativo
            recall = self.recall_at_k(
                set(scenario["expected_relevant_docs"]), scenario["retrieved_docs"]
            )
            results["Recall adaptativo"] = round(recall, 2)

            # Métrica de cobertura temática
            coverage = self.thematic_coverage(
                scenario.get("interests", []), scenario["retrieved_docs"]
            )
            results["Cobertura Temática"] = round(coverage, 2)

            # Métrica de coherencia semántica
            if "reference_response" in scenario and "generated_response" in scenario:
                coherence = self.semantic_similarity(
                    scenario["generated_response"], scenario["reference_response"]
                )
                results["Coherencia Semántica"] = round(coherence, 2)

        return results
